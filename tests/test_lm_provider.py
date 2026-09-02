"""Unit tests for the cross-provider fallback task LM. No network: a stub
`litellm` module answers per model string, so every provider's behaviour --
a 402, a hang, a recovery, an outage -- is scripted rather than waited for.

The contract these pin is `docs/provider_fallback.md` §1:

  C1  one pinned model, several providers, never a different model
  C2  every model id in MODEL_ROUTES and nowhere else
  C3  one preference order, covers derived from it
  C4  per-call divert on provider errors only
  C5  a sustained outage costs O(1), not O(calls)
  C6  retry budget matched to the cover
  C7  configurable by environment, defaulted in code
  C8  the wiring lives outside the package CodeEvolver evolves

Run with: python -m pytest tests/test_lm_provider.py
"""

from __future__ import annotations

import pathlib
import sys
import types

import litellm
import openai
import pytest

from src import lm_provider
from src.lm_provider import (
    COVER_MAX_ATTEMPTS,
    DEFAULT_FALLBACK,
    DEFAULT_PROVIDER,
    GMI_API_BASE,
    MAX_ATTEMPTS,
    MODEL_ROUTES,
    PROVIDER_PREFERENCE,
    READ_GAP_TIMEOUT_S,
    REASONING_EFFORT,
    SOLO_MAX_ATTEMPTS,
    TASK_MODEL,
    CallStats,
    ProviderBreaker,
    Route,
    TaskLM,
    build_task_lm,
    route_for,
    should_fallback,
)

# Every model string comes from the one routing table -- the tests read it the
# same way the module does, so a model swap stays a one-row edit (C2).
GMI_MODEL = MODEL_ROUTES[TASK_MODEL]["gmi"].model
DEEPINFRA_MODEL = MODEL_ROUTES[TASK_MODEL]["deepinfra"].model
DEEPSEEK_MODEL = MODEL_ROUTES[TASK_MODEL]["deepseek"].model

MESSAGES = [{"role": "user", "content": "review this paper"}]

GMI_402_TEXT = (
    "OpenAIException - Error code: 402 - "
    "{'error': 'Insufficient balance', 'reason': 'model_access_denied'}"
)


def _api_error(status_code: int = 402, text: str = GMI_402_TEXT) -> litellm.APIError:
    return litellm.APIError(
        status_code=status_code, message=text, llm_provider="openai", model=GMI_MODEL
    )


def _chunks(text: str, size: int = 40):
    """`text` as a stream of delta chunks, the shape litellm yields."""

    class _Delta:
        def __init__(self, c):
            self.content = c

    class _Choice:
        def __init__(self, c):
            self.delta = _Delta(c)

    class _Chunk:
        def __init__(self, c):
            self.choices = [_Choice(c)]

    return [_Chunk(text[i : i + size]) for i in range(0, len(text), size)]


class FakeProviders:
    """A stub `litellm.completion` that answers per model string.

    Configure `fake.gmi` / `fake.deepinfra` / `fake.deepseek` with a string (the
    streamed response), an exception instance (raised), or a LIST of either --
    consumed one per attempt, the last entry repeating, which is how a
    "fails once then recovers" provider is scripted.
    """

    def __init__(self, gmi="from-gmi", deepinfra="from-deepinfra", deepseek="from-deepseek"):
        self.gmi = gmi
        self.deepinfra = deepinfra
        self.deepseek = deepseek
        self.calls: list[dict] = []

    @property
    def providers_called(self) -> list[str]:
        return [call["model"] for call in self.calls]

    def _behaviour(self, model):
        attr = {GMI_MODEL: "gmi", DEEPINFRA_MODEL: "deepinfra", DEEPSEEK_MODEL: "deepseek"}[model]
        value = getattr(self, attr)
        if isinstance(value, list):
            return value.pop(0) if len(value) > 1 else value[0]
        return value

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        behaviour = self._behaviour(kwargs["model"])
        if isinstance(behaviour, BaseException):
            raise behaviour
        return _chunks(behaviour)


@pytest.fixture(autouse=True)
def fresh_breakers():
    """Provider health is process-global; no test may inherit another's."""
    lm_provider.reset_breakers()
    yield
    lm_provider.reset_breakers()


@pytest.fixture(autouse=True)
def keys(monkeypatch):
    monkeypatch.delenv("GMI_CLOUD_API_KEY", raising=False)
    monkeypatch.setenv("GMI_API_KEY", "test-gmi-key")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "test-deepinfra-key")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-deepseek-key")
    # The routing env vars must not leak in from the shell running pytest.
    for var in ("LM_MODEL", "LM_PROVIDER", "LM_FALLBACK", "LM_BREAKER",
                "LM_BREAKER_FAILURES", "LM_BREAKER_COOLDOWN"):
        monkeypatch.delenv(var, raising=False)


@pytest.fixture
def fake(monkeypatch):
    """Install a FakeProviders as the `litellm` module the LM calls."""
    fp = FakeProviders()
    stub = types.ModuleType("litellm")
    stub.completion = fp.completion
    # The classifier resolves these by name off whatever `litellm` is installed;
    # a stub without them would make every error look divertible.
    for name in lm_provider.NON_FALLBACK_ERROR_NAMES:
        setattr(stub, name, getattr(litellm, name))
    monkeypatch.setitem(sys.modules, "litellm", stub)
    return fp


# ---------------------------------------------------------------------------
# C4 -- per-call divert, on provider errors only
# ---------------------------------------------------------------------------


def test_construction_pins_both_routes_to_the_same_model():
    """C1: the cover serves the SAME weights, reached by a different id."""
    lm = build_task_lm()
    assert lm.task_model == TASK_MODEL
    assert lm.model == GMI_MODEL
    assert lm.api_base == GMI_API_BASE
    assert lm.fallback_provider == "deepinfra"
    assert lm.fallback_model == DEEPINFRA_MODEL
    assert MODEL_ROUTES[TASK_MODEL][lm.fallback_provider] is not None


def test_missing_gmi_key_fails_at_construction(monkeypatch):
    monkeypatch.delenv("GMI_API_KEY", raising=False)
    monkeypatch.delenv("GMI_CLOUD_API_KEY", raising=False)
    with pytest.raises(KeyError, match="GMI_CLOUD_API_KEY"):
        build_task_lm()


def test_either_spelling_of_the_gmi_key_is_accepted(monkeypatch):
    """The fleet exports GMI_CLOUD_API_KEY on some runs and GMI_API_KEY on others."""
    monkeypatch.delenv("GMI_API_KEY", raising=False)
    monkeypatch.setenv("GMI_CLOUD_API_KEY", "cloud-spelling")
    assert build_task_lm()._call_kwargs["api_key"] == "cloud-spelling"


def test_a_402_reissues_the_identical_request_on_the_cover(fake, capsys):
    fake.gmi = _api_error(402)
    lm = build_task_lm()
    stats = CallStats()

    assert lm.complete(MESSAGES, stats=stats) == "from-deepinfra"

    assert fake.providers_called == [GMI_MODEL, GMI_MODEL, DEEPINFRA_MODEL]
    primary, cover = fake.calls[0], fake.calls[-1]
    assert cover["messages"] is primary["messages"], "the cover must see the same request"
    assert cover["reasoning_effort"] == primary["reasoning_effort"] == REASONING_EFFORT
    assert cover["stream"] is True and cover["timeout"] == READ_GAP_TIMEOUT_S
    assert stats.diverted and stats.served_by == "deepinfra"
    err = capsys.readouterr().err
    assert "gmi error" in err and "deepinfra" in err


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 422, 429, 500, 502, 503])
def test_every_http_status_is_diverted(fake, status):
    """LiteLLM retries only 408/409/429/5xx; a cover can answer any of them."""
    fake.gmi = _api_error(status)
    assert build_task_lm().complete(MESSAGES) == "from-deepinfra"


@pytest.mark.parametrize(
    "exc",
    [
        openai.APITimeoutError(request=None),
        openai.APIConnectionError(request=None),
        _api_error(500),
    ],
)
def test_every_api_error_kind_is_diverted(fake, exc):
    fake.gmi = exc
    assert build_task_lm().complete(MESSAGES) == "from-deepinfra"


@pytest.mark.parametrize("name", list(lm_provider.NON_FALLBACK_ERROR_NAMES))
def test_request_is_the_problem_errors_are_not_diverted(fake, name):
    """The identical request fails identically elsewhere -- diverting hides a bug."""
    exc = getattr(litellm, name)(
        message="too long", model=GMI_MODEL, llm_provider="openai"
    )
    fake.gmi = exc
    with pytest.raises(type(exc)):
        build_task_lm().complete(MESSAGES)
    assert DEEPINFRA_MODEL not in fake.providers_called


@pytest.mark.parametrize("exc", [ValueError("program bug"), KeyError("Decision")])
def test_non_api_errors_are_not_diverted(fake, exc):
    """A ValueError from the program must stay visible to whoever evolves it."""
    fake.gmi = exc
    with pytest.raises(type(exc)):
        build_task_lm().complete(MESSAGES)
    assert DEEPINFRA_MODEL not in fake.providers_called


def test_fallback_failure_reraises_chained_from_the_primary_error(fake, capsys):
    primary = _api_error(402)
    fake.gmi = primary
    fake.deepinfra = _api_error(503, "deepinfra is down too")
    with pytest.raises(litellm.APIError) as caught:
        build_task_lm().complete(MESSAGES)
    assert "deepinfra is down too" in str(caught.value)
    assert caught.value.__cause__ is primary, "the primary error must stay attached"
    assert "fallback failed" in capsys.readouterr().err


def test_a_hang_that_recovers_never_reaches_the_cover(fake):
    """The hang guard's local retry is not a divert."""
    fake.gmi = [TimeoutError("silent stream"), "from-gmi"]
    stats = CallStats()
    assert build_task_lm().complete(MESSAGES, stats=stats) == "from-gmi"
    assert stats.attempts == 2 and not stats.diverted
    assert fake.providers_called == [GMI_MODEL, GMI_MODEL]


def test_should_fallback_classifier():
    assert should_fallback(_api_error(402))
    assert should_fallback(openai.APITimeoutError(request=None))
    assert not should_fallback(ValueError("program bug"))
    assert not should_fallback(
        litellm.ContextWindowExceededError(
            message="too long", model=GMI_MODEL, llm_provider="openai"
        )
    )


# ---------------------------------------------------------------------------
# C3 -- one preference order
# ---------------------------------------------------------------------------


def test_default_routing_is_gmi_primary_covered_by_deepinfra():
    lm = build_task_lm()
    assert (lm.provider, lm.fallback_provider) == ("gmi", "deepinfra")


def test_the_default_provider_is_a_member_of_the_preference_order():
    assert DEFAULT_PROVIDER in PROVIDER_PREFERENCE


def test_each_provider_is_covered_by_the_next_one_down_the_order():
    for i, provider in enumerate(PROVIDER_PREFERENCE):
        expected = PROVIDER_PREFERENCE[(i + 1) % len(PROVIDER_PREFERENCE)]
        assert DEFAULT_FALLBACK[provider] == expected


def test_default_fallback_pairs_are_declared_for_every_provider():
    assert set(DEFAULT_FALLBACK) == set(PROVIDER_PREFERENCE)
    assert all(v != k for k, v in DEFAULT_FALLBACK.items())


def test_the_unreachable_provider_ranks_last():
    """DeepSeek's first-party API is not egress-allowed from the runner, so it
    is a local-only route and must never be anyone's default cover on the fleet."""
    assert PROVIDER_PREFERENCE[-1] == "deepseek"
    assert "deepseek" not in (DEFAULT_FALLBACK["gmi"],)


# ---------------------------------------------------------------------------
# C7 -- configurable by environment
# ---------------------------------------------------------------------------


def test_lm_provider_repoints_the_primary_and_its_cover(monkeypatch, fake):
    monkeypatch.setenv("LM_PROVIDER", "deepinfra")
    lm = build_task_lm()
    assert (lm.provider, lm.fallback_provider) == ("deepinfra", "deepseek")
    assert lm.complete(MESSAGES) == "from-deepinfra"
    assert fake.providers_called == [DEEPINFRA_MODEL]


def test_deepseek_primary_sends_the_request_to_deepseek(monkeypatch, fake):
    monkeypatch.setenv("LM_PROVIDER", "deepseek")
    assert build_task_lm().complete(MESSAGES) == "from-deepseek"
    assert fake.providers_called == [DEEPSEEK_MODEL]


def test_deepinfra_primary_falls_back_to_deepseek(monkeypatch, fake):
    monkeypatch.setenv("LM_PROVIDER", "deepinfra")
    fake.deepinfra = _api_error(500)
    assert build_task_lm().complete(MESSAGES) == "from-deepseek"


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "none"])
def test_lm_fallback_off_builds_a_single_provider_lm(monkeypatch, value):
    monkeypatch.setenv("LM_FALLBACK", value)
    lm = build_task_lm()
    assert lm.fallback_provider is None
    assert lm.breaker is None, "skipping the primary is meaningless with no cover"


def test_lm_fallback_off_lets_the_provider_error_propagate(monkeypatch, fake):
    """Mandatory for any per-provider measurement: an armed divert would put
    another provider's calls inside the arm being measured."""
    monkeypatch.setenv("LM_FALLBACK", "0")
    fake.gmi = _api_error(402)
    with pytest.raises(litellm.APIError):
        build_task_lm().complete(MESSAGES)
    assert set(fake.providers_called) == {GMI_MODEL}


@pytest.mark.parametrize("value", ["1", "true", "yes", "on"])
def test_lm_fallback_truthy_values_keep_it_armed(monkeypatch, value):
    monkeypatch.setenv("LM_FALLBACK", value)
    assert build_task_lm().fallback_provider == DEFAULT_FALLBACK[DEFAULT_PROVIDER]


def test_lm_fallback_can_name_the_covering_provider(monkeypatch, fake):
    monkeypatch.setenv("LM_FALLBACK", "deepseek")
    fake.gmi = _api_error(402)
    lm = build_task_lm()
    assert lm.fallback_provider == "deepseek"
    assert lm.complete(MESSAGES) == "from-deepseek"


def test_a_provider_cannot_cover_for_itself(monkeypatch):
    monkeypatch.setenv("LM_PROVIDER", "gmi")
    monkeypatch.setenv("LM_FALLBACK", "gmi")
    with pytest.raises(ValueError, match="cannot be its own fallback"):
        build_task_lm()


def test_explicit_arguments_beat_the_environment(monkeypatch):
    monkeypatch.setenv("LM_PROVIDER", "gmi")
    monkeypatch.setenv("LM_FALLBACK", "deepinfra")
    lm = build_task_lm(provider="deepinfra", fallback="deepseek")
    assert (lm.provider, lm.fallback_provider) == ("deepinfra", "deepseek")


@pytest.mark.parametrize(
    "var,value,match",
    [
        ("LM_PROVIDER", "together", "not a known provider"),
        ("LM_FALLBACK", "together", "neither a boolean nor a provider"),
        ("LM_MODEL", "gpt-4", "is not in MODEL_ROUTES"),
        ("LM_BREAKER", "maybe", "is not a boolean"),
    ],
)
def test_unknown_routing_values_fail_loudly(monkeypatch, var, value, match):
    monkeypatch.setenv(var, value)
    with pytest.raises(ValueError, match=match):
        build_task_lm()


def test_extra_keywords_reach_both_routes(fake):
    fake.gmi = _api_error(402)
    build_task_lm(temperature=0.0).complete(MESSAGES)
    assert all(call["temperature"] == 0.0 for call in fake.calls)


# ---------------------------------------------------------------------------
# C2 -- the routing table is the only place a model id lives
# ---------------------------------------------------------------------------


def test_every_preferred_provider_has_a_route_for_the_task_model():
    for provider in PROVIDER_PREFERENCE:
        assert route_for(TASK_MODEL, provider).model


@pytest.mark.parametrize("provider", list(PROVIDER_PREFERENCE))
def test_each_provider_serves_the_model_string_the_table_gives_it(provider, fake):
    build_task_lm(provider=provider, fallback=False).complete(MESSAGES)
    assert fake.providers_called == [MODEL_ROUTES[TASK_MODEL][provider].model]


def test_only_the_openai_style_route_carries_a_base_url_and_an_explicit_key():
    routes = MODEL_ROUTES[TASK_MODEL]
    assert routes["gmi"].api_base == GMI_API_BASE and routes["gmi"].api_key_env
    for native in ("deepinfra", "deepseek"):
        assert routes[native].api_base is None, "a native LiteLLM provider knows its endpoint"
        assert not routes[native].api_key_env, "its key must stay out of the trace files"


def test_the_deepseek_route_uses_the_first_party_id_not_the_reseller_one():
    assert DEEPSEEK_MODEL == "deepseek/deepseek-v4-flash"
    assert "deepseek-ai/" not in DEEPSEEK_MODEL


def test_serving_a_different_model_is_a_one_row_edit(monkeypatch, fake):
    monkeypatch.setitem(
        MODEL_ROUTES,
        "other-model",
        {
            "gmi": Route("openai/other", api_base=GMI_API_BASE, api_key_env=("GMI_API_KEY",)),
            "deepinfra": Route("deepinfra/other"),
            "deepseek": Route("deepseek/other"),
        },
    )
    monkeypatch.setenv("LM_MODEL", "other-model")
    lm = build_task_lm()
    assert (lm.model, lm.fallback_model) == ("openai/other", "deepinfra/other")


def test_a_provider_missing_from_the_row_fails_at_construction(monkeypatch):
    monkeypatch.setitem(MODEL_ROUTES, "partial", {"gmi": MODEL_ROUTES[TASK_MODEL]["gmi"]})
    monkeypatch.setenv("LM_MODEL", "partial")
    with pytest.raises(ValueError, match="has no route for model 'partial'"):
        build_task_lm()


def test_an_unknown_model_names_the_rows_that_exist(monkeypatch):
    monkeypatch.setenv("LM_MODEL", "llama-3")
    with pytest.raises(ValueError, match=TASK_MODEL):
        build_task_lm()


# ---------------------------------------------------------------------------
# C6 -- retry budget matched to the cover
# ---------------------------------------------------------------------------


def test_the_primary_gets_one_retry_while_a_cover_is_idle(fake):
    fake.gmi = _api_error(500)
    lm = build_task_lm()
    assert lm.max_attempts == MAX_ATTEMPTS == 2
    lm.complete(MESSAGES)
    assert fake.providers_called.count(GMI_MODEL) == 2


def test_a_single_provider_run_keeps_the_full_retry_budget(monkeypatch, fake):
    monkeypatch.setenv("LM_FALLBACK", "0")
    fake.gmi = _api_error(500)
    lm = build_task_lm()
    assert lm.max_attempts == SOLO_MAX_ATTEMPTS > MAX_ATTEMPTS
    with pytest.raises(litellm.APIError):
        lm.complete(MESSAGES)
    assert fake.providers_called.count(GMI_MODEL) == SOLO_MAX_ATTEMPTS


def test_the_cover_keeps_the_full_budget_because_there_is_nowhere_after_it(fake):
    fake.gmi = _api_error(402)
    fake.deepinfra = _api_error(500)
    with pytest.raises(litellm.APIError):
        build_task_lm().complete(MESSAGES)
    assert fake.providers_called.count(DEEPINFRA_MODEL) == COVER_MAX_ATTEMPTS


def test_stats_survive_a_call_that_never_returns(fake):
    """The failure path must leave the same telemetry as the success path --
    an exhausted call used to record nothing but the exception."""
    fake.gmi = _api_error(402)
    fake.deepinfra = _api_error(500)
    stats = CallStats()
    with pytest.raises(litellm.APIError):
        build_task_lm().complete(MESSAGES, stats=stats)
    assert stats.attempts == MAX_ATTEMPTS + COVER_MAX_ATTEMPTS
    assert len(stats.errors) == stats.attempts
    assert stats.diverted and stats.provider == "gmi" and stats.served_by == "deepinfra"


# ---------------------------------------------------------------------------
# C5 -- a sustained outage costs O(1), not O(calls)
# ---------------------------------------------------------------------------


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


@pytest.fixture
def clock():
    return FakeClock()


@pytest.fixture
def breaker(clock, monkeypatch):
    """Install a clock-controlled breaker as gmi's process-wide flag."""
    b = ProviderBreaker("gmi", threshold=3, cooldown=60.0, clock=clock)
    monkeypatch.setitem(lm_provider._BREAKERS, "gmi", b)
    return b


def test_a_breaker_starts_closed_and_ignores_isolated_failures(breaker):
    assert breaker.state == "closed"
    breaker.record_failure()
    breaker.record_failure()
    breaker.record_success()
    breaker.record_failure()
    assert breaker.state == "closed" and breaker.allow()


def test_consecutive_failures_open_the_breaker(breaker):
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state == "open"
    assert not breaker.allow(), "an open breaker costs the call nothing"


def test_an_open_breaker_lets_exactly_one_probe_through_after_the_cooldown(breaker, clock):
    for _ in range(3):
        breaker.record_failure()
    clock.advance(59)
    assert not breaker.allow()
    clock.advance(2)
    assert breaker.allow(), "the cooldown elapsed: one probe"
    assert not breaker.allow(), "only one probe may be in flight"


def test_a_failed_probe_restarts_the_cooldown(breaker, clock):
    for _ in range(3):
        breaker.record_failure()
    clock.advance(61)
    assert breaker.allow()
    breaker.record_failure()
    assert breaker.state == "open"
    clock.advance(59)
    assert not breaker.allow()
    clock.advance(2)
    assert breaker.allow()


def test_a_successful_probe_closes_the_breaker(breaker, clock, capsys):
    for _ in range(3):
        breaker.record_failure()
    clock.advance(61)
    breaker.allow()
    breaker.record_success()
    assert breaker.state == "closed" and breaker.allow()
    assert "healthy again" in capsys.readouterr().err


def test_an_outage_stops_costing_primary_attempts_once_the_breaker_opens(fake, breaker):
    """The whole point: 800 calls into an outage must not be 800 wasted waves."""
    fake.gmi = _api_error(503)
    lm = build_task_lm()
    for _ in range(10):
        assert lm.complete(MESSAGES) == "from-deepinfra"
    # Three diverted failures open it; every later call skips gmi entirely.
    assert breaker.state == "open"
    assert fake.providers_called.count(GMI_MODEL) == 3 * MAX_ATTEMPTS
    assert fake.providers_called.count(DEEPINFRA_MODEL) == 10


def test_a_stray_error_never_opens_the_breaker(fake, breaker):
    """A 402 followed by a success is the observed GMI behaviour, not an outage."""
    lm = build_task_lm()
    for _ in range(10):
        fake.gmi = _api_error(402)
        lm.complete(MESSAGES)
        fake.gmi = "from-gmi"
        lm.complete(MESSAGES)
    assert breaker.state == "closed"


def test_the_primary_comes_back_by_itself_after_the_cooldown(fake, breaker, clock):
    fake.gmi = _api_error(503)
    lm = build_task_lm()
    for _ in range(5):
        lm.complete(MESSAGES)
    assert breaker.state == "open"

    fake.gmi = "from-gmi"
    clock.advance(61)
    assert lm.complete(MESSAGES) == "from-gmi", "the probe must reach the primary"
    assert breaker.state == "closed"


def test_the_breaker_is_shared_by_every_rebuild_of_the_lm(fake, breaker):
    """Every row builds its own ReviewPipeline; health must not be re-learned."""
    fake.gmi = _api_error(503)
    for _ in range(4):
        build_task_lm().complete(MESSAGES)
    assert breaker.state == "open"
    assert fake.providers_called.count(GMI_MODEL) == 3 * MAX_ATTEMPTS


def test_non_provider_errors_never_count_toward_the_breaker(fake, breaker):
    """A ContextWindowExceededError says nothing about provider health."""
    fake.gmi = litellm.ContextWindowExceededError(
        message="too long", model=GMI_MODEL, llm_provider="openai"
    )
    lm = build_task_lm()
    for _ in range(5):
        with pytest.raises(litellm.ContextWindowExceededError):
            lm.complete(MESSAGES)
    assert breaker.state == "closed"


def test_a_skipped_primary_is_recorded_in_the_stats(fake, breaker):
    fake.gmi = _api_error(503)
    lm = build_task_lm()
    for _ in range(3):
        lm.complete(MESSAGES)
    stats = CallStats()
    lm.complete(MESSAGES, stats=stats)
    assert stats.primary_skipped and stats.served_by == "deepinfra"
    assert stats.attempts == 1, "a skipped primary costs no attempt at all"


def test_lm_breaker_off_retries_the_dead_primary_on_every_call(monkeypatch, fake):
    monkeypatch.setenv("LM_BREAKER", "0")
    fake.gmi = _api_error(503)
    lm = build_task_lm()
    assert lm.breaker is None
    for _ in range(4):
        lm.complete(MESSAGES)
    assert fake.providers_called.count(GMI_MODEL) == 4 * MAX_ATTEMPTS


def test_breaker_thresholds_come_from_the_environment(monkeypatch):
    monkeypatch.setenv("LM_BREAKER_FAILURES", "7")
    monkeypatch.setenv("LM_BREAKER_COOLDOWN", "300")
    b = lm_provider.breaker_for("gmi")
    assert (b.threshold, b.cooldown) == (7, 300.0)


def test_breaker_for_returns_one_instance_per_provider():
    assert lm_provider.breaker_for("gmi") is lm_provider.breaker_for("gmi")
    assert lm_provider.breaker_for("gmi") is not lm_provider.breaker_for("deepinfra")


# ---------------------------------------------------------------------------
# C8 -- the wiring is outside the package CodeEvolver evolves
# ---------------------------------------------------------------------------

PROGRAM_DIR = pathlib.Path(__file__).resolve().parent.parent / "src" / "program"


def test_the_module_is_not_inside_the_evolvable_program_package():
    assert pathlib.Path(lm_provider.__file__).parent.name == "src"


@pytest.mark.parametrize(
    "needle",
    ["deepseek-ai/", "gmi-serving.com", "GMI_API_KEY", "DEEPINFRA_API_KEY", "api.deepseek.com"],
)
def test_no_provider_or_model_id_appears_in_the_program_package(needle):
    """C2/C8: switching model or provider must never mean editing the program."""
    for path in PROGRAM_DIR.glob("*.py"):
        assert needle not in path.read_text(), f"{path.name} names {needle!r}"


def test_the_pipeline_reaches_its_provider_through_build_task_lm(fake):
    from src.program.review_pipeline import ReviewPipeline

    pipeline = ReviewPipeline()
    assert isinstance(pipeline.lm, TaskLM)
    assert pipeline.lm.model == GMI_MODEL
