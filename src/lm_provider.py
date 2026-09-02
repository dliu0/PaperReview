"""Task LM for the review program: one pinned model, served by a ranked list of
providers, with a per-call cross-provider fallback and a shared health flag.

This is `docs/provider_fallback.md` implemented for this repo. The contract
(§1 of that doc) is what has to hold; the reference implementation it names --
`langProBe/lm_provider.py` in LangProBe-CodeEvolver -- is a `dspy.LM` subclass,
and this repo has no dspy: the solver is one streamed `litellm.completion` with
a hang guard. So the routing table, the preference order, the env vars, the
error classifier and the circuit breaker are ported as-is, and only the call
layer differs -- `TaskLM.complete()` in place of `ProviderFallbackLM.forward()`.

Why this exists
---------------
The resellers serving this model are individually unreliable, usually for
billing rather than outage reasons. GMI answers a single request with ``Error
code: 402 - {'error': 'Insufficient balance', 'reason': 'model_access_denied'}``
and serves the very next one normally. LiteLLM retries 408/409/429/5xx and
treats every other 4xx as permanent, so an unhandled provider error fails the
call, `ReviewPipeline.__call__` returns "" and the row scores 0.0 -- and the
optimizer attributes an infrastructure failure to the program it is evolving.

Where a model id lives
----------------------
In ``MODEL_ROUTES`` and nowhere else (C2). One row per model, one column per
provider, holding that provider's LiteLLM model string plus whatever else its
route needs (``api_base``, an explicit key). The three providers name the same
weights differently -- ``openai/deepseek-ai/DeepSeek-V4-Flash`` on GMI,
``deepinfra/deepseek-ai/DeepSeek-V4-Flash`` on DeepInfra,
``deepseek/deepseek-v4-flash`` on DeepSeek's own API -- so switching models is a
one-row edit, not a hunt for three constants. ``TASK_MODEL`` (or ``$LM_MODEL``)
picks the row. Nothing in ``src/program/`` names a model, an endpoint or a key.

Where this module lives, and why
--------------------------------
``src/`` and NOT ``src/program/`` (C8). CodeEvolver rewrites the program
package; the provider wiring and the pinned model are benchmark constraints, so
they sit outside it. The program imports ``build_task_lm`` and nothing else.

The hang guard lives here too
-----------------------------
``READ_GAP_TIMEOUT_S`` / ``TOTAL_BUDGET_S`` / ``MAX_ATTEMPTS`` moved out of
`ReviewPipeline._complete` into this module. They are provider behaviour --
GMI can accept a connection and then send zero bytes until its gateway kills it
minutes later, walling a whole 50-row eval at the straggler -- and the divert
has to wrap the streaming call anyway, so keeping them in the program would
have meant two copies of the same retry loop. Every LLM call the program makes
gets the guard for free, on whichever provider serves it.

Keys
----
Only routes reached through LiteLLM's generic ``openai/`` provider carry an
explicit key; GMI is the only one of the three. DeepInfra and DeepSeek are
native LiteLLM providers that read ``DEEPINFRA_API_KEY`` / ``DEEPSEEK_API_KEY``
themselves at call time, which also keeps those values out of the trace files.
The primary's key is needed to BUILD the LM; the cover's must be present before
the first divert.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import openai  # installed by litellm; the base class of every LiteLLM error

# ---------------------------------------------------------------------------
# Model routing table
# ---------------------------------------------------------------------------

# GMI Cloud has no LiteLLM provider of its own; it is reached through the
# OpenAI-compatible route (model="openai/<id>" + api_base=<GMI endpoint>).
GMI_API_BASE = "https://api.gmi-serving.com/v1"


@dataclass(frozen=True)
class Route:
    """How ONE provider serves ONE model.

    ``model`` is the LiteLLM model string. ``api_base`` and ``api_key_env`` are
    only needed by providers reached through the generic ``openai/`` route --
    LiteLLM's native providers (``deepinfra/``, ``deepseek/``) know their own
    endpoint and read their own key from the environment at call time.

    ``api_key_env`` is a tuple because the fleet exports the GMI key under
    either spelling: this repo's solver has always read ``GMI_CLOUD_API_KEY or
    GMI_API_KEY`` (CodeEvolver's `runner/entrypoint.py` fingerprints both), and
    silently ignoring the one that is actually set would be a false outage.
    """

    model: str
    api_base: str | None = None
    api_key_env: tuple[str, ...] = ()

    def call_kwargs(self) -> dict[str, Any]:
        """``litellm.completion`` kwargs pinning a request to this route."""
        kwargs: dict[str, Any] = {"model": self.model}
        if self.api_base is not None:
            kwargs["api_base"] = self.api_base
        if self.api_key_env:
            # Explicit: LiteLLM's openai/ route would otherwise fall back to
            # OPENAI_API_KEY -- which the harness points at a DIFFERENT model.
            # Raising here is the right failure: the run cannot use this
            # provider without its key, and finding that out at construction
            # beats finding it out 50 rows into an eval.
            for name in self.api_key_env:
                value = os.environ.get(name)
                if value:
                    kwargs["api_key"] = value
                    break
            else:
                raise KeyError(
                    f"none of {list(self.api_key_env)} is set; the route "
                    f"{self.model!r} cannot be served without its key"
                )
        return kwargs


# One row per model, one column per provider. This is the ONLY place a model id
# lives: to serve a different model, add a row here and point TASK_MODEL (or
# $LM_MODEL) at it. A provider missing from a row cannot serve that model and
# says so at construction rather than at the first call.
MODEL_ROUTES: dict[str, dict[str, Route]] = {
    "deepseek-v4-flash": {
        "gmi": Route(
            "openai/deepseek-ai/DeepSeek-V4-Flash",
            api_base=GMI_API_BASE,
            api_key_env=("GMI_CLOUD_API_KEY", "GMI_API_KEY"),
        ),
        "deepinfra": Route("deepinfra/deepseek-ai/DeepSeek-V4-Flash"),
        # DeepSeek's first-party API uses its own short ids rather than the
        # HuggingFace-style ``deepseek-ai/<name>`` the resellers use; picking
        # the wrong one silently benchmarks a DIFFERENT model. This one is
        # listed by GET https://api.deepseek.com/models.
        "deepseek": Route("deepseek/deepseek-v4-flash"),
    },
}

# The model the benchmark pins; $LM_MODEL selects another row of MODEL_ROUTES.
TASK_MODEL = "deepseek-v4-flash"
MODEL_ENV_VAR = "LM_MODEL"

# Reasoning is enabled via the standard OpenAI `reasoning_effort` param. None of
# the routes allow it by default, so it must be forwarded via
# allowed_openai_params=[...] (BerriAI/litellm#14039). Do NOT use
# thinking={"type": "enabled"}: DeepInfra's endpoint rejects the kwarg.
REASONING_EFFORT = "high"
_ALLOWED_OPENAI_PARAMS = ["reasoning_effort"]

# ---------------------------------------------------------------------------
# Provider preference
# ---------------------------------------------------------------------------

# Most preferred first, and the single ordering that decides both who serves a
# call and who covers for whom (C3).
#
# This differs from LangProBe's ("gmi", "deepseek", "deepinfra") on purpose, and
# the reason is reachability rather than taste. CodeEvolver's runner egresses
# through squid, and `runner/squid.conf` allows only `.gmi-serving.com` and
# `.deepinfra.com`; `runner/preflight.py` probes only those two hosts, and
# nothing in the runner forwards `DEEPSEEK_API_KEY` into the container. So on
# the fleet the first-party DeepSeek API is not a route this benchmark can take
# -- it stays in the table (it works fine for a local run that exports the key)
# but it ranks last, because a cover that cannot be reached is not a cover.
# Restoring the LangProBe order is a one-line edit once the runner allowlists
# `api.deepseek.com` and passes the key through.
PROVIDER_PREFERENCE = ("gmi", "deepinfra", "deepseek")
PROVIDERS = PROVIDER_PREFERENCE

# The primary when $LM_PROVIDER is unset. LangProBe currently pins this one step
# down its order because GMI's account there is unfunded; this repo's runs are
# the ones GMI is funded for, so it stays at the head. If GMI starts failing
# every call, move this to PROVIDER_PREFERENCE[1] -- the covers follow the same
# list, so that single edit is the whole change.
DEFAULT_PROVIDER = PROVIDER_PREFERENCE[0]

PROVIDER_ENV_VAR = "LM_PROVIDER"
FALLBACK_ENV_VAR = "LM_FALLBACK"
# Who covers for whom when LM_FALLBACK is on but does not name a provider: the
# next provider down the preference order, wrapping at the end. So GMI diverts
# to DeepInfra, DeepInfra to DeepSeek, DeepSeek back to GMI.
DEFAULT_FALLBACK = {
    provider: PROVIDER_PREFERENCE[(i + 1) % len(PROVIDER_PREFERENCE)]
    for i, provider in enumerate(PROVIDER_PREFERENCE)
}

# ---------------------------------------------------------------------------
# Hang guard and retry budget
# ---------------------------------------------------------------------------

# GMI can "hang" a request: zero bytes until its gateway kills the connection
# minutes later, which walls an entire parallel eval at the straggler. Streaming
# makes hangs detectable -- GMI streams reasoning deltas continuously (measured
# inter-chunk gap ~3s), so READ_GAP_TIMEOUT_S of total silence is an unambiguous
# hang: abort fast and retry instead of waiting for the gateway. httpx applies
# `timeout` per READ on a stream, so long generations are unaffected;
# TOTAL_BUDGET_S caps the row across every attempt on every provider.
READ_GAP_TIMEOUT_S = 240
TOTAL_BUDGET_S = 2400

# Attempts on the primary before the call is handed to the cover: two, i.e. one
# retry (C6). A third attempt is dead time when an equivalent provider is
# sitting idle. With no cover the budget doubles -- there is nowhere else to go,
# so retrying is all that can save the row.
MAX_ATTEMPTS = 2
SOLO_MAX_ATTEMPTS = 4
COVER_MAX_ATTEMPTS = 4

# ---------------------------------------------------------------------------
# Circuit breaker
# ---------------------------------------------------------------------------

BREAKER_ENV_VAR = "LM_BREAKER"
BREAKER_FAILURES_ENV_VAR = "LM_BREAKER_FAILURES"
BREAKER_COOLDOWN_ENV_VAR = "LM_BREAKER_COOLDOWN"
# Three CONSECUTIVE diverted failures. A stray 402 is followed by a success,
# which resets the count, so strays never trip it; an outage trips it inside the
# first wave of concurrent calls.
BREAKER_FAILURES = 3
BREAKER_COOLDOWN_SECONDS = 60.0

_TRUTHY = {"1", "true", "yes", "on"}
_FALSY = {"0", "false", "no", "off", "none", ""}

_LOG_ERROR_CHARS = 300

# Request-is-the-problem errors: the identical request on the identical model
# fails the same way on the other provider, so diverting only hides a program
# fault. Resolved lazily by name -- `litellm` is imported at CALL time (see
# `_litellm`), so the tests can stand a stub in for it.
NON_FALLBACK_ERROR_NAMES = ("ContextWindowExceededError", "UnsupportedParamsError")


def _litellm():
    """The litellm module, resolved per call rather than bound at import.

    `tests/test_tracing.py` swaps a stub into ``sys.modules`` and re-imports the
    program; a module-level ``import litellm`` here would keep handing back the
    real one and the stub would never be exercised.
    """
    import litellm

    return litellm


def non_fallback_errors() -> tuple[type[BaseException], ...]:
    """The exception classes a second provider could not help with."""
    litellm = _litellm()
    return tuple(
        exc
        for exc in (getattr(litellm, name, None) for name in NON_FALLBACK_ERROR_NAMES)
        if isinstance(exc, type) and issubclass(exc, BaseException)
    )


def _status_code(exc: BaseException) -> int | None:
    """HTTP status carried by a LiteLLM/OpenAI exception, or None."""
    code = getattr(exc, "status_code", None)
    if isinstance(code, bool) or not isinstance(code, int):
        return None
    return code


def should_fallback(exc: BaseException) -> bool:
    """True for a provider/API error a second provider could plausibly answer.

    ``openai.APIError`` is the common base of every LiteLLM exception (4xx, 5xx,
    timeouts, connection errors); anything else is not an API error and is left
    alone -- a ``ValueError`` from the program must stay visible to whoever is
    evolving it (C4). ``non_fallback_errors()`` carves out the
    request-is-the-problem cases.
    """
    if not isinstance(exc, openai.APIError):
        return False
    return not isinstance(exc, non_fallback_errors())


def _env_flag(name: str, flag: bool | None = None, default: bool = True) -> bool:
    """A boolean env var: explicit ``flag`` wins, then ``$name``, then ``default``."""
    if flag is not None:
        return bool(flag)
    value = os.environ.get(name, "").strip().lower()
    if not value:
        return default
    if value in _TRUTHY:
        return True
    if value in _FALSY:
        return False
    raise ValueError(
        f"{name}={value!r} is not a boolean; expected one of {sorted(_TRUTHY)} "
        f"or {sorted(v for v in _FALSY if v)}"
    )


def resolve_model(name: str | None = None) -> str:
    """The model row: explicit ``name``, else ``$LM_MODEL``, else ``TASK_MODEL``."""
    value = (name if name is not None else os.environ.get(MODEL_ENV_VAR, "")).strip()
    value = value or TASK_MODEL
    if value not in MODEL_ROUTES:
        raise ValueError(
            f"{value!r} is not in MODEL_ROUTES; known models: {list(MODEL_ROUTES)}. "
            f"Add a row (one entry per provider that serves it) to route it."
        )
    return value


def resolve_provider(name: str | None = None) -> str:
    """The primary: explicit ``name``, else ``$LM_PROVIDER``, else DEFAULT_PROVIDER."""
    value = (
        name if name is not None else os.environ.get(PROVIDER_ENV_VAR, "")
    ).strip().lower()
    if not value:
        return DEFAULT_PROVIDER
    if value not in PROVIDERS:
        raise ValueError(
            f"{PROVIDER_ENV_VAR}={value!r} is not a known provider; expected one of "
            f"{list(PROVIDERS)}"
        )
    return value


def resolve_fallback(primary: str, flag: bool | str | None = None) -> str | None:
    """The provider that covers for ``primary``, or None if diversion is off.

    ``$LM_FALLBACK`` (or ``flag``) is either a boolean -- on meaning
    ``DEFAULT_FALLBACK[primary]`` -- or the name of a provider to use instead.
    """
    if flag is None:
        flag = os.environ.get(FALLBACK_ENV_VAR, "").strip().lower()
    if flag is True:
        return DEFAULT_FALLBACK[primary]
    if flag is False:
        return None
    value = str(flag).strip().lower()
    if not value or value in _TRUTHY:
        return DEFAULT_FALLBACK[primary]
    if value in _FALSY:
        return None
    if value in PROVIDERS:
        if value == primary:
            raise ValueError(
                f"{FALLBACK_ENV_VAR}={value!r} names the primary provider; a provider "
                f"cannot be its own fallback"
            )
        return value
    raise ValueError(
        f"{FALLBACK_ENV_VAR}={value!r} is neither a boolean nor a provider; expected "
        f"one of {sorted(_TRUTHY)}, {sorted(v for v in _FALSY if v)} or {list(PROVIDERS)}"
    )


def route_for(model: str, provider: str) -> Route:
    """The ``Route`` serving ``model`` on ``provider``, or a loud failure."""
    routes = MODEL_ROUTES[model]
    try:
        return routes[provider]
    except KeyError:
        raise ValueError(
            f"provider {provider!r} has no route for model {model!r}; "
            f"MODEL_ROUTES[{model!r}] serves {sorted(routes)}. Add the provider's "
            f"model string to that row, or route the run elsewhere with "
            f"{PROVIDER_ENV_VAR}/{FALLBACK_ENV_VAR}."
        ) from None


def _describe(exc: BaseException) -> str:
    text = " ".join(str(exc).split())
    if len(text) > _LOG_ERROR_CHARS:
        text = text[:_LOG_ERROR_CHARS] + "..."
    return f"{type(exc).__name__}: {text}"


def _warn(msg: str) -> None:
    print(f"[WARNING] {msg}", file=sys.stderr, flush=True)


def _error(msg: str) -> None:
    print(f"[ERROR] {msg}", file=sys.stderr, flush=True)


def _mark_span(**attrs: Any) -> None:
    """Best-effort: stamp the routing on the active OTel span (never raises)."""
    try:
        from opentelemetry import trace

        span = trace.get_current_span()
        for key, value in attrs.items():
            span.set_attribute(key, value)
    except Exception:  # noqa: BLE001 -- tracing must never break a call
        pass


class ProviderBreaker:
    """Process-wide health flag for one provider, shared by every thread.

    CLOSED -- calls go to the provider. ``threshold`` consecutive diverted
    failures (any success resets the count) open it.

    OPEN -- ``allow()`` returns False and callers skip the provider entirely:
    no request, no retries, no backoff. Costs nothing per call, which is the
    whole point at 50 threads.

    PROBING -- after ``cooldown`` seconds exactly one call is let through; its
    outcome closes the breaker or restarts the cooldown. Only one probe is in
    flight at a time, so a still-dead provider costs one call per cooldown.

    State is three numbers mutated under a lock that is never held across an I/O
    call. Instances are process-global (see ``breaker_for``) rather than
    per-LM: provider health is a property of the provider, and a per-LM breaker
    would fragment it across the 50 threads an eval runs.
    """

    def __init__(
        self,
        provider: str,
        threshold: int = BREAKER_FAILURES,
        cooldown: float = BREAKER_COOLDOWN_SECONDS,
        clock=time.monotonic,
    ):
        self.provider = provider
        self.threshold = threshold
        self.cooldown = cooldown
        self._clock = clock
        self._lock = threading.Lock()
        self._failures = 0
        self._opened_at: float | None = None  # None => closed
        self._probing = False
        self._skipped = 0

    @property
    def state(self) -> str:
        with self._lock:
            if self._opened_at is None:
                return "closed"
            return "probing" if self._probing else "open"

    def allow(self) -> bool:
        """False when the provider is known-unhealthy and should be skipped."""
        with self._lock:
            if self._opened_at is None:
                return True
            if self._probing or self._clock() - self._opened_at < self.cooldown:
                self._skipped += 1
                return False
            self._probing = True
            waited = self._clock() - self._opened_at
        _warn(
            f"{self.provider} has been skipped for {waited:.0f}s "
            f"({self._skipped} calls served by the fallback) -- probing it with one call"
        )
        return True

    def record_success(self) -> None:
        with self._lock:
            recovered = self._opened_at is not None
            skipped = self._skipped
            self._failures = 0
            self._opened_at = None
            self._probing = False
            self._skipped = 0
        if recovered:
            _warn(
                f"{self.provider} answered the probe -- marking it healthy again "
                f"({skipped} calls went to the fallback while it was down)"
            )
            _mark_span(
                **{"lm.breaker.provider": self.provider, "lm.breaker.state": "closed"}
            )

    def record_failure(self) -> None:
        """Record a diverted provider error; may open (or re-open) the breaker."""
        with self._lock:
            was_probe = self._probing
            self._probing = False
            self._failures += 1
            failures = self._failures
            opened = False
            if self._opened_at is not None:
                self._opened_at = self._clock()  # probe failed: restart the cooldown
            elif failures >= self.threshold:
                self._opened_at = self._clock()
                opened = True
        if opened:
            _warn(
                f"{self.provider} failed {failures} calls in a row -- marking it "
                f"unhealthy and sending every call straight to the fallback for "
                f"{self.cooldown:.0f}s"
            )
            _mark_span(
                **{"lm.breaker.provider": self.provider, "lm.breaker.state": "open"}
            )
        elif was_probe:
            _warn(
                f"{self.provider} failed the probe -- still unhealthy, "
                f"skipping it for another {self.cooldown:.0f}s"
            )


_BREAKERS: dict[str, ProviderBreaker] = {}
_BREAKERS_LOCK = threading.Lock()


def breaker_for(provider: str) -> ProviderBreaker:
    """The shared breaker for ``provider`` -- one per process, not per LM.

    The LM stores only the provider NAME and looks the breaker up per call, so
    that copying or rebuilding a ``TaskLM`` (every row builds its own
    ``ReviewPipeline``) cannot fragment the recorded health.
    """
    breaker = _BREAKERS.get(provider)
    if breaker is None:
        with _BREAKERS_LOCK:
            breaker = _BREAKERS.setdefault(
                provider,
                ProviderBreaker(
                    provider,
                    threshold=int(
                        os.environ.get(BREAKER_FAILURES_ENV_VAR, "").strip()
                        or BREAKER_FAILURES
                    ),
                    cooldown=float(
                        os.environ.get(BREAKER_COOLDOWN_ENV_VAR, "").strip()
                        or BREAKER_COOLDOWN_SECONDS
                    ),
                ),
            )
    return breaker


def reset_breakers() -> None:
    """Forget every provider's recorded health (tests; a fresh run in-process)."""
    with _BREAKERS_LOCK:
        _BREAKERS.clear()


@dataclass
class CallStats:
    """What one ``TaskLM.complete()`` cost, for the caller's `llm` span.

    The program owns its span; this is what it stamps on it. Mutated in place as
    the call proceeds so the FAILURE path has the same telemetry as the success
    path -- an exhausted call used to record nothing but the exception.
    """

    provider: str = ""
    model: str = ""
    api_base: str | None = None
    fallback_provider: str | None = None
    served_by: str = ""
    attempts: int = 0
    stream_chunks: int = 0
    elapsed_s: float = 0.0
    primary_skipped: bool = False
    diverted: bool = False
    errors: list[str] = field(default_factory=list)


class TaskLM:
    """``TASK_MODEL`` on ``provider``; a provider error re-issues the call elsewhere.

    ``model`` / ``provider`` / ``fallback`` / ``breaker`` default to ``$LM_MODEL``,
    ``$LM_PROVIDER``, ``$LM_FALLBACK`` and ``$LM_BREAKER``. With ``fallback=False``
    no cover route is resolved, the breaker is inert, and errors propagate.
    """

    def __init__(
        self,
        provider: str | None = None,
        fallback: bool | str | None = None,
        model: str | None = None,
        breaker: bool | None = None,
        **overrides: Any,
    ):
        task_model = resolve_model(model)
        provider = resolve_provider(provider)
        cover = resolve_fallback(provider, fallback)

        self.task_model = task_model
        self.provider = provider
        self.fallback_provider: str | None = cover
        self.overrides = overrides

        # Resolved (and keys read) at construction, so a missing key or a
        # provider absent from the row fails before the eval starts, not 50
        # rows in. Both routes: the cover's key must be there before the first
        # divert, and a divert is not the moment to discover it is not.
        self._route = route_for(task_model, provider)
        self._call_kwargs = self._route.call_kwargs()
        self._cover_route: Route | None = None
        self._cover_kwargs: dict[str, Any] | None = None
        if cover is not None:
            self._cover_route = route_for(task_model, cover)
            self._cover_kwargs = self._cover_route.call_kwargs()

        # Armed only when there is somewhere to divert to: skipping the primary
        # is meaningless without a cover.
        self.breaker_enabled = cover is not None and _env_flag(BREAKER_ENV_VAR, breaker)
        self.max_attempts = MAX_ATTEMPTS if cover is not None else SOLO_MAX_ATTEMPTS

    # -- introspection ------------------------------------------------------

    @property
    def model(self) -> str:
        """The primary route's LiteLLM model string (what the span records)."""
        return self._route.model

    @property
    def api_base(self) -> str | None:
        return self._route.api_base

    @property
    def fallback_model(self) -> str | None:
        return self._cover_route.model if self._cover_route is not None else None

    @property
    def breaker(self) -> ProviderBreaker | None:
        """The primary's shared health flag, or None when it is not armed."""
        return breaker_for(self.provider) if self.breaker_enabled else None

    # -- the call -----------------------------------------------------------

    def complete(self, messages, stats: CallStats | None = None, **overrides: Any) -> str:
        """Stream one completion and return its text, diverting on a provider error.

        ``stats`` is filled in as the call proceeds (attempts, chunks, elapsed,
        every failed attempt's error) and is readable by the caller whether the
        call returns or raises. Exhausting every route re-raises the last
        exception, exactly as the un-routed call used to.
        """
        stats = stats if stats is not None else CallStats()
        stats.provider = self.provider
        stats.model = self.model
        stats.api_base = self.api_base
        stats.fallback_provider = self.fallback_provider
        stats.served_by = self.provider
        start = time.monotonic()
        deadline = start + TOTAL_BUDGET_S

        breaker = self.breaker
        if breaker is not None and not breaker.allow():
            stats.primary_skipped = True
            stats.served_by = self.fallback_provider or self.provider
            _mark_span(**{"lm.primary_skipped": True, "lm.primary": self.provider})
            return self._run_cover(messages, start, deadline, stats, overrides)

        try:
            text = self._attempt_route(
                self._call_kwargs, messages, self.max_attempts,
                start, deadline, stats, overrides,
            )
        except Exception as exc:  # noqa: BLE001 -- classified below
            if self._cover_route is None or not should_fallback(exc):
                stats.elapsed_s = round(time.monotonic() - start, 1)
                raise
            if breaker is not None:
                breaker.record_failure()
            code = _status_code(exc)
            self._log_diverted(code, exc)
            stats.diverted = True
            stats.served_by = self.fallback_provider or self.provider
            diverted_at = time.monotonic()
            try:
                text = self._run_cover(messages, start, deadline, stats, overrides)
            except Exception as exc2:  # noqa: BLE001
                self._log_fallback_failed(code, exc2, diverted_at)
                raise exc2 from exc
            self._log_fallback_ok(code, diverted_at)
            return text

        if breaker is not None:
            breaker.record_success()
        return text

    def _run_cover(self, messages, start, deadline, stats: CallStats, overrides) -> str:
        assert self._cover_kwargs is not None  # only reached when a cover exists
        return self._attempt_route(
            self._cover_kwargs, messages, COVER_MAX_ATTEMPTS,
            start, deadline, stats, overrides,
        )

    def _attempt_route(
        self, call_kwargs, messages, max_attempts, start, deadline,
        stats: CallStats, overrides,
    ) -> str:
        """Stream one route with the hang guard; raise the last error if all fail.

        The guard retries ANY exception (that is what makes a hang recoverable),
        while the divert above triggers only on the LAST error and only when
        `should_fallback` accepts it. So a hang that recovers on the retry never
        reaches the cover, and a 402 costs one wasted retry before it does --
        cheap next to re-issuing a 50k-token prefill on the wrong provider.
        """
        last_err: BaseException | None = None
        for _attempt in range(max_attempts):
            if time.monotonic() > deadline - READ_GAP_TIMEOUT_S:
                stats.errors.append(
                    "skipped attempt: too little of the total budget left"
                )
                break
            stats.attempts += 1
            try:
                text = self._stream(call_kwargs, messages, deadline, stats, overrides)
                stats.elapsed_s = round(time.monotonic() - start, 1)
                return text
            except Exception as exc:  # noqa: BLE001 -- hang/gap/transient
                last_err = exc
                stats.errors.append(_describe(exc))
        stats.elapsed_s = round(time.monotonic() - start, 1)
        raise last_err if last_err else RuntimeError("completion failed")

    def _stream(self, call_kwargs, messages, deadline, stats: CallStats, overrides) -> str:
        stream = _litellm().completion(
            messages=messages,
            reasoning_effort=REASONING_EFFORT,
            allowed_openai_params=list(_ALLOWED_OPENAI_PARAMS),
            stream=True,
            # Per-read gap cap on a stream (NOT total duration): only trips when
            # the connection goes fully silent (a real hang).
            timeout=READ_GAP_TIMEOUT_S,
            **call_kwargs,
            **self.overrides,
            **overrides,
        )
        parts = []
        for chunk in stream:
            if time.monotonic() > deadline:
                raise TimeoutError(f"row exceeded total budget {TOTAL_BUDGET_S}s")
            if chunk.choices:
                stats.stream_chunks += 1
                delta = chunk.choices[0].delta
                if delta is not None and getattr(delta, "content", None):
                    parts.append(delta.content)
        return "".join(parts)

    # -- logging ------------------------------------------------------------

    def _log_diverted(self, code: int | None, exc: BaseException) -> None:
        desc = _describe(exc)
        _warn(
            f"{self.provider} error on {self.model} (status={code}, {desc}) "
            f"-- retrying this call on {self.fallback_provider} ({self.fallback_model})"
        )
        _mark_span(
            **{
                "lm.fallback.provider": self.fallback_provider,
                "lm.fallback.model": self.fallback_model,
                "lm.fallback.primary_status": code if code is not None else -1,
                "lm.fallback.primary_error": desc,
            }
        )

    def _log_fallback_ok(self, code: int | None, started: float) -> None:
        elapsed = time.monotonic() - started
        _warn(
            f"{self.fallback_provider} fallback succeeded in {elapsed:.1f}s "
            f"(primary_status={code})"
        )
        _mark_span(**{"lm.fallback.outcome": "ok", "lm.fallback.seconds": elapsed})

    def _log_fallback_failed(
        self, code: int | None, exc: BaseException, started: float
    ) -> None:
        elapsed = time.monotonic() - started
        desc = _describe(exc)
        _error(
            f"{self.fallback_provider} fallback failed after {elapsed:.1f}s "
            f"(primary_status={code}, fallback_status={_status_code(exc)}): {desc}"
        )
        _mark_span(
            **{
                "lm.fallback.outcome": "failed",
                "lm.fallback.seconds": elapsed,
                "lm.fallback.error": desc,
            }
        )


def build_task_lm(**overrides: Any) -> TaskLM:
    """The benchmark's task LM -- the entrypoint every pipeline calls.

    ``model`` / ``provider`` / ``fallback`` / ``breaker`` select the routing
    (defaulting to ``$LM_MODEL``, ``$LM_PROVIDER``, ``$LM_FALLBACK`` and
    ``$LM_BREAKER``); every other keyword is forwarded to ``litellm.completion``
    on BOTH routes.
    """
    return TaskLM(**overrides)
