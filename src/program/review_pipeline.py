# Seed program: LLM paper reviewer producing an Accept/Reject decision.
# Prompt adapted from the meta-hyperagents paper_review domain (itself adapted
# from SakanaAI/AI-Scientist perform_review.py).

import json
import os
import re

import litellm

REVIEWER_SYSTEM_PROMPT = (
    "You are an AI researcher who is reviewing a paper that was submitted to a "
    "prestigious ML venue. Be critical and cautious in your decision."
)

NEURIPS_FORM = """
## Review Form
Below is a description of the questions you will be asked on the review form for each paper and some guidelines on what to consider when answering these questions.

1. Summary: Briefly summarize the paper and its contributions. This is not the place to critique the paper; the authors should generally agree with a well-written summary.
  - Strengths and Weaknesses: Please provide a thorough assessment of the strengths and weaknesses of the paper, touching on each of the following dimensions:
  - Originality: Are the tasks or methods new? Is the work a novel combination of well-known techniques? (This can be valuable!) Is it clear how this work differs from previous contributions? Is related work adequately cited?
  - Quality: Is the submission technically sound? Are claims well supported (e.g., by theoretical analysis or experimental results)? Are the methods used appropriate? Is this a complete piece of work or work in progress? Are the authors careful and honest about evaluating both the strengths and weaknesses of their work?
  - Clarity: Is the submission clearly written? Is it well organized? Does it adequately inform the reader?
  - Significance: Are the results important? Are others (researchers or practitioners) likely to use the ideas or build on them? Does the submission address a difficult task in a better way than previous work? Does it advance the state of the art in a demonstrable way?

2. Questions: Please list up and carefully describe any questions and suggestions for the authors.

3. Limitations: Have the authors adequately addressed the limitations and potential negative societal impact of their work?

4. Ethical concerns: If there are ethical issues with this paper, please flag the paper for an ethics review.

5. Soundness: 4: excellent / 3: good / 2: fair / 1: poor

6. Presentation: 4: excellent / 3: good / 2: fair / 1: poor

7. Contribution: 4: excellent / 3: good / 2: fair / 1: poor

8. Overall: 1-10 (very strong reject to award quality)

9. Confidence: 1-5

Respond in the following format:

THOUGHT:
<THOUGHT>

REVIEW JSON:
```json
<JSON>
```

In <THOUGHT>, first briefly discuss your intuitions and reasoning for the evaluation.
Detail your high-level arguments, necessary choices and desired outcomes of the review.
Do not make generic comments here, but be specific to your current paper.

In <JSON>, provide the review in JSON format with the following fields in the order:
- "Summary": A summary of the paper content and its contributions.
- "Strengths": A list of strengths of the paper.
- "Weaknesses": A list of weaknesses of the paper.
- "Originality": A rating from 1 to 4 (low, medium, high, very high).
- "Quality": A rating from 1 to 4 (low, medium, high, very high).
- "Clarity": A rating from 1 to 4 (low, medium, high, very high).
- "Significance": A rating from 1 to 4 (low, medium, high, very high).
- "Questions": A set of clarifying questions to be answered by the paper authors.
- "Limitations": A set of limitations and potential negative societal impacts of the work.
- "Ethical Concerns": A boolean value indicating whether there are ethical concerns.
- "Soundness": A rating from 1 to 4 (poor, fair, good, excellent).
- "Presentation": A rating from 1 to 4 (poor, fair, good, excellent).
- "Contribution": A rating from 1 to 4 (poor, fair, good, excellent).
- "Overall": A rating from 1 to 10 (very strong reject to award quality).
- "Confidence": A rating from 1 to 5 (low, medium, high, very high, absolute).
- "Decision": A decision that has to be one of the following: Accept, Reject.

For the "Decision" field, don't use Weak Accept, Borderline Accept, Borderline Reject, or Strong Reject. Instead, only use Accept or Reject.
This JSON will be automatically parsed, so ensure the format is precise.
"""


def _extract_last_json(text: str):
    """Extract the last parseable JSON object from the response text."""
    candidates = re.findall(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    if not candidates:
        # Fall back to any brace-balanced object, scanning from the end.
        starts = [m.start() for m in re.finditer(r"\{", text)]
        for s in reversed(starts):
            depth = 0
            for i in range(s, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        candidates.append(text[s : i + 1])
                        break
            if candidates:
                break
    for c in reversed(candidates):
        try:
            return json.loads(c)
        except Exception:
            continue
    return None


class ReviewPipeline:
    """One LLM call per paper: full NeurIPS-form review, then extract Decision."""

    def __init__(self):
        self.model = "openai/gpt-5.4-mini"
        # Pin the solver to the real OpenAI endpoint. The server environment
        # points OPENAI_* at GMI Cloud (for the reflection LM), so explicit
        # kwargs are required here; REAL_OPENAI_API_KEY carries the OpenAI key.
        self.api_base = "https://api.openai.com/v1"
        self.api_key = os.environ.get("REAL_OPENAI_API_KEY") or os.environ.get(
            "OPENAI_API_KEY"
        )

    def __call__(self, paper_text: str = "", **kwargs) -> str:
        prompt = (
            NEURIPS_FORM
            + f"""
Here is the paper you are asked to review:
```
{paper_text}
```"""
        )
        try:
            response = litellm.completion(
                model=self.model,
                api_base=self.api_base,
                api_key=self.api_key,
                messages=[
                    {"role": "system", "content": REVIEWER_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                timeout=600,
                num_retries=2,
            )
            content = response.choices[0].message.content or ""
        except Exception as e:
            print(f"[review] LLM call error: {e}")
            return ""

        review = _extract_last_json(content)
        if not review or "Decision" not in review:
            print("[review] could not extract Decision from response")
            return ""
        return str(review["Decision"]).strip().lower()
