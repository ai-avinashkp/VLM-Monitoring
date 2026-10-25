SYSTEM = (
    "You are the eyes of a live webcam monitor. The camera faces one person (the user) at "
    "their desk. Describe only what is clearly visible, in natural and specific sentences. "
    "Never use class labels or confidence scores.\n"
    "Pay attention to the person: anything on the face or head (glasses, sunglasses, mask, "
    "headphones, earbuds, cap, hood) and exactly how it is worn, clothing, what they hold or use, "
    "what they are doing, where they are looking, posture, and expression when it is obvious. "
    "Mention the room or lighting only if notable.\n"
    "If something is unclear, say so. Do not guess identity, age, or anything not visible.\n"
    "Always speak to the person directly as \"you\" / \"your\" (\"You're wearing...\"), "
    "never \"a man\", \"a woman\" or \"the person\"."
)

# The reply is started for the model: forces raw JSON (no fences) and second-person wording.
DESCRIBE_PREFILL = '{"summary": "You'

# Each frame is described on its own. With the previous description in the prompt this 2B model
# copies it, stops returning boxes, and can loop; what changed is asked separately (CHANGES),
# showing it both frames. The wording below was picked by measurement (boxes on 18/23 details vs
# 6-13 for other variants); oddly the unused empty "changes" slot helps grounding, so it stays.
DESCRIBE = """Describe the current frame.
This is the first frame, so "changes" must be an empty string.
Reply with ONLY one compact JSON object on a single line, no markdown:
{"summary": "<one short sentence>", "observations": [["<detail>", x1, y1, x2, y2], ...], "changes": "<what changed, or empty string>"}
Rules: 3 or 4 observations, most important first, each a separate detail under 10 words written to
"you" (e.g. "Your black glasses sit low on your nose"). x1, y1, x2, y2 is the region the detail is
about in 0-1000 coordinates (x1,y1 = top-left); write just ["<detail>"] if it is about the whole scene."""

MAX_OBSERVATIONS = 5  # generation stops once this many are complete (or on a repeat)

# Before/after comparison. On its own the 2B model often invents a "you put on X" for an item that
# was there all along, so every claim is checked against one frame with CHANGE_CHECK (measured:
# false claims on test pairs went from 4/6 to 1/6; removals are mostly missed rather than misreported).
CHANGES = """Image 1 is the webcam a few seconds ago, image 2 is now.
Did you put on, take off, pick up or put down anything between image 1 and image 2? Ignore pose, head movement and lighting.
If not, answer exactly: none
If yes, answer in one short sentence to "you", e.g. "You took off your cap"."""

CHANGE_CHECK = """Claim: "{claim}"
This image was taken {when} that change. {question} Answer only yes or no."""

ASK = """Question: {question}
Answer from what is visible in this frame only, in 1 to 3 natural sentences."""


def ask_prompt(question: str) -> str:
    return ASK.format(question=question.strip())
