from app.parsing import normalize_box, parse_partial, parse_scene


def test_plain_json():
    s = parse_scene('{"summary": "You sit.", "observations": [{"text": "Glasses on", "box": [100, 200, 300, 400]}], "changes": ""}')
    assert s.parsed and s.summary == "You sit."
    assert s.observations[0].text == "Glasses on"
    assert s.observations[0].box == [0.1, 0.2, 0.3, 0.4]
    assert s.changes == ""


def test_fenced_json_with_trailing_text_and_commas():
    text = 'Sure!\n```json\n{"summary": "A", "observations": ["Mask under chin",], "changes": "None."}\n```\nHope that helps.'
    s = parse_scene(text)
    assert s.parsed
    assert [o.text for o in s.observations] == ["Mask under chin"]
    assert s.observations[0].box is None
    assert s.changes == ""


def test_think_block_and_alt_box_keys():
    s = parse_scene('<think>hmm</think>{"summary": "B", "observations": [{"text": "Cup", "bbox_2d": [0.5, 0.5, 0.7, 0.9]}]}')
    assert s.observations[0].box == [0.5, 0.5, 0.7, 0.9]


def test_unparseable_falls_back_to_prose():
    s = parse_scene("You are wearing glasses and smiling.")
    assert not s.parsed
    assert "glasses" in s.summary
    assert s.observations == []


def test_no_change_phrases_are_cleared():
    for phrase in ["No changes", "no notable changes detected", "Nothing has changed.", "N/A", "none"]:
        s = parse_scene('{"summary": "x", "observations": [], "changes": "%s"}' % phrase)
        assert s.changes == "", phrase
    s = parse_scene('{"summary": "x", "observations": [], "changes": "You took off your glasses."}')
    assert s.changes == "You took off your glasses."


def test_normalize_box_variants():
    assert normalize_box([300, 100, 100, 50]) == [0.1, 0.05, 0.3, 0.1]  # swapped corners
    assert normalize_box([0, 0, 1280, 360], 1280, 720) == [0.0, 0.0, 1.0, 0.5]  # absolute pixels
    assert normalize_box([0, 0, 1000, 1000]) is None  # whole frame: nothing to point at
    assert normalize_box([0, 0, 1280, 720]) is None  # pixels but size unknown
    assert normalize_box([10, 10, 10, 10]) is None  # degenerate
    assert normalize_box(["a", 1, 2, 3]) is None
    assert normalize_box(None) is None


def test_compact_list_observations():
    s = parse_scene('{"summary": "You sit.", "changes": "", "observations": [["Black glasses", 100, 200, 300, 400], ["Calm mood"]]}')
    assert [(o.text, o.box) for o in s.observations] == [("Black glasses", [0.1, 0.2, 0.3, 0.4]), ("Calm mood", None)]


def test_partial_reads_only_completed_parts():
    text = '{"summary": "You\'re at a desk.", "changes": "You put on a mask.", "observations": [["Blue mask", 1, 2, 300, 400], ["Your han'
    p = parse_partial(text)
    assert p["summary"] == "You're at a desk." and p["changes"] == "You put on a mask."
    assert [o["text"] for o in p["observations"]] == ["Blue mask"]
    assert parse_partial('{"summary": "You are sitt') == {"summary": "", "changes": "", "observations": [], "repeats": 0}


def test_partial_handles_escaped_quotes_and_brackets_in_text():
    p = parse_partial('{"summary": "x", "observations": [["A \\"quoted\\" [word]", 1, 2, 300, 400], "Plain string", ["Next')
    assert [o["text"] for o in p["observations"]] == ['A "quoted" [word]', "Plain string"]


def test_reply_stopped_early_still_parses():
    text = '{"summary": "You sit.", "changes": "", "observations": [["Glasses", 1, 2, 300, 400], ["Shirt", 5, 6, 700, 800]'
    s = parse_scene(text)
    assert s.parsed and s.summary == "You sit." and [o.text for o in s.observations] == ["Glasses", "Shirt"]


def test_packed_and_repeated_observations():
    s = parse_scene('{"summary": "x", "observations": [["Glasses", "Shirt", "Earbud", "Earbud", "Earbud"]]}')
    assert [o.text for o in s.observations] == ["Glasses", "Shirt", "Earbud"]
    p = parse_partial('{"summary": "x", "observations": [["Mask", 1, 2, 300, 400], ["Mask", 1, 2, 300, 400], ["Ne')
    assert [o["text"] for o in p["observations"]] == ["Mask"] and p["repeats"] == 1
