from godel_engine.llm_json import parse_llm_json_object


def test_parse_llm_json_tolerates_trailing_comma() -> None:
    text = '{"solution": "x", "edit_type": "rewrite", "strategy_note": "y",}'
    data = parse_llm_json_object(text)
    assert data["solution"] == "x"
    assert data["edit_type"] == "rewrite"


def test_parse_llm_json_strips_markdown_fence() -> None:
    text = '```json\n{"solution": "a", "edit_type": "rewrite", "strategy_note": "b"}\n```\n'
    data = parse_llm_json_object(text)
    assert data["solution"] == "a"
