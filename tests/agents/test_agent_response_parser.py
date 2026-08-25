"""Parser recovery for local-model XML tool-call output."""

from __future__ import annotations

from src.agents.utils import parse_agent_turn, parse_xml_tool_calls


def test_parse_xml_tool_call_without_args():
    parsed = parse_xml_tool_calls(
        '<tool_call><function=git_diff></function></tool_call>'
    )
    assert parsed == {
        "tool": "git_diff",
        "args": {},
        "reasoning": "",
    }


def test_parse_xml_tool_call_batch():
    parsed = parse_xml_tool_calls(
        '<tool_call><function=git_diff></function></tool_call>'
        '<tool_call><function=list_directory></function></tool_call>'
    )
    assert parsed == {
        "calls": [
            {"tool": "git_diff", "args": {}, "reasoning": ""},
            {"tool": "list_directory", "args": {}, "reasoning": ""},
        ]
    }


def test_parse_xml_tool_call_with_json_payload():
    parsed = parse_xml_tool_calls(
        '<tool_call>{"name":"read_file","arguments":{"file_path":"app.py"}}</tool_call>'
    )
    assert parsed == {
        "tool": "read_file",
        "args": {"file_path": "app.py"},
        "reasoning": "",
    }


def test_parse_agent_turn_prefers_json_then_xml():
    json_parsed = parse_agent_turn(
        '{"tool":"git_diff","args":{},"reasoning":"Inspect diff."}'
    )
    assert json_parsed == {
        "tool": "git_diff",
        "args": {},
        "reasoning": "Inspect diff.",
    }

    xml_parsed = parse_agent_turn(
        "Review the diff first.\n"
        '<tool_call><function=git_diff></function></tool_call>'
    )
    assert xml_parsed == {
        "tool": "git_diff",
        "args": {},
        "reasoning": "",
    }
