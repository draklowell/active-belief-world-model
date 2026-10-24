import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from agents.qwen3vl import Qwen3VLAgent
from prompts import render

SNAPSHOTS = json.loads(
    (Path(__file__).parent / "fixtures" / "prompts" / "qwen3vl.json").read_text(encoding="utf-8")
)


@pytest.fixture(autouse=True)
def configure_local_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide dummy client settings without calling an inference endpoint."""
    monkeypatch.setenv("OPENAI_BASE_URL", "http://127.0.0.1:1/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


@pytest.mark.parametrize("snapshot", SNAPSHOTS, ids=[item["id"] for item in SNAPSHOTS])
def test_agent_prompts_and_messages_match_pre_migration_snapshot(snapshot: dict) -> None:
    """Preserve prompt bytes, roles, history pairing, and image placement."""
    agent = Qwen3VLAgent(
        {"coordinate_type": snapshot["coordinate_type"], "history_n": snapshot["history_n"]}
    )
    agent.actions = snapshot["actions"].copy()
    agent.responses = snapshot["responses"].copy()
    agent.screenshots = snapshot["screenshots"].copy()
    history = render(
        "qwen3vl/history.j2",
        actions=agent.actions[snapshot["history_start"] :],
        start_step=snapshot["history_start"] + 1,
    )
    description = agent._build_description_prompt(1280, 720)
    system = agent._build_system_prompt(description)
    assert history == snapshot["previous_actions"]
    assert description == snapshot["description"]
    assert system == snapshot["system"]
    assert (
        agent._build_messages(snapshot["instruction"], "current-image", system, history)
        == snapshot["messages"]
    )


def test_predict_preserves_history_and_action_execution_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use rendered instructions over successive predictions without changing actions."""
    agent = Qwen3VLAgent({"history_n": 2})
    requests: list[list[dict]] = []
    responses = iter(
        [
            'Action: Wait\n<tool_call>{"name":"computer_use",'
            '"arguments":{"action":"wait","time":1}}</tool_call>',
            'Action: Finish\n<tool_call>{"name":"computer_use",'
            '"arguments":{"action":"terminate","status":"success"}}</tool_call>',
        ]
    )

    def capture_request(messages: list[dict]) -> str:
        """Capture the real assembled messages at the network boundary."""
        requests.append(messages)
        return next(responses)

    monkeypatch.setattr(agent, "call_llm", capture_request)
    buffer = BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, format="PNG")
    observation = {"screenshot": buffer.getvalue()}
    assert agent.predict("Keep {{ literal }} & <tag>\n", observation)[1] == ["WAIT"]
    assert agent.predict("Keep {{ literal }} & <tag>\n", observation)[1] == ["DONE"]
    assert [message["role"] for message in requests[1]] == ["system", "user", "assistant", "user"]
    instruction = requests[1][1]["content"][1]["text"]
    assert "Keep {{ literal }} & <tag>\n" in instruction
    assert "Previous actions:\nStep 1: Action: Wait" in instruction
    assert [part["type"] for part in requests[1][-1]["content"]] == ["image_url"]
    assert agent.actions == ["Action: Wait", "Action: Finish"]
    assert len(agent.screenshots) == 2
    assert len(agent.responses) == 2
