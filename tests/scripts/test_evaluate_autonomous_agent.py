"""The acceptance harness must never escape its synthetic network boundary."""
from datetime import UTC, datetime
import json
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from scripts.evaluate_autonomous_agent import (
    BOT, GROUP, MEMBER, SCENARIOS, EvaluationBudgetExceeded, RequestBudget,
    ScriptedDecisionClient, VirtualGateway, build_parser, synthetic_image_bytes,
    usage_report,
    run_scenario,
)


def test_default_cli_is_offline_and_has_bounded_provider_limits():
    args = build_parser().parse_args([])
    assert args.mode == "offline"
    assert args.max_model_calls == 24
    assert args.max_image_calls == 2
    assert args.env_file.name == ".env"


def test_request_budget_counts_physical_retries_and_image_responses_separately():
    budget = RequestBudget(model_limit=2, image_limit=1)
    text = httpx.Request("POST", "https://synthetic.example.test/v1/responses", json={"tools": []})
    image = httpx.Request("POST", "https://synthetic.example.test/v1/responses", json={"tools": [{"type": "image_generation"}]})
    budget.hook(text)
    budget.hook(text)
    budget.hook(image)
    with pytest.raises(EvaluationBudgetExceeded):
        budget.hook(text)
    with pytest.raises(EvaluationBudgetExceeded):
        budget.hook(image)
    budget.hook(httpx.Request("GET", "https://synthetic.example.test/generated.png"))
    assert budget.report() == {"model_requests": 2, "image_requests": 1,
                               "model_limit": 2, "image_limit": 1, "blocked_requests": 2}


def test_budget_attaches_only_once_and_only_changes_isolated_client_retry_limit():
    budget = RequestBudget(model_limit=1, image_limit=0)
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, json={}))) as http_client:
        client = SimpleNamespace(http_client=http_client, REQUEST_MAX_ATTEMPTS=3)
        budget.attach(client)
        budget.attach(client)
        assert len(http_client.event_hooks["request"]) == 1
        assert len(http_client.event_hooks["response"]) == 1
        assert client.REQUEST_MAX_ATTEMPTS == 1
        http_client.post("https://synthetic.example.test/responses", json={})
        assert budget.response_observations == [{"status_code": 200, "content_type": "application/json"}]
        with pytest.raises(EvaluationBudgetExceeded):
            http_client.post("https://synthetic.example.test/responses", json={})


@pytest.mark.asyncio
async def test_virtual_gateway_rejects_cross_group_account_and_unsupported_actions():
    gateway = VirtualGateway(SCENARIOS[0], datetime.now(UTC))
    with pytest.raises(AssertionError):
        await gateway.call_api("send_group_msg", {"group_id": GROUP + 1})
    with pytest.raises(AssertionError):
        await gateway.call_api("set_group_card", {"group_id": GROUP, "user_id": MEMBER, "card": "x"})
    with pytest.raises(AssertionError):
        await gateway.call_api("send_private_msg", {"user_id": BOT})
    with pytest.raises(AssertionError):
        await gateway.call_api("get_group_member_info", {"user_id": 900000999})


@pytest.mark.asyncio
async def test_virtual_send_receipt_and_unknown_write_never_use_real_network():
    gateway = VirtualGateway(SCENARIOS[0], datetime.now(UTC))
    receipt = await gateway.call_api("send_group_msg", {"group_id": GROUP, "message": []})
    assert receipt["data"]["message_id"].startswith("synthetic-")
    unknown = VirtualGateway(next(row for row in SCENARIOS if row.uncertain), datetime.now(UTC))
    with pytest.raises(TimeoutError):
        await unknown.call_api("send_group_msg", {"group_id": GROUP, "message": []})
    assert len(unknown.calls) == 1


def test_actual_pixel_fixture_has_distinct_visual_targets():
    import io
    picture = Image.open(io.BytesIO(synthetic_image_bytes()))
    assert picture.size == (256, 192)
    assert picture.getpixel((60, 100)) == (255, 0, 0)
    assert picture.getpixel((180, 100)) == (0, 0, 255)
    assert picture.getpixel((0, 0)) == (255, 255, 255)


def test_empty_usage_report_does_not_invent_provider_cost(sqlite_engine):
    report = usage_report(sqlite_engine)
    assert report["records"] == []
    assert report["input_tokens"] == 0
    assert report["billed_cost"] is None
    assert report["cost_status"].startswith("unknown")


def test_scripted_provider_labels_fixture_and_rejects_foreign_private_prompt():
    from scripts.evaluate_autonomous_agent import FOREIGN_MARKER
    client = ScriptedDecisionClient(SCENARIOS[0])
    assert client.model == "scripted-offline-fixture"
    with pytest.raises(AssertionError):
        client.generate_text([FOREIGN_MARKER])
    value = json.loads(client.generate_text(["synthetic group observation"]))
    assert value["action"] == "send_group"


@pytest.mark.asyncio
@pytest.mark.parametrize("name", [row.name for row in SCENARIOS])
async def test_virtual_scenario_runs_production_composition_and_dispatch_ledger(tmp_path, name):
    scenario = next(row for row in SCENARIOS if row.name == name)
    args = build_parser().parse_args(["--mode", "offline"])
    result = await run_scenario(scenario, args, tmp_path, RequestBudget(model_limit=8, image_limit=1))
    assert result["passed"], result
    assert result["usage"]["billed_cost"] is None
    if scenario.permission and not scenario.image_failure:
        assert sum(row["status"] in {"sent", "uncertain"} for row in result["actions"]) == 1
    if name == "group_vision":
        assert result["scripted_model_calls"][0]["images"] == 1
    if name in {"group_draw", "qzone_draw"}:
        assert result["artifacts"][0]["dimensions"] == [256, 192]
        assert result["scripted_model_calls"][1]["images"] == 1
