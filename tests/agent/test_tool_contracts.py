"""Descriptions cannot hide local mutations or expand child permissions."""
from app.agent.tool_agent import AgentTool
from app.agent.tool_contracts import capability_contract, with_contract_descriptions
from app.agent.delegated_research import RESEARCH_READ_TOOLS


def test_local_persona_and_public_pixels_are_not_described_as_pure_reads():
    persona = AgentTool('set_group_persona', 'switch current persona', {}, None)
    pixels = AgentTool('observe_post_images', 'read real public pixels', {}, None)
    assert capability_contract(persona)['side_effects'] == ['local_state']
    assert capability_contract(pixels)['side_effects'] == ['save_artifact']
    described, = with_contract_descriptions((persona,))
    assert described.effect == persona.effect == 'read'
    assert 'local_state' in described.description
    assert persona.name not in RESEARCH_READ_TOOLS and pixels.name not in RESEARCH_READ_TOOLS
