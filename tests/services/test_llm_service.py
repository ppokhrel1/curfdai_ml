import pytest
from unittest.mock import AsyncMock, MagicMock
import asyncio
from app.services.llm_service import LLMService

@pytest.mark.asyncio
async def test_llm_generate_json(mock_hf_client):
    service = LLMService("qwen:test")
    
    # Mock the generate_text method since generate_json calls it
    # We need to ensure it returns valid JSON
    mock_response = '{"key_components": ["component1", "component2"], "summary": "test summary"}'
    service.generate_text = AsyncMock(return_value=mock_response)
    
    result = await service.generate_json('{"test": "data"}')
    
    assert isinstance(result, dict)
    assert "key_components" in result
    assert result["key_components"] == ["component1", "component2"]
    assert result["summary"] == "test summary"
    
    # Verify generate_text was called with the right prompt
    expected_prompt = '{"test": "data"}\nRespond only with valid JSON and no additional text.'
    service.generate_text.assert_called_once_with(expected_prompt, temperature=0.1, max_new_tokens=150)

@pytest.mark.asyncio
async def test_llm_memory_usage(mock_hf_client):
    service = LLMService("qwen:test")
    assert service.get_memory_usage_mb() == 3500

@pytest.mark.asyncio
async def test_llm_unload(mock_hf_client):
    service = LLMService("qwen:test")
    # Should not raise
    await service.unload()