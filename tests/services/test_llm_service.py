import pytest
from app.services.llm_service import LLMService

@pytest.mark.asyncio
async def test_llm_generate_json(mock_ollama_client):
    service = LLMService("qwen:test")
    result = await service.generate_json('{"test": "data"}')
    
    assert isinstance(result, dict)
    assert "key_components" in result

@pytest.mark.asyncio
async def test_llm_memory_usage():
    service = LLMService("qwen:test")
    assert service.get_memory_usage_mb() == 3500

@pytest.mark.asyncio
async def test_llm_unload():
    service = LLMService("qwen:test")
    # Should not raise
    await service.unload()