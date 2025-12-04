import pytest
import json
from app.core.rag_orchestrator import SelfHostedRAGSystem
from unittest.mock import AsyncMock, MagicMock, patch, Mock

@pytest.mark.asyncio
async def test_complete_workflow(
    mock_ollama_client,
    mock_supabase,
    mock_redis,
    mock_filesystem,
    mock_diffusers_pipeline,
    sample_requirements
):
    rag = SelfHostedRAGSystem()
    rag.supabase.upload = AsyncMock(return_value="http://mock.supabase/test.png")
    
    # Create a proper mock for ImageService
    mock_image_service = MagicMock()
    mock_image_service.generate = AsyncMock(return_value="/tmp/test.png")
    rag._image = mock_image_service  # Set it directly to bypass lazy loading
    
    # Mock Hunyuan3DService to return proper structure
    mock_hunyuan_service = MagicMock()
    mock_hunyuan_service.generate_3d_asset = AsyncMock(
        return_value={'download_url': {'download_url': 'http://mock/mesh.stl'}}
    )
    rag._hunyuan = mock_hunyuan_service
    
    # Mock LLMService
    mock_llm_service = MagicMock()
    mock_llm_service.generate_json = AsyncMock(return_value=sample_requirements)
    mock_llm_service.generate_text = AsyncMock(return_value="<sdf>mock sdf</sdf>")
    rag._llm = mock_llm_service
    
    # Mock the cache to return None (cache miss)
    rag.cache.get = Mock(return_value=None)
    rag.cache.set = Mock()
    
    # Mock the database operations
    with patch('app.core.rag_orchestrator.AssetRepository') as mock_repo:
        mock_repo_instance = AsyncMock()
        mock_repo_instance.create_complete_model = AsyncMock(return_value="test_asset_id")
        mock_repo.return_value = mock_repo_instance
        
        
        result = await rag.generate_complete_model("test robot")
    
    assert "asset_id" in result
    assert result["asset_id"] == "test_asset_id"
    assert len(result["meshes"]) == 2  # Should have 2 meshes from sample_requirements
    
    # Verify the calls
    mock_llm_service.generate_json.assert_called()
    mock_image_service.generate.assert_called_once_with(
        "industrial inspection robot", width=1024, height=1024
    )
    mock_hunyuan_service.generate_3d_asset.assert_called()
    assert mock_hunyuan_service.generate_3d_asset.call_count == 2

@pytest.mark.asyncio
async def test_cache_hit_shortcuts_llm():
    """Test that generate_complete_model uses cache for requirements"""
    rag = SelfHostedRAGSystem()
    
    # Mock cache to return cached requirements
    cached_data = {
        "model_type": "cached_robot",
        "key_components": ["wheel"],
        "primary_function": "cached",
        "mobility_type": "wheeled",
        "environment": "indoor",
        "complexity_level": "simple"
    }
    
    rag.cache.get = Mock(return_value=cached_data)
    rag.cache.set = Mock()
    
    # Mock all other methods
    rag._generate_and_upload_image = AsyncMock(return_value="http://mock.url/image.png")
    rag._generate_and_upload_mesh = AsyncMock(return_value={"part_name": "wheel", "url": "http://mock.url/mesh.stl"})
    rag._create_assembly_plan = AsyncMock(return_value={"model_name": "test_model"})
    rag._generate_model_files = AsyncMock(return_value={"model.sdf": "<sdf>", "model.config": "<config>"})
    rag._store_all_assets = AsyncMock(return_value="test_asset_id")
    
    # Mock the LLM to verify it's not called
    mock_llm = MagicMock()
    mock_llm.generate_json = AsyncMock()
    rag._llm = mock_llm
    
    result = await rag.generate_complete_model("test robot")
    
    # Verify cache was checked
    rag.cache.get.assert_called_once()
    
    # Verify LLM was NOT called for requirements
    mock_llm.generate_json.assert_not_called()
    
    # Verify result contains cached requirements
    assert result["requirements"]["model_type"] == "cached_robot"

@pytest.mark.asyncio
async def test_vram_sequential_loading(mock_ollama_client, mock_vram):
    """Test GPU memory management without real GPU"""
    rag = SelfHostedRAGSystem()
    
    # Create mock services with proper memory usage methods
    mock_llm_service = MagicMock()
    mock_llm_service.get_memory_usage_mb = Mock(return_value=8000)
    mock_llm_service.load = AsyncMock()
    mock_llm_service.unload = AsyncMock()
    
    mock_image_service = MagicMock()
    mock_image_service.get_memory_usage_mb = Mock(return_value=12000)
    mock_image_service.load = AsyncMock()
    mock_image_service.unload = AsyncMock()
    
    # Mock the lazy loader methods
    with patch.object(rag, 'llm', new=AsyncMock(return_value=mock_llm_service)):
        with patch.object(rag, 'image', new=AsyncMock(return_value=mock_image_service)):
            
            # Mock ModelManager methods
            rag.model_manager._has_enough_vram = Mock(return_value=True)
            rag.model_manager._get_free_vram_mb = Mock(return_value=25000)
            
            # Load LLM
            await rag.model_manager.load(mock_llm_service)
            assert rag.model_manager.loaded_service == mock_llm_service
            
            # Load image service (should unload LLM first)
            await rag.model_manager.load(mock_image_service)
            assert rag.model_manager.loaded_service == mock_image_service
            
            # Verify unload was called on LLM
            mock_llm_service.unload.assert_called_once()
