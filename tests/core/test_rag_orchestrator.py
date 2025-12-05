import pytest
import json
from app.core.rag_orchestrator import SelfHostedRAGSystem
from unittest.mock import AsyncMock, MagicMock, patch, Mock

@pytest.mark.asyncio
async def test_complete_workflow(
    mock_hf_client,
    mock_supabase,
    mock_redis,
    mock_filesystem,
    mock_diffusers_pipeline,
    sample_requirements
):
    rag = SelfHostedRAGSystem()
    
    # Mock image return as a tuple (url, image)
    rag.supabase.upload = AsyncMock(return_value="http://mock.supabase/test.png")
    
    # Create a proper mock for ImageService
    mock_image_service = MagicMock()
    mock_image = MagicMock()
    mock_image_service.generate = AsyncMock(return_value=("/tmp/test.png", mock_image))
    rag._image = mock_image_service
    
    # Mock Hunyuan3DService to return proper structure
    mock_hunyuan_service = MagicMock()
    mock_hunyuan_service.generate_3d_asset = AsyncMock(
        return_value={
            'download_url': 'http://mock/mesh.stl',
            'status': 'success',
            'filename': 'test.stl',
            'format': 'stl',
            'prompt': 'test'
        }
    )
    rag._hunyuan = mock_hunyuan_service
    
    # Create an AsyncMock for LLMService with proper async methods
    mock_llm_service = AsyncMock()  # Use AsyncMock instead of MagicMock
    
    # Mock async methods properly
    mock_llm_service.generate_json = AsyncMock(return_value=sample_requirements)
    mock_llm_service.generate_text = AsyncMock(return_value="<sdf>mock sdf</sdf>")
    mock_llm_service.get_embedding = AsyncMock(return_value=[0.1] * 384)  # Mock embedding vector
    rag._llm = mock_llm_service
    
    # Mock the cache to return None (cache miss)
    rag.cache.get = Mock(return_value=None)
    rag.cache.set = Mock()
    
    # Mock _generate_and_upload_mesh_from_images
    rag._generate_and_upload_mesh_from_images = AsyncMock(
        return_value={"part_name": "camera_sensor", "url": "http://mock/mesh.stl"}
    )
    
    # We need to patch the functions that are called in generate_complete_model
    with patch('app.core.rag_orchestrator._hybrid_search_parts', new_callable=AsyncMock) as mock_hybrid_search, \
         patch('app.core.rag_orchestrator._create_assembly_plan_with_llm', new_callable=AsyncMock) as mock_create_plan, \
         patch('app.core.rag_orchestrator._generate_model_files', new_callable=AsyncMock) as mock_generate_files, \
         patch('app.helpers.llm_prompt_helpers.engine') as mock_engine:
        
        # Set up the mocks
        mock_hybrid_search.return_value = []
        mock_create_plan.return_value = {
            "model_name": "test_model", 
            "description": "Test robot",
            "parts": [], 
            "joints": []
        }
        mock_generate_files.return_value = {
            "model.yaml": "dummy yaml",
            "model.sdf": "<?xml version='1.0'?><sdf><model name='test'></model></sdf>"
        }
        
        # Mock async connection for database
        mock_conn = AsyncMock()
        mock_result = AsyncMock()
        mock_engine.connect.return_value.__aenter__.return_value = mock_conn
        mock_conn.execute.return_value = mock_result
        mock_result.fetchall = AsyncMock(return_value=[])
        
        # Mock the database operations (AssetRepository)
        with patch('app.core.rag_orchestrator.AssetRepository') as mock_repo:
            mock_repo_instance = AsyncMock()
            mock_repo_instance.create_complete_model = AsyncMock(return_value="test_asset_id")
            mock_repo.return_value = mock_repo_instance
            
            result = await rag.generate_complete_model("test robot")
    
    assert "asset_id" in result
    assert result["asset_id"] == "test_asset_id"
    
    # Verify the calls
    mock_llm_service.generate_json.assert_called()

@pytest.mark.asyncio
async def test_cache_hit_shortcuts_llm():
    """Test that generate_complete_model uses cache for requirements"""
    rag = SelfHostedRAGSystem()
    
    # Mock cache to return the FULL cached result as a JSON string
    # This should match what generate_complete_model returns
    full_cached_result = {
        "specification": {
            "model_name": "test_model",
            "description": "Cached robot",
            "parts": [],
            "joints": []
        },
        "model_files": {
            "model.sdf": "<sdf>mock sdf</sdf>",
            "model.config": "<config>mock config</config>"
        },
        "asset_id": "test_asset_id",
        "requirements": {
            "model_type": "cached_robot",
            "key_components": ["wheel", "sensor"],
            "primary_function": "cached",
            "mobility_type": "wheeled",
            "environment": "indoor",
            "complexity_level": "simple"
        },
        "generation_time": 1.5
    }
    
    rag.cache.get = Mock(return_value=json.dumps(full_cached_result))
    rag.cache.set = Mock()
    
    # Create mock image object
    mock_image = MagicMock()
    
    # Mock all other methods with new signatures
    rag._generate_and_upload_image = AsyncMock(
        return_value=("http://mock.url/image.png", mock_image)
    )
    rag._generate_and_upload_mesh_from_images = AsyncMock(
        return_value={"part_name": "wheel", "url": "http://mock.url/mesh.stl"}
    )
    rag._create_assembly_plan = AsyncMock(return_value={"model_name": "test_model"})
    rag._generate_model_files = AsyncMock(return_value={
        "model.sdf": "<sdf>", 
        "model.config": "<config>"
    })
    rag._store_all_assets = AsyncMock(return_value="test_asset_id")
    
    # Mock the LLM to verify it's not called
    mock_llm = AsyncMock()
    mock_llm.generate_json = AsyncMock()
    mock_llm.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
    rag._llm = mock_llm
    
    # Mock image and hunyuan services to avoid lazy loading
    rag._image = MagicMock()
    rag._hunyuan = MagicMock()
    
    # We need to patch the functions that are called in generate_complete_model
    with patch('app.core.rag_orchestrator._hybrid_search_parts', new_callable=AsyncMock) as mock_hybrid_search:
        mock_hybrid_search.return_value = []
        
        with patch('app.core.rag_orchestrator._create_assembly_plan_with_llm', new_callable=AsyncMock) as mock_create_plan:
            mock_create_plan.return_value = {"model_name": "test_model", "parts": [], "joints": []}
            
            with patch('app.core.rag_orchestrator._generate_model_files', new_callable=AsyncMock) as mock_generate_files:
                mock_generate_files.return_value = {
                    "model.yaml": "dummy yaml",
                    "model.sdf": "<?xml version='1.0'?><sdf><model name='test'></model></sdf>"
                }
                
                with patch('app.helpers.llm_prompt_helpers.engine') as mock_engine:
                    mock_conn = AsyncMock()
                    mock_result = AsyncMock()
                    mock_engine.connect.return_value.__aenter__.return_value = mock_conn
                    mock_conn.execute.return_value = mock_result
                    mock_result.fetchall = AsyncMock(return_value=[])
                    
                    result = await rag.generate_complete_model("test robot")
    
    # Verify cache was checked
    rag.cache.get.assert_called()
    
    # Verify LLM was NOT called for requirements (cache hit)
    mock_llm.generate_json.assert_not_called()
    
    # Verify result contains cached requirements from the full result structure
    assert result["requirements"]["model_type"] == "cached_robot"
    assert len(result["requirements"]["key_components"]) == 2
    assert result["asset_id"] == "test_asset_id"
    assert "specification" in result
    assert "model_files" in result
    assert "generation_time" in result

@pytest.mark.asyncio
async def test_cache_miss_calls_llm():
    """Test that cache miss triggers LLM call"""
    rag = SelfHostedRAGSystem()
    
    # Mock cache to return None (cache miss)
    rag.cache.get = Mock(return_value=None)
    rag.cache.set = Mock()
    
    # Create mock image object
    mock_image = MagicMock()
    
    # Create mock LLM that will be called - USE AsyncMock
    mock_llm = AsyncMock()
    mock_llm.generate_json = AsyncMock(return_value={
        'model_name': 'robot',
        "model_type": "robot",
        "key_components": ["wheel", "sensor"],
        "primary_function": "test",
        "mobility_type": "wheeled",
        "environment": "indoor",
        "complexity_level": "simple"
    })
    mock_llm.generate_text = AsyncMock(return_value="""model:
                                       name: hello
                                       links: abc
                                       joints: a""")  # Add this
    mock_llm.get_embedding = AsyncMock(return_value=[0.1, 0.2, 0.3])
    rag._llm = mock_llm
    
    # Mock all other methods with new signatures
    rag._generate_and_upload_image = AsyncMock(
        return_value=("http://mock.url/image.png", mock_image)
    )
    rag._generate_and_upload_mesh_from_images = AsyncMock(
        return_value={"part_name": "wheel", "url": "http://mock.url/mesh.stl"}
    )
    rag._create_assembly_plan = AsyncMock(return_value={"model_name": "test_model"})
    rag._generate_model_files = AsyncMock(return_value={
        "model.sdf": "<sdf>",
        "model.config": "<config>"
    })
    rag._store_all_assets = AsyncMock(return_value="test_asset_id")
    
    # Mock image and hunyuan services
    rag._image = MagicMock()
    rag._hunyuan = MagicMock()
    
    # Patch the standalone functions that will be imported and called
    with patch('app.core.rag_orchestrator._hybrid_search_parts', new_callable=AsyncMock) as mock_hybrid_search:
        mock_hybrid_search.return_value = []
        
        with patch('app.core.rag_orchestrator._create_assembly_plan_with_llm', new_callable=AsyncMock) as mock_create_plan:
            mock_create_plan.return_value = {
                "model_name": "test_model", 
                "description": "Test robot",
                "parts": [], 
                "joints": []
            }
            
            # Mock _generate_yaml_content to return valid YAML
            with patch('app.helpers.llm_prompt_helpers._generate_yaml_content', new_callable=AsyncMock) as mock_generate_yaml:
                mock_generate_yaml.return_value = """
model:
  name: "test_model"
  version: "1.0"
  description: "Test robot"
  links:
    - name: "base_link"
      pose: [0, 0, 0, 0, 0, 0]
      visual:
        geometry:
          type: "mesh"
          uri: "package://model_library/meshes/base.stl"
      collision:
        geometry:
          type: "mesh"
          uri: "package://model_library/meshes/base.stl"
      inertial:
        mass: 1.0
        inertia: [0.1, 0, 0, 0.1, 0, 0.1]
  joints: []
  plugins: []
"""
                
                # Mock database engine
                with patch('app.helpers.llm_prompt_helpers.engine') as mock_engine:
                    mock_conn = AsyncMock()
                    mock_result = AsyncMock()
                    mock_engine.connect.return_value.__aenter__.return_value = mock_conn
                    mock_conn.execute.return_value = mock_result
                    mock_result.fetchall = AsyncMock(return_value=[])
                    
                    # Mock upload_to_supabase
                    rag.upload_to_supabase = Mock(return_value="http://mock.url/file")
                    #rag._store_generated_assets = Mock(return_value="test_asset_id")
                    rag.use_cache = True
                    
                    result = await rag.generate_complete_model("test robot")
    
    
    # Verify cache was checked (miss)
    rag.cache.get.assert_called()
    
    # Verify LLM WAS called for requirements (cache miss)
    mock_llm.generate_json.assert_called()
    
    # Verify cache was set with the new requirements
    rag.cache.set.assert_called()

@pytest.mark.asyncio
async def test_vram_sequential_loading(mock_hf_client, mock_vram):
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

@pytest.mark.asyncio
async def test_generate_and_upload_image():
    """Test the updated _generate_and_upload_image method"""
    rag = SelfHostedRAGSystem()
    
    # Mock the image service
    mock_image_service = MagicMock()
    mock_image = MagicMock()
    mock_image_service.generate = AsyncMock(return_value=("/tmp/test.png", mock_image))
    rag._image = mock_image_service
    
    # Mock supabase upload
    rag.supabase.upload = AsyncMock(return_value="http://mock.supabase/test.png")
    
    # Test the method
    url, image = await rag._generate_and_upload_image("test prompt")
    
    # Verify results
    assert url == "http://mock.supabase/test.png"
    assert image == mock_image
    
    # Verify the calls
    mock_image_service.generate.assert_called_once_with("test prompt", width=1024, height=1024)
    rag.supabase.upload.assert_called_once()