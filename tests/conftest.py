import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, Mock
import tempfile
import json
from pathlib import Path
import sys
from types import ModuleType
import os
from types import SimpleNamespace


def pytest_configure():
    """
    Automatically mock hy3dgen and its submodules before any test imports.
    """

    # ---- CREATE ROOT MODULE ---------------------------------------------------
    hy3dgen = ModuleType("hy3dgen")
    sys.modules["hy3dgen"] = hy3dgen

    # ---- CREATE SUBMODULES ----------------------------------------------------
    shapegen = ModuleType("hy3dgen.shapegen")
    text2image = ModuleType("hy3dgen.text2image")
    rembg = ModuleType("hy3dgen.rembg")

    sys.modules["hy3dgen.shapegen"] = shapegen
    sys.modules["hy3dgen.text2image"] = text2image
    sys.modules["hy3dgen.rembg"] = rembg

    # Attach to root module
    hy3dgen.shapegen = shapegen
    hy3dgen.text2image = text2image
    hy3dgen.rembg = rembg

    # ---- FAKE PIPELINES ------------------------------------------------------
    class FakePipelineBase:
        def __init__(self, *args, **kwargs):
            self.args = args
            self.kwargs = kwargs

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            return {"mock": True, "args": args, "kwargs": kwargs}

        def compile(self):
            return None

        def enable_flashvdm(self, *args, **kwargs):
            return None

    class FakeBackgroundRemover:
        def __call__(self, image):
            # Return the image unchanged (mock)
            return image

    # ---- ASSIGN MOCK CLASSES TO MODULES --------------------------------------
    # 3D generator
    shapegen.Hunyuan3DDiTFlowMatchingPipeline = FakePipelineBase

    # 2D text-to-image model
    text2image.HunyuanDiTPipeline = FakePipelineBase

    # Background remover
    rembg.BackgroundRemover = FakeBackgroundRemover



# Mock environment variables
@pytest.fixture(autouse=True)
def mock_env(monkeypatch):
    """Set secure test environment variables"""
    monkeypatch.setenv("DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    monkeypatch.setenv("SUPABASE_URL", "http://mock.supabase.co")
    monkeypatch.setenv("SUPABASE_KEY", "mock-key")
    monkeypatch.setenv("REDIS_URL", "fakeredis://localhost")
    monkeypatch.setenv("COMFYUI_API_URL", "http://mock-comfyui:8188")
    monkeypatch.setenv("TESTING", "true")  # Disable VRAM checks


@pytest.fixture(autouse=True)
def mock_filesystem():
    """Mock ALL filesystem operations including directory creation"""
    with patch('pathlib.Path.mkdir') as mock_mkdir, \
         patch('pathlib.Path.exists', return_value=True) as mock_exists, \
         patch('pathlib.Path.__truediv__', return_value=Path("/tmp/test")) as mock_div, \
         patch('builtins.open', create=True) as mock_open, \
         patch('os.makedirs') as mock_makedirs:  # THIS IS CRITICAL
        
        # Mock file operations
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.read = MagicMock(return_value=b"fake_image_data")
        mock_file.write = MagicMock()
        mock_open.return_value = mock_file
        
        # Prevent directory creation
        mock_mkdir.return_value = None
        mock_makedirs.return_value = None
        
        yield {
            'mkdir': mock_mkdir,
            'exists': mock_exists,
            'div': mock_div,
            'open': mock_open,
            'makedirs': mock_makedirs
        }

@pytest.fixture(autouse=True)
def env_setup():
    """Set test environment variables"""
    os.environ['TESTING'] = 'true'
    sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_ollama_client():
    """Mock Ollama at module level before import"""
    with patch('app.services.llm_service.ollama.AsyncClient') as mock:
        client = AsyncMock()
        client.pull = AsyncMock(return_value=None)
        client.generate = AsyncMock(return_value={
            "response": json.dumps({
                "model_name": "robot",
                "model_type": "robot",
                "primary_function": "test robot",
                "key_components": ["wheel", "sensor"],
                "mobility_type": "wheeled",
                "environment": "indoor",
                "complexity_level": "simple"
            })
        })
        mock.return_value = client
        yield mock

@pytest.fixture(autouse=True)
def mock_diffusers_pipeline():
    """Mock StableDiffusionPipeline.from_pretrained to return a working fake pipeline."""
    with patch("app.services.image_service.StableDiffusionPipeline.from_pretrained") as mock_from_pretrained:

        mock_pipeline = MagicMock()
        mock_pipeline.to.return_value = mock_pipeline
        mock_pipeline.enable_vae_slicing.return_value = None

        # mock a fake generated image
        mock_image = MagicMock()

        mock_pipeline.__call__ = MagicMock(
            return_value=SimpleNamespace(images=[mock_image])
        )

        mock_from_pretrained.return_value = mock_pipeline

        yield mock_from_pretrained


@pytest.fixture
def mock_supabase():
    """Mock Supabase storage"""
    with patch('app.storage.supabase_client.create_client') as mock:
        client = MagicMock()
        storage = MagicMock()
        storage.upload = MagicMock(return_value={"data": {"path": "test.png"}})
        storage.get_public_url = MagicMock(return_value="http://mock.supabase/test.png")
        storage.from_ = MagicMock(return_value=storage)
        client.storage = storage
        mock.return_value = client
        yield mock

@pytest.fixture
def mock_redis():
    """Mock Redis with fakeredis"""
    import fakeredis
    with patch('app.storage.cache_manager.redis.from_url') as mock:
        fake_client = fakeredis.FakeStrictRedis()
        mock.return_value = fake_client
        yield fake_client

@pytest.fixture
def sample_requirements():
    return {
        "model_name": "robot",
        "model_type": "robot",
        "primary_function": "industrial inspection",
        "key_components": ["camera_sensor", "hexapod_leg"],
        "mobility_type": "legged",
        "environment": "industrial",
        "complexity_level": "moderate"
    }

@pytest.fixture(autouse=True)
def mock_vram():
    """Mock torch.cuda to simulate GPU"""
    mock_cuda = MagicMock()
    mock_cuda.is_available.return_value = True
    mock_cuda.mem_get_info.return_value = (25000 * 1024 * 1024, 25000 * 1024 * 1024)
    
    with patch('torch.cuda', mock_cuda):
        yield


@pytest.fixture
def sample_mesh_bytes():
    """Minimal valid OBJ file bytes"""
    return b"""# Test mesh
v 0.0 0.0 0.0
v 1.0 0.0 0.0
v 1.0 1.0 0.0
f 1 2 3
"""