import pytest
from app.services.image_service import ImageService


@pytest.mark.asyncio
async def test_comfyui_workflow_loading():
    service = ImageService("http://mock:1188")
    service.model_id == "http://mock:1188"