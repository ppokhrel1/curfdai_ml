from pydantic_settings import BaseSettings
import os
import torch
from pydantic import Field

class Config(BaseSettings):
    # Database
    DATABASE_URL: str = "postgresql+asyncpg://user:pass@localhost:5432/ragdb"
    
    # Supabase
    SUPABASE_URL: str = ""
    SUPABASE_KEY: str = ""
    
    # Redis
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # RunPod
    RUNPOD_API_KEY: str = ""
    RUNPOD_ENDPOINT_ID: str = "0u14kaulxa76s6"
    
    # Models
    LLM_MODEL: str = "qwen:3.5b-instruct-q4"
    SDXL_MODEL: str = "stabilityai/stable-diffusion-xl-base-1.0"
    HUNYUAN_LOCAL_PATH: str = "workspace/models/hunyuan3d-2"
    HUNYUAN_MODEL: str = HUNYUAN_LOCAL_PATH + "/" + 'hunyuan3d-dit-v2-mv-turbo'
    HUNYUAN3D_TEXT_TO_IMG_LOCAL_PATH: str = HUNYUAN_LOCAL_PATH + "/" + 'hunyuan_text_to_img'
    
    SUPABASE_URL = os.environ.get("SUPABASE_URL", "")
    SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "")
    DEVICE: str = Field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    # Paths
    OUTPUT_DIR: str = "/app/output"
    
    # VRAM settings (RTX A5000 24GB)
    MAX_VRAM_MB: float = 22000
    SAFETY_MARGIN_MB: float = 500

config = Config()