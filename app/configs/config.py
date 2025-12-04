from pydantic_settings import BaseSettings

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
    HUNYUAN_MODEL: str = "hunyuan/hunyuan3d-shape-v2-1"
    
    # Paths
    OUTPUT_DIR: str = "/app/output"
    
    # VRAM settings (RTX A5000 24GB)
    MAX_VRAM_MB: float = 22000
    SAFETY_MARGIN_MB: float = 500

config = Config()