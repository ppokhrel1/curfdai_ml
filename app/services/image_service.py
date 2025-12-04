import torch
from diffusers import StableDiffusionPipeline
from pathlib import Path
import time

class ImageService:
    def __init__(self, model_id: str, output_dir: str = 'workspace'):
        self.model_id = model_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pipeline = None
        self._memory_mb = 12000
    
    async def load(self):
        self.pipeline = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            cache_dir="/cache"
        ).to("cuda")
        self.pipeline.enable_vae_slicing()
        
    
    async def unload(self):
        self.pipeline = None
    
    def get_memory_usage_mb(self) -> float:
        return self._memory_mb
    
    async def generate(self, prompt: str, width: int = 1024, height: int = 1024) -> str:
        image = self.pipeline(prompt, width=width, height=height, num_inference_steps=30).images[0]
        path = self.output_dir / f"{int(time.time())}.png"
        #image.save(path)
        #save to supabase
        return str(path), image