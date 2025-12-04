import torch
import gc
import asyncio
import logging
from typing import Optional, Protocol

logger = logging.getLogger(__name__)

class ModelService(Protocol):
    async def load(self) -> None: ...
    async def unload(self) -> None: ...
    def get_memory_usage_mb(self) -> float: ...

class ModelManager:
    """Sequential GPU loader for 24GB VRAM constraint"""
    
    def __init__(self, max_vram_mb: float = 22000):
        self.max_vram_mb = max_vram_mb
        self.loaded_service: Optional[ModelService] = None
        self.lock = asyncio.Lock()
    
    async def load_model(self, service: ModelService):
        async with self.lock:
            if self.loaded_service == service:
                return
            
            if self.loaded_service:
                await self._unload_current()
            
            required = service.get_memory_usage_mb()
            if not self._has_enough_vram(required):
                raise RuntimeError(f"Insufficient VRAM: need {required}MB, have {self._get_free_vram_mb()}MB")
            
            await service.load()
            self.loaded_service = service
            logger.info(f"Loaded {service.__class__.__name__}")
    
    async def _unload_current(self):
        if not self.loaded_service:
            return
        
        await self.loaded_service.unload()
        self.loaded_service = None
        
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("Unloaded model and cleared VRAM")
    
    def _has_enough_vram(self, required: float) -> bool:
        return self._get_free_vram_mb() > required + 500
    
    def _get_free_vram_mb(self) -> float:
        if not torch.cuda.is_available():
            return 0
        return torch.cuda.mem_get_info()[0] / 1024 / 1024
    
    async def unload_all(self):
        async with self.lock:
            await self._unload_current()