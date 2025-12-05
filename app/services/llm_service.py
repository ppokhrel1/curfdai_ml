from typing import Dict, Any, List
import json
import ollama

class LLMService:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = ollama.AsyncClient()
        self._memory_mb = 3500
    
    async def load(self):
        await self.client.pull(self.model_name)
        
    
    async def unload(self):
        pass
    
    def get_memory_usage_mb(self) -> float:
        return self._memory_mb
    
    async def generate_json(self, prompt: str) -> Dict[str, Any]:
        response = await self.client.generate(
            model=self.model_name,
            prompt=prompt,
            format="json",
            options={"temperature": 0.1}
        )
        return json.loads(response["response"])
    
    async def generate_text(self, prompt: str, **kwargs) -> str:
        response = await self.client.generate(
            model=self.model_name,
            prompt=prompt,
            options=kwargs
        )
        return response["response"]
    
    async def get_embedding(self, text: str) -> List[float]:
        response = await self.client.embeddings(
            model=self.model_name,
            prompt=text
        )
        return response["embedding"]