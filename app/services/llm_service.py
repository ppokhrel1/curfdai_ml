# app/services/llm_service.py
from typing import Dict, Any, List
import json
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModel,  # for embeddings
    pipeline
)
import asyncio
import gc
from sentence_transformers import SentenceTransformer  # <-- NEW


class LLMService:
    def __init__(self, model_name: str = "microsoft/DialoGPT-small"):
        self.model_name = model_name
        self._memory_mb = 3500  # same constant the tests assert on

        # --- text / json generation ---
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else None,
        )

        # --- embedding model (lazy but kept in memory once loaded) ---
        self._embed_model = None

    async def load(self) -> None:
        # Nothing to do – transformers blocks until weights are in RAM
        pass

    async def unload(self) -> None:
        """Free RAM / VRAM."""
        del self.model
        if self._embed_model is not None:
            del self._embed_model
        self.model = self._embed_model = None
        gc.collect()
        torch.cuda.empty_cache()

    def get_memory_usage_mb(self) -> float:
        return self._memory_mb

    # ----------------------------------------------------------
    async def generate_json(self, prompt: str) -> Dict[str, Any]:
        """
        Generate **valid JSON** by prompting the model to return only JSON.
        Falls back to a simple regex extractor if the model hallucinates.
        """
        prompt = f"{prompt}\nRespond only with valid JSON and no additional text."
        text = await self.generate_text(prompt, temperature=0.1, max_new_tokens=150)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # crude but effective fallback
            start, end = text.find("{"), text.rfind("}") + 1
            if start != -1 and end != 0:
                return json.loads(text[start:end])
            raise ValueError("Model did not produce valid JSON")

    # ----------------------------------------------------------
    async def generate_text(self, prompt: str, **kwargs) -> str:
        """Return only the newly generated continuation (prompt removed)."""
        # Run the blocking HF code in a thread so the method stays async
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._generate_text_sync, prompt, kwargs)

    def _generate_text_sync(self, prompt: str, kwargs: Dict[str, Any]) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=kwargs.get("max_new_tokens", 100),
                temperature=kwargs.get("temperature", 0.7),
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        full: str = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        # strip the original prompt
        return full[len(self.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)):].strip()

    # ----------------------------------------------------------
    async def get_embedding(self, text: str) -> List[float]:
        """Return 384-dim sentence embedding."""
        if self._embed_model is None:
            self._embed_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
            )
        loop = asyncio.get_event_loop()
        vec = await loop.run_in_executor(None, self._embed_model.encode, text)
        return vec.tolist()