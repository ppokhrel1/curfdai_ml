import logging
import uuid
import json
import time
from pathlib import Path
from typing import Dict, Any, List
import asyncio

from app.services.llm_service import LLMService
from app.services.image_service import ImageService
from app.services.hunyuan3d_service import Hunyuan3DService
from app.storage.supabase_client import SupabaseStorage
from app.storage.cache_manager import CacheManager
from app.core.model_manager import ModelManager
from app.repositories.asset_repository import AssetRepository
from app.configs.config import config
from app.db.engine import DatabaseManager
logger = logging.getLogger(__name__)

class SelfHostedRAGSystem:
    def __init__(self):
        # Infrastructure
        self.supabase = SupabaseStorage(config.SUPABASE_URL, config.SUPABASE_KEY)
        self.cache = CacheManager(config.REDIS_URL)
        self.model_manager = ModelManager(config.MAX_VRAM_MB)
        
        # Lazy-loaded services
        self._llm: LLMService = None
        self._image: ImageService = None
        self._hunyuan: Hunyuan3DService = None
    
    async def llm(self) -> LLMService:
        if not self._llm:
            self._llm = LLMService(config.LLM_MODEL)
            await self.model_manager.load(self._llm)
        return self._llm
    
    async def image(self) -> ImageService:
        if not self._image:
            self._image = ImageService(config.SDXL_MODEL, f"{config.OUTPUT_DIR}/images")
            await self.model_manager.load(self._image)
        return self._image
    
    async def hunyuan(self) -> Hunyuan3DService:
        if not self._hunyuan:
            self._hunyuan = Hunyuan3DService(config.HUNYUAN_MODEL, f"{config.OUTPUT_DIR}/meshes")
            await self.model_manager.load(self._hunyuan)
        return self._hunyuan
    
    async def generate_complete_model(self, user_prompt: str) -> Dict[str, Any]:
        """Main workflow: LLM → Image → Meshes → Store"""
        logger.info(f"Processing: {user_prompt}")
        
        # 1. Analyze requirements (LLM)
        cache_key = self.cache.get_key(f"req:{user_prompt}")
        if cached := self.cache.get(cache_key):
            requirements = cached
        else:
            requirements = await self._analyze_requirements(user_prompt)
            self.cache.set(cache_key, requirements)
        
        # 2. Generate concept image
        image_url = await self._generate_and_upload_image(
            f"{requirements['primary_function']} {requirements['model_type']}"
        )
        
        # 3. Generate meshes for components
        mesh_tasks = [
            self._generate_and_upload_mesh(part, requirements['primary_function'])
            for part in requirements['key_components']
        ]
        meshes = await asyncio.gather(*mesh_tasks)
        
        # 4. Create assembly plan
        assembly_plan = await self._create_assembly_plan(requirements, meshes)
        
        # 5. Generate model files
        model_files = await self._generate_model_files(assembly_plan)
        
        # 6. Store everything in DB
        asset_id = await self._store_all_assets(assembly_plan, model_files, meshes)
        
        return {
            "asset_id": asset_id,
            "requirements": requirements,
            "image_url": image_url,
            "meshes": meshes,
            "files": list(model_files.keys())
        }
    
    async def _analyze_requirements(self, prompt: str) -> Dict[str, Any]:
        system_prompt = """You are a mechanical engineering expert. Extract technical requirements from descriptions.
        Return ONLY valid JSON with fields: model_type, primary_function, key_components (list), 
        mobility_type, environment, size_constraints, performance_requirements, target_simulator, complexity_level."""
        
        llm = await self.llm()
        return await llm.generate_json(f"{system_prompt}\n\nPrompt: {prompt}")
    
    async def _generate_and_upload_image(self, prompt: str) -> str:
        image_path = await (await self.image()).generate(prompt, width=1024, height=1024)
        with open(image_path, "rb") as f:
            return await self.supabase.upload(f.read(), f"images/{uuid.uuid4()}.png", "image/png")
    
    async def _generate_and_upload_mesh(self, part_name: str, function: str) -> Dict[str, Any]:
        results = await (await self.hunyuan()).generate_3d_asset({
            "prompt": f"3D model of {part_name} for {function}",
            "type": "text",
            "filename": part_name.replace(' ', '_')
        })
        result = results.get('download_url', '')
        if result:
            return {"part_name": part_name, "url": result['download_url']}
        return {"part_name": part_name, "url": None}
    
    async def _create_assembly_plan(self, req: Dict, meshes: List) -> Dict[str, Any]:
        llm = await self.llm()
        prompt = f"""Create assembly plan for {req['primary_function']} with {len(meshes)} components.
        Return JSON with: model_name, parts (list), joints, parameters."""
        return await llm.generate_json(prompt)
    
    async def _generate_model_files(self, plan: Dict) -> Dict[str, str]:
        llm = await self.llm()
        
        # Generate SDF
        sdf_prompt = f"Generate SDF XML for: {json.dumps(plan)}"
        sdf_content = await llm.generate_text(sdf_prompt, max_tokens=4000)
        
        # Generate config
        config_content = f"""<?xml version="1.0"?>
        <model>
            <name>{plan['model_name']}</name>
            <sdf version="1.7">{plan['model_name']}.sdf</sdf>
            <description>Generated model</description>
        </model>"""
        
        return {
            "model.sdf": sdf_content,
            "model.config": config_content
        }
    
    async def _store_all_assets(
        self,
        assembly_plan: Dict,
        files: Dict[str, str],
        meshes: List[Dict]
    ) -> str:
        """Store in PostgreSQL using ORM"""
        
        
        db = DatabaseManager(config.DATABASE_URL)
        async with db.async_session() as session:
            repo = AssetRepository(session)
            
            # Prepare mesh data for linking
            mesh_data = [
                {
                    "file_id": mesh.get("file_id", str(uuid.uuid4())),
                    "format": "stl",
                    "name": mesh["part_name"]
                }
                for mesh in meshes if mesh.get("url")
            ]
            
            model_id = await repo.create_complete_model(
                model_name=assembly_plan['model_name'],
                files=files,
                assembly_plan=assembly_plan,
                meshes=mesh_data
            )
            
            await session.commit()
            return model_id