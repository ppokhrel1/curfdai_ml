import logging
import uuid
import json
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import asyncio

from app.helpers.llm_prompt_helpers import _analyze_requirements_with_llm, _create_assembly_plan_with_llm, _generate_model_files, _hybrid_search_parts
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
        
        start_time = time.time()
        
        # Check cache first
        
        cached_result = self.cache.get(cache_key)
        if cached_result:
            logger.info("Returning cached result")
            return json.loads(cached_result)
        
        try:
            llm = await self.llm()
            # Step 1: LLM-based requirement analysis
            requirements = await _analyze_requirements_with_llm(llm, user_prompt)
            if isinstance(requirements, list) and requirements:
                requirements = requirements[0]            

            relevant_parts = await _hybrid_search_parts(llm, requirements)            

            assembly_plan = await _create_assembly_plan_with_llm(llm, requirements, relevant_parts)
            # Ensure assembly_plan is dict, in case LLM added extra text
            if isinstance(assembly_plan, list) and assembly_plan:
                assembly_plan = assembly_plan[0]
            
            logger.info(f"Created assembly plan with {len(assembly_plan.get('parts', []))} parts")
            
            # Step 4: Generate model files (XACRO, config, and ERB)
            model_files = await _generate_model_files(llm, assembly_plan)
            logger.info(f"Generated model files: {list(model_files.keys())}")
            
            model_name_prefix = assembly_plan['model_name'] # Use this for Supabase path
            
            # Step 5: Store generated assets and link meshes
            asset_id = await self._store_all_assets(assembly_plan, model_files)
            
            uploaded_files = {}
            
            for filename, content in model_files.items():
                content_bytes = content.encode('utf-8')
                content_type = 'text/plain'
                
                download_url = self.supabase.upload(
                    file_bytes=content_bytes, 
                    filename=f"{model_name_prefix}/{filename}", 
                    content_type=content_type
                )
                uploaded_files[filename] = download_url
            spec_content_bytes = json.dumps(assembly_plan, indent=2).encode('utf-8')
            spec_filename = f"{model_name_prefix}/specification.json"
            self.supabase.upload(
                file_bytes=spec_content_bytes, 
                filename=spec_filename, 
                content_type="application/json"
            )
            
            result = {
                "specification": assembly_plan,
                "model_files": model_files,
                "asset_id": asset_id,
                "requirements": requirements,
                "generation_time": time.time() - start_time
            }
            
            # Cache result
            try:
                self.cache.set(cache_key, 3600, json.dumps(result))  #cache for 1 hour
            except Exception as e:
                logger.warning(f"Cache write failed: {e}")
        
            return result
                
        except Exception as e:
            logger.error(f"Model generation failed: {e}")
            raise
    
    async def _analyze_requirements(self, prompt: str) -> Dict[str, Any]:
        system_prompt = """You are a mechanical engineering expert. Extract technical requirements from descriptions.
        Return ONLY valid JSON with fields: model_type, primary_function, key_components (list), 
        mobility_type, environment, size_constraints, performance_requirements, target_simulator, complexity_level."""
        
        llm = await self.llm()
        return await llm.generate_json(f"{system_prompt}\n\nPrompt: {prompt}")
    
    async def _generate_and_upload_image(self, prompt: str) -> Tuple[str, Any]:
        image_path, image = await (await self.image()).generate(prompt, width=1024, height=1024)
        url = await self.supabase.upload(
            image, f"images/{uuid.uuid4()}.png", "image/png"
        )
        return url, image

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
    
    async def _generate_and_upload_mesh_from_images(self, images, part_name: str) -> Dict[str, Any]:
        results = await (await self.hunyuan()).generate_3d_asset({
            "images": images,
            "type": "images",
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
        meshes: List[Dict] = []
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