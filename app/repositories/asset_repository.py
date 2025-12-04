from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from typing import List, Dict, Any, Optional
from models.db_models import AssetFile, GeneratedAsset
import json
import uuid

class AssetRepository:    
    def __init__(self, session: AsyncSession):
        self.session = session
    
    async def create_asset_file(
        self, 
        bucket: str, 
        path: str, 
        file_type: str, 
        filesize: int, 
        metadata: dict
    ) -> AssetFile:
        asset_file = AssetFile(
            bucket=bucket,
            path=path,
            file_type=file_type,
            filesize=filesize,
            metadata=metadata
        )
        self.session.add(asset_file)
        await self.session.flush()
        return asset_file
    
    async def create_generated_asset(
        self,
        resource_type: str,
        resource_id: str,
        target_format: str,
        file_id: str,
        metadata: dict
    ) -> GeneratedAsset:
        asset = GeneratedAsset(
            resource_type=resource_type,
            resource_id=resource_id,
            target_format=target_format,
            file_id=file_id,
            metadata=metadata
        )
        self.session.add(asset)
        await self.session.flush()
        return asset
    
    async def get_meshes_by_pattern(self, pattern: str) -> List[Dict[str, Any]]:
        stmt = select(AssetFile.id, AssetFile.path, AssetFile.file_type).where(
            and_(
                AssetFile.file_type.in_(['stl', 'obj', 'glb']),
                AssetFile.path.ilike(f"%{pattern}%")
            )
        ).limit(50)
        
        result = await self.session.execute(stmt)
        return [dict(row._mapping) for row in result]
    
    async def create_complete_model(
        self,
        model_name: str,
        files: Dict[str, str],
        assembly_plan: dict,
        meshes: List[Dict[str, Any]]
    ) -> str:
        model_id = str(uuid.uuid4())
        
        # Create asset files and links
        for filename, content in files.items():
            asset_file = await self.create_asset_file(
                bucket='generated-models',
                path=f"{model_name}/{filename}",
                file_type=filename.split('.')[-1],
                filesize=len(content),
                metadata=assembly_plan
            )
            
            await self.create_generated_asset(
                resource_type='model',
                resource_id=model_id,
                target_format=self._determine_format(filename),
                file_id=asset_file.id,
                metadata=assembly_plan
            )
        
        # Link meshes
        for mesh in meshes:
            await self.create_generated_asset(
                resource_type='mesh_link',
                resource_id=model_id,
                target_format=mesh['format'],
                file_id=mesh['file_id'],
                metadata={"part_name": mesh['name']}
            )
        
        return model_id
    
    def _determine_format(self, filename: str) -> str:
        if filename.endswith('.xacro'): return 'xacro'
        elif filename.endswith('.sdf'): return 'sdf'
        elif 'config' in filename: return 'config'
        return 'unknown'