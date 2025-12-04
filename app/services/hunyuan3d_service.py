import torch
import os
import logging
import tempfile
from typing import Dict, Any
from datetime import datetime
from PIL import Image
import trimesh
import numpy as np
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
from hy3dgen.text2image import HunyuanDiTPipeline
from hy3dgen.rembg import BackgroundRemover

from supabase import create_client

from app.configs.config import config


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


HUNYUAN3D_TEXT_TO_IMG_LOCAL_PATH = config.HUNYUAN_LOCAL_PATH + "/" + 'hunyuan_text_to_img'


# Supabase client
def get_supabase_client():
    print("supabase", config.SUPABASE_KEY, config.SUPABASE_URL)
    if not config.SUPABASE_URL or not config.SUPABASE_KEY:
        return None
    try:
        return create_client(config.SUPABASE_URL, config.SUPABASE_KEY)
    except Exception as e:
        logger.error(f"Supabase client failed: {e}")
        return None

_hunyuan_dit_pipeline = None

def get_hunyuan_dit_pipeline():
    global _hunyuan_dit_pipeline
    if _hunyuan_dit_pipeline is None:
        try:
            # You might want to pre-load this model locally if possible, similar to Hunyuan3D
            _hunyuan_dit_pipeline = HunyuanDiTPipeline(model_path=config.HUNYUAN3D_TEXT_TO_IMG_LOCAL_PATH, 
                                                       device=config.DEVICE)
            _hunyuan_dit_pipeline.compile() # Optional, but good for hot-start
            logger.info("HunyuanDiT 2D pipeline loaded.")
        except Exception as e:
            logger.error(f"Failed to load HunyuanDiT 2D pipeline: {e}")
            _hunyuan_dit_pipeline = None
    return _hunyuan_dit_pipeline

class Hunyuan3DService:
    def __init__(self, hunyuan_model: str = None, output_dir: str = None):
        self.model = None
        self.supabase = get_supabase_client()
        self.dit_pipeline = get_hunyuan_dit_pipeline()
        self.output_dir = output_dir or "/app/output/meshes"
        self.hunyuan_model = hunyuan_model or config.HUNYUAN_LOCAL_PATH
        self._memory_mb = 12000

    def load(self):
        if self.model is not None:
            return
        
        # Load model from local path
        self.model_image = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
            self.hunyuan_model,
            #subfolder='hunyuan3d-dit-v2-mini-turbo',
            subfolder = 'hunyuan3d-dit-v2-mv-turbo',
            device_map=config.DEVICE,
            #variant='fp16',
            use_safetensors=False,
            local_files_only=True
        )

        self.model_image.enable_flashvdm(topk_mode='merge')
        # self.model_tex.enable_flashvdm(topk_mode='merge')
        logger.info("Hunyuan3D model loaded from local path")
        

    def upload_to_supabase(self, file_bytes: bytes, filename: str, content_type: str, file_path: str = "generated_models") -> str:
        if not self.supabase:
            raise Exception("Supabase client not available")
        
        # Upload bytes directly to Supabase
        res = self.supabase.storage.from_(file_path).upload(
            filename,
            file_bytes,
            {"content-type": content_type}
        )
        
        if hasattr(res, 'error') and res.error:
            raise Exception(f"Upload failed: {res.error}")
        
        # Return public URL
        return self.supabase.storage.from_("generated_models").get_public_url(filename)

    def generate_3d_asset(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        try:
            self.load_model()
            
            prompt = input_data.get('prompt', '')
            input_type = input_data.get('type', 'text')
            
            file_path = input_data.get('file_path', 'generated_models')
            output_format = input_data.get('format', 'glb').lower() 
            filename = input_data.get('filename', '') + "." + output_format
            

            if not prompt and not input_data.get('image', ''):
                return {"status": "error", "error": "Prompt/Image is required"}

            # Generate 3D model
            gen_args = {
                'output_type': output_format, 
                'num_inference_steps': 5,
                'octree_resolution': 200,
                'num_chunks': 20000,
                'output_type': 'trimesh'
            }
            
            temp_files = []
            temp_paths = []

            if input_type == 'images' and input_data.get('images'):
                gen_args['image'] = input_data['images']
            else:
                prompt_temp = prompt
                logger.info(f"Generating 2D image for prompt: {prompt}")

                generated_image = self.dit_pipeline(prompt=prompt_temp + " front view, white background", seed=input_data.get('seed', 0))
                generated_image1 = self.dit_pipeline(prompt=prompt_temp + " back view, white background", seed=input_data.get('seed', 0))
                
                generated_image_list = {'front': generated_image, 'back': generated_image1}

                print("generated_image_list", generated_image_list)
                for a in generated_image_list:
                    image = generated_image_list[a].convert("RGBA")
                    
                    if image.mode == 'RGB':
                        rembg = BackgroundRemover()
                        image = rembg(image)
                    generated_image_list[a] = image
                
                gen_args['image'] = generated_image_list
                logger.info("2D image generated and passed to 3D pipeline.")
                
            result = self.model_image(**gen_args)
            logger.info(result)

            if isinstance(result, list):
                if not result:
                    return {"status": "error", "error": "3D generation returned an empty list of assets"}
                result = result[0] 
            file_bytes = None
            
            if isinstance(result, trimesh.base.Trimesh):
                try:
                    file_bytes = result.export(file_type=output_format) 
                except Exception as export_e:
                    logger.error(f"Trimesh IN-MEMORY EXPORT failed for format '{output_format}': {export_e}")
                    return {"status": "error", "error": f"Failed to export asset as '{output_format}': {str(export_e)}"}
            
            elif isinstance(result, bytes):
                file_bytes = result
            
            else:
                return {"status": "error", "error": f"Unsupported final result type: {type(result)}"}
            
            if not file_bytes:
                return {"status": "error", "error": "Failed to generate or export model data."}
            
            # Create filename and upload
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_prompt = "".join(c for c in prompt[:30] if c.isalnum() or c in ('-', '_')).rstrip()
            safe_prompt = safe_prompt.replace(' ', '_') if safe_prompt else "model"
            if filename == "":
                filename = f"{safe_prompt}_{timestamp}.{output_format}"
            
            content_types = {
                'stl': 'model/stl',
                'glb': 'model/gltf-binary', 
                'obj': 'model/obj'
            }
            content_type = content_types.get(output_format, 'application/octet-stream')
            
            # Upload the in-memory bytes to Supabase
            download_url = self.upload_to_supabase(file_bytes, filename, content_type, file_path)
            
            return {
                "status": "success",
                "download_url": download_url,
                "filename": filename,
                "format": output_format,
                "prompt": prompt
            }
            
        except Exception as e:
            logger.error(f"3D generation failed: {e}")
            return {"status": "error", "error": str(e)}
        
    def get_memory_usage_mb(self) -> float:
        return self._memory_mb
       