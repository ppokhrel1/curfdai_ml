from supabase import create_client, Client
from typing import Optional

class SupabaseStorage:
    def __init__(self, url: str, key: str):
        self.client: Optional[Client] = None
        if url and key:
            self.client = create_client(url, key)
    
    async def upload(self, file_bytes: bytes, filename: str, content_type: str) -> str:
        if not self.client:
            raise RuntimeError("Supabase not configured")
        
        self.client.storage.from_("generated_models").upload(
            path=filename,
            file=file_bytes,
            file_options={"content-type": content_type}
        )
        return self.client.storage.from_("generated_models").get_public_url(filename)