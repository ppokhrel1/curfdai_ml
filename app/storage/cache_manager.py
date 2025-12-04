# storage/cache_manager.py
import redis
import hashlib
import json
from typing import Any, Optional

class CacheManager:
    """Isolated Redis caching operations"""
    
    def __init__(self, redis_url: str):
        self.client: Optional[redis.Redis] = None
        self.use_cache = False
        
        try:
            self.client = redis.from_url(redis_url)
            self.client.ping()
            self.use_cache = True
        except Exception as e:
            print(f"Redis not available: {e}")
    
    def get_key(self, data: Any) -> str:
        """Generate cache key"""
        return hashlib.md5(str(data).encode()).hexdigest()
    
    def get(self, key: str) -> Optional[Any]:
        if not self.use_cache or not self.client:
            return None
        
        try:
            if data := self.client.get(key):
                return json.loads(data)
        except Exception:
            pass
        return None
    
    def set(self, key: str, value: Any, ttl: int = 3600) -> None:
        if not self.use_cache or not self.client:
            return
        
        try:
            self.client.setex(key, ttl, json.dumps(value))
        except Exception:
            pass