import json
from unittest.mock import Mock
import pytest
from app.storage.cache_manager import CacheManager

def test_cache_manager_direct():
    """Test CacheManager methods directly without mocking Redis"""
    # Create CacheManager instance
    cache = CacheManager("fakeredis://localhost")
    
    # Manually set properties to bypass connection logic
    cache.use_cache = True
    cache.client = Mock()
    
    # Setup mock behavior
    cache.client.setex = Mock()
    cache.client.get = Mock(return_value=json.dumps({"value": 123}))
    
    # Test
    cache.set("test_key", {"value": 123})
    result = cache.get("test_key")
    
    assert result == {"value": 123}
    cache.client.setex.assert_called_once()
    cache.client.get.assert_called_once_with("test_key")

def test_cache_miss():
    cache = CacheManager("invalid://url")
    result = cache.get("missing_key")
    assert result is None

def test_cache_key_generation():
    cache = CacheManager("fakeredis://localhost")
    key1 = cache.get_key("same_data")
    key2 = cache.get_key("same_data")
    assert key1 == key2
    assert len(key1) == 32  # MD5 hash length