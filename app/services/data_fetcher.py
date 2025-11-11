"""
Data Fetcher Service

Fetches messages from external API with:
- Pagination support
- Disk caching for performance
- Robust error handling and retries
- Fallback to cached data on failures
"""

import httpx
import asyncio
import json
import os
import time
import warnings
from typing import List, Dict, Any, Tuple
from app.config import settings

# Suppress SSL warnings for problematic API certificate
warnings.filterwarnings('ignore', message='Unverified HTTPS request')


class DataFetcher:
    """Fetches and caches messages from external API"""
    
    def __init__(self, cache_dir: str = "./data_cache"):
        self.base_url = settings.member_api_url
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "messages_cache.json")
        self.cache_ttl = 3600  # 1 hour
        self.client = None
        os.makedirs(cache_dir, exist_ok=True)
    
    async def fetch_messages(self, use_cache: bool = True) -> List[Dict[str, Any]]:
        """Fetch all messages with caching"""
        # Try cache first
        if use_cache:
            cached = self._load_cache()
            if cached and len(cached) >= 1000:
                print(f"Using cached messages: {len(cached)}")
                return cached
        
        # Fetch from API
        print(f"Fetching messages from API: {self.base_url}/messages/")
        try:
            messages, total = await self._fetch_all_paginated()
            
            if not messages:
                print("❌ No messages fetched from API")
                # Fallback to expired cache
                cached = self._load_cache(allow_expired=True)
                if cached:
                    print(f"✅ Using cached messages as fallback: {len(cached)}")
                    return cached
                raise ValueError("No messages fetched and no cache available")
            
            print(f"✅ Fetched {len(messages)} messages from API")
            
            # Cache if fetch is reasonably complete
            if self._should_cache(messages, total):
                self._save_cache(messages)
            
            return messages
        except Exception as e:
            print(f"❌ Fetch failed: {type(e).__name__}: {e}")
            # Fallback to expired cache
            cached = self._load_cache(allow_expired=True)
            if cached:
                print(f"✅ Using cached messages as fallback: {len(cached)}")
                return cached
            raise
    
    async def close(self):
        """Close HTTP client"""
        if self.client:
            await self.client.aclose()
            self.client = None
    
    # Private methods
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self.client is None:
            # Disable SSL verification for problematic API certificate
            self.client = httpx.AsyncClient(timeout=30.0, verify=False)
        return self.client
    
    async def _fetch_all_paginated(self) -> Tuple[List[Dict[str, Any]], int]:
        """Fetch all messages via pagination"""
        client = await self._get_client()
        all_messages = []
        skip = 0
        limit = 100
        total = None
        max_retries = 3
        consecutive_failures = 0
        
        while True:
            # Try fetching page with retries
            success, items, total = await self._fetch_page(
                client, skip, limit, max_retries
            )
            
            if not success:
                consecutive_failures += 1
                if consecutive_failures >= 3:
                    print(f"Stopping after {consecutive_failures} consecutive failures")
                    break
                skip += limit
                await asyncio.sleep(2.0)
                continue
            
            # Reset failure counter on success
            consecutive_failures = 0
            
            if not items:
                # No more items
                return all_messages, total or len(all_messages)
            
            all_messages.extend(items)
            
            # Check if done
            if total and len(all_messages) >= total:
                return all_messages, total
            
            if len(items) < limit:
                return all_messages, total or len(all_messages)
            
            skip += limit
            print(f"Fetched {len(all_messages)}/{total if total else '?'} messages...")
            await asyncio.sleep(0.2)
        
        return all_messages, total or len(all_messages)
    
    async def _fetch_page(
        self, 
        client: httpx.AsyncClient, 
        skip: int, 
        limit: int, 
        max_retries: int
    ) -> Tuple[bool, List[Dict], int]:
        """
        Fetch a single page with retries
        
        Returns:
            (success, items, total)
        """
        for attempt in range(max_retries):
            try:
                response = await client.get(
                    f"{self.base_url}/messages/",
                    params={"skip": skip, "limit": limit}
                )
                
                if response.status_code == 200:
                    data = response.json()
                    items = data.get("items", [])
                    total = data.get("total", None)
                    return True, items, total
                
                elif response.status_code == 404:
                    # End of data
                    return True, [], None
                
                elif response.status_code in [401, 429]:
                    # Rate limited - retry with backoff
                    if attempt < max_retries - 1:
                        wait = 2.0 * (2 ** attempt)
                        print(f"Rate limited, waiting {wait}s...")
                        await asyncio.sleep(wait)
                        continue
                    return False, [], None
                
                elif response.status_code == 402:
                    # Quota reached
                    print(f"Quota reached at {skip} messages")
                    return True, [], None
                
                else:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(1.0)
                        continue
                    return False, [], None
            
            except Exception as e:
                print(f"Error fetching page (attempt {attempt+1}/{max_retries}): {type(e).__name__}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1.0)
                    continue
                return False, [], None
        
        return False, [], None
    
    def _should_cache(self, messages: List[Dict], total: int) -> bool:
        """Determine if messages should be cached"""
        if not messages:
            return False
        
        # Cache if we got 90%+ of expected
        if total and len(messages) >= total * 0.9:
            return True
        
        # Cache if we got all expected
        if total and len(messages) >= total:
            return True
        
        # Cache if substantial amount (even without total)
        if not total and len(messages) >= 2000:
            return True
        
        return False
    
    def _save_cache(self, messages: List[Dict[str, Any]]):
        """Save messages to disk cache"""
        try:
            data = {
                "messages": messages,
                "timestamp": time.time(),
                "count": len(messages)
            }
            with open(self.cache_file, 'w') as f:
                json.dump(data, f)
            print(f"Cached {len(messages)} messages to disk")
        except Exception as e:
            print(f"Failed to save cache: {e}")
    
    def _load_cache(self, allow_expired: bool = False) -> List[Dict[str, Any]]:
        """Load messages from disk cache"""
        if not os.path.exists(self.cache_file):
            return None
        
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
            
            cache_age = time.time() - data.get("timestamp", 0)
            
            # Check if expired
            if cache_age > self.cache_ttl and not allow_expired:
                return None
            
            messages = data.get("messages", [])
            if messages:
                status = "expired" if cache_age > self.cache_ttl else "valid"
                print(f"Cache {status}: {len(messages)} messages (age: {cache_age:.0f}s)")
                return messages
            
            return None
        except Exception as e:
            print(f"Failed to load cache: {e}")
            return None


# Global instance
data_fetcher = DataFetcher()
