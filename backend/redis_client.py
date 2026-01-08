import redis
import json
from typing import List, Dict, Optional
import os

class RedisBookCache:
    def __init__(self):
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        self.client = redis.from_url(redis_url, decode_responses=True)
        self.books_key = "books:metadata"
    
    def add_book(self, book_id: str, book_name: str, author: str, total_chunks: int = 0, code_chunks: int = 0, indexed_at: str = ""):
        """Add a single book to Redis cache"""
        book_data = {
            "id": book_id,
            "name": book_name,
            "author": author,
            "total_chunks": total_chunks,
            "code_chunks": code_chunks,
            "indexed_at": indexed_at
        }
        # Use hash to store book metadata
        self.client.hset(self.books_key, book_id, json.dumps(book_data))
    
    def get_all_books(self) -> List[Dict]:
        """Retrieve all books from Redis"""
        books_hash = self.client.hgetall(self.books_key)
        return [json.loads(book_data) for book_data in books_hash.values()]
    
    def get_book(self, book_id: str) -> Optional[Dict]:
        """Get a specific book by ID"""
        book_data = self.client.hget(self.books_key, book_id)
        return json.loads(book_data) if book_data else None
    
    def delete_book(self, book_id: str):
        """Remove a book from cache"""
        self.client.hdel(self.books_key, book_id)
    
    def clear_all_books(self):
        """Clear all books from cache"""
        self.client.delete(self.books_key)
    
    def book_exists(self, book_id: str) -> bool:
        """Check if a book exists in cache"""
        return self.client.hexists(self.books_key, book_id)

# Global instance
redis_cache = RedisBookCache()
