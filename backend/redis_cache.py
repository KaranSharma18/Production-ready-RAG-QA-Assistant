import redis
import json
import threading
from vector_store import delete_session_embeddings

# Initialize Redis
redis_client = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)

def save_session(session_id, files):
    """Save session details (file names) in Redis with a 30-minute TTL."""
    session_data = {"files": files}
    redis_client.setex(session_id, 1800, json.dumps(session_data))

def get_session(session_id):
    """Retrieve session details from Redis. If session exists, extend TTL."""
    data = redis_client.get(session_id)
    if data:
        redis_client.expire(session_id, 1800)  # Refresh TTL on user activity
        return json.loads(data)
    return None  # If TTL expired, session will be None

def delete_session(session_id):
    """Remove session from Redis and trigger embedding cleanup in Pinecone."""
    redis_client.delete(session_id)
    delete_session_embeddings(session_id)  # Cleanup embeddings

def redis_key_expiry_listener():
    """Continuously listen for session expiration events and clean up embeddings."""
    pubsub = redis_client.pubsub()
    pubsub.psubscribe("__keyevent@0__:expired")  # Listen for key expiration events

    for message in pubsub.listen():
        if message["type"] == "pmessage":
            expired_key = message["data"]
            print(f"Session expired: {expired_key}. Cleaning up embeddings...")
            delete_session_embeddings(expired_key)

# Run expiry listener in a background thread
threading.Thread(target=redis_key_expiry_listener, daemon=True).start()
