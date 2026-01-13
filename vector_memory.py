from mem0 import Memory
import os
import time
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not set")

if not QDRANT_API_KEY:
    raise RuntimeError("QDRANT_API_KEY not set")


def create_memory():
    config = {
        "embedder": {
            "provider": "openai",
            "config": {
                "model": "text-embedding-3-small"
            }
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {
                "url": "https://5a45e681-6bcd-49f6-a1c5-96f20b1eb272.europe-west3-0.gcp.cloud.qdrant.io",
                "api_key": QDRANT_API_KEY,
                "collection_name": "agentic_ai"
            }
        }
    }

    return Memory.from_config(config)


memory = create_memory()


def mem0_add(
    *,
    user_id: str,
    content: str,
    memory_type: str,
    confidence: float,
    source: str
):
    text = (
        f"[type={memory_type}] "
        f"[confidence={confidence}] "
        f"[source={source}] "
        f"{content}"
    )

    try:
        memory.add(text, user_id=user_id)
        return {"response": "added successfully"}
    except Exception as e:
        return {"response": str(e)}


def mem0_search(
    *,
    user_id: str,
    query: str,
    limit: int = 5
):
    response = memory.search(
        query,
        user_id=user_id,
        limit=limit
    )

    return [
        {
            "memory": r.get("memory"),
            "score": r.get("score")
        }
        for r in response.get("results", [])
    ]

