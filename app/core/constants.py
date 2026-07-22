"""Application-wide constants.

Values here are truly constant (app name, version string). Anything
tuneable belongs in config.py, not here.
"""

APP_NAME = "medical-rag"
APP_VERSION = "0.1.0"

# Default LLM model identifier for Groq API
DEFAULT_LLM_MODEL = "llama-3.3-70b-versatile"

# Chroma collection name
CHROMA_COLLECTION_NAME = "medical_chunks"

# Chunk ID hash length (first N hex chars of SHA-256)
CHUNK_ID_HASH_LENGTH = 16
