import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key")
INPUT_FILE_PATH = os.path.join("data", "input.txt")
EMBEDDINGS_FILE_PATH = os.path.join("data", "embeddings.json")
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536
CHAT_MODEL = "gpt-4o-mini"
CHAT_INSTRUCTIONS = "You are an intelligent assistant. The user will provide a request or question. You will receive a list of the most relevant extracted passages from the available documents, but these passages may not always be fully relevant. If their relevance is unclear or additional details are needed, ask the user for clarification. Always specify exactly what you are referring to, including any relevant details such as names, dates, amounts, or other key information. If you do not have directly relevant information or a precise answer, clearly state that."
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
