from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MXBAI_API_KEY = os.getenv("MXBAI_API_KEY")

EMBEDDING_MODEL = "mixedbread-ai/mxbai-embed-large-v1"
LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

VECTOR_DB_PATH = "output/vectordb"
IMAGE_OUTPUT_PATH = "output/images"

TOP_K = 5