import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the repo root (two levels up from this file)
_env_path = Path(__file__).resolve().parents[3] / ".env"
load_dotenv(_env_path)

model_config = {
    "openrouter": {
        "api_key": os.environ.get("OPENROUTER_API_KEY", "Your OpenRouter API Key"),
        "base_url": "https://openrouter.ai/api/v1",
    },
    "openai": {
        "api_key": os.environ.get("OPENAI_API_KEY"),
        "base_url": None,
    },
}
