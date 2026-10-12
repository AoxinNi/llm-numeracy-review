"""Template for api_config.py. Copy to api_config.py; keys are read from the environment.

Never put literal keys in this file or in api_config.py. Set them in your shell instead, e.g. in
~/.zshrc (then run `source ~/.zshrc`):

    export OPENAI_API_KEY="..."      export OPENAI_BASE_URL="..."
    export ANTHROPIC_API_KEY="..."   export ANTHROPIC_BASE_URL="..."
    export GOOGLE_API_KEY="..."      export GOOGLE_BASE_URL="..."
    export LLM_API_KEY="..."         export LLM_BASE_URL="..."      # GLM / Qwen / DeepSeek

Leave a *_BASE_URL unset to use the provider's official endpoint.
"""

import os

# OpenAI / GPT
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "")
OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.4")

# Anthropic / Claude
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
ANTHROPIC_BASE_URL = os.environ.get("ANTHROPIC_BASE_URL", "")
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6")

# Google / Gemini
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
GOOGLE_BASE_URL = os.environ.get("GOOGLE_BASE_URL", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3-flash")

# GLM / Qwen / DeepSeek share one OpenAI-compatible key and base URL
API_KEY = os.environ.get("LLM_API_KEY", "")
BASE_URL = os.environ.get("LLM_BASE_URL", "")
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-5.1")
QWEN_MODEL = os.environ.get("QWEN_MODEL", "qwen3.7-max")
DEEPSEEK_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-pro")
