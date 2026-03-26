import os
import logging
from typing import List, Dict

from openai import OpenAI
from openai.error import AuthenticationError, RateLimitError, APIError

# set the keys

# initiate the open ai key here 
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_MAX_TOKENS = int(os.getenv("MAX_TOKENS", "300"))
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))


DEFAULT_SYSTEM_PROMPT = os.getenv(
    "SYSTEM_PROMPT",
    "You are a helpful backend assistant. "
    "Respond accurately and concisely. "
    "Use a friendly and professional tone. "
    "Do not speculate or fabricate information. "
    "If the request is ambiguous or lacks context, ask for clarification. "
    "Prefer simple explanations over complex wording."
)

if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in environment variables")

# logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# openai client
client = OpenAI(
    api_key=OPENAI_API_KEY,
    timeout=20.0
)

# =========================
# Prompt Construction
# =========================

def build_messages(
    user_input: str,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_input},
    ]

# chat completion function

def chat_completion(
    user_input: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> Dict[str, object]:
    messages = build_messages(user_input, system_prompt)

    logger.info("Sending request to LLM (input_length=%d)", len(user_input))
    logger.debug("Request messages: %s", messages)

    try:
        response = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=DEFAULT_TEMPERATURE,
        )

        content = response.choices[0].message.content

        logger.info("Received response from LLM")
        logger.debug("Response content: %s", content)

        return {
            "content": content,
            "model": OPENAI_MODEL,
            "tokens_used": response.usage.total_tokens
        }

    except AuthenticationError:
        logger.error("Authentication failed: invalid API key")
        raise

    except RateLimitError:
        logger.warning("Rate limit exceeded")
        raise

    except APIError as e:
        logger.error("OpenAI API error: %s", str(e))
        raise

    except Exception as e:
        logger.exception("Unexpected error during LLM request")
        raise
