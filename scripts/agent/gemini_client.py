"""공용 Gemini 클라이언트. AI Studio API 키(.env의 GEMINI_API_KEY)를 사용한다."""

import functools
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import errors

BASE_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(BASE_DIR / ".env")

DECISION_MODEL = "gemini-flash-lite-latest"
# gemini-2.5-flash는 무료 티어 일일 한도(20회)를 이미 소진해 flash-lite로 전환.
# 한도가 초기화되거나 유료 플랜으로 전환하면 "gemini-2.5-flash"로 되돌리면 된다.
EMBEDDING_MODEL = "gemini-embedding-001"

_client = None


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client()  # GEMINI_API_KEY 환경변수 자동 인식
    return _client


def with_rate_limit_retry(fn):
    """무료 티어 분당 호출 한도(429)에 걸리면 잠깐 쉬었다 재시도한다."""

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        for attempt in range(5):
            try:
                return fn(*args, **kwargs)
            except errors.ClientError as e:
                message = str(e)
                if "PerDay" in message:
                    # 일일 무료 할당량 초과 — 재시도해도 소용없음, 즉시 실패
                    raise
                if e.code == 429 and attempt < 4:
                    wait = 15 * (attempt + 1)
                    print(f"  (무료 티어 분당 rate limit — {wait}초 대기 후 재시도 {attempt + 1}/4)")
                    time.sleep(wait)
                    continue
                raise

    return wrapper
