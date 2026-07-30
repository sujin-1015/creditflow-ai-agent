"""ADK 에이전트 뼈대 — 데이터 조회 tool + 모델 예측 호출 tool 등록.

Gemini가 사용자 요청("10736번 신청자 심사해줘")을 보고 스스로
get_applicant_data / predict_risk 두 tool을 언제 호출할지 판단(function calling)한다.
"""

import asyncio

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types

from data_tools import get_applicant_data, predict_risk
from gemini_client import DECISION_MODEL

APP_NAME = "creditflow_agent"
USER_ID = "demo_user"

underwriting_agent = Agent(
    name="underwriting_data_agent",
    model=DECISION_MODEL,
    instruction=(
        "당신은 소상공인 대출 사전심사 에이전트입니다. "
        "신청자 ID가 주어지면 get_applicant_data로 신청자 정보를 조회하고, "
        "predict_risk로 부도확률과 정량 등급을 계산한 뒤, "
        "두 결과를 종합해 한국어로 간결하게 요약해 보고하세요."
    ),
    tools=[get_applicant_data, predict_risk],
)


async def run_query(user_text: str) -> str:
    session_service = InMemorySessionService()
    session = await session_service.create_session(app_name=APP_NAME, user_id=USER_ID)
    runner = Runner(agent=underwriting_agent, app_name=APP_NAME, session_service=session_service)

    final_text = ""
    async for event in runner.run_async(
        user_id=USER_ID,
        session_id=session.id,
        new_message=types.Content(role="user", parts=[types.Part(text=user_text)]),
    ):
        if event.content and event.content.parts:
            for part in event.content.parts:
                if part.function_call:
                    print(f"  [tool 호출] {part.function_call.name}({dict(part.function_call.args)})")
                if part.function_response:
                    print(f"  [tool 응답] {part.function_response.name} -> {part.function_response.response}")
                if part.text:
                    final_text += part.text

    return final_text


if __name__ == "__main__":
    result = asyncio.run(run_query("신청자 10736번을 심사해줘. 부도확률과 등급을 알려줘."))
    print("\n=== 최종 응답 ===")
    print(result)
