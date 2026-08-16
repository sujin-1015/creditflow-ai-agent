"""
온체인 결제 실행 레이어 — 실제 Solana devnet 집행 (+ 목업 폴백)

원래 계획(pay.sh CLI, solana-foundation/pay)은 이 PC에 Windows Hello가 설정되어
있지 않아 계정 생성 단계에서 막혔다 (OS 레벨 생체인증 미설정 — 코드로 우회 불가).
대신 Solana 공식 파이썬 SDK(solana-py/solders)로 공개 devnet(api.devnet.solana.com)에
직접 붙어 실제 트랜잭션을 실행한다 (devnet_transfer.py). MOCK_MODE=False가 기본값이며,
devnet RPC 장애 등으로 실거래가 안 될 때만 MOCK_MODE=True로 바꿔 로컬 시뮬레이션으로
대체한다 (그 경로는 실제 네트워크 호출이 전혀 없다 — 비용/실거래 없음).
"""

import hashlib
import json
import secrets
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import devnet_transfer

sys.path.insert(0, str(Path(__file__).resolve().parent / "agent"))
import bigquery_logger  # noqa: E402

MOCK_MODE = False  # True로 바꾸면 실제 devnet 호출 없이 로컬 시뮬레이션만 수행

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_PATH = BASE_DIR / "onchain" / "payments_log.json"

DEVNET_TEST_AMOUNT_USDC = 1.00  # devnet 왕복 증빙용 고정 소액 (실제 대출액과 별개)
CONDITIONAL_AMOUNT_USDC = DEVNET_TEST_AMOUNT_USDC / 2  # 정책 3항: 조건부승인은 한도의 50%만 우선 집행
CURRENCY = "USDC"  # pay.sh/x402 기준 정산 통화에 맞춰 SOL 대신 devnet USDC 사용
_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"


@dataclass
class PaymentResult:
    applicant_id: int
    decision: str  # "approve" | "conditional" | "reject"
    wallet_address: Optional[str]
    requested_loan_krw: int
    devnet_test_amount: float
    currency: str
    status: str  # "EXECUTED" | "SKIPPED"
    tx_signature: Optional[str]
    network: str
    is_mock: bool
    timestamp: str
    rationale: str
    rationale_hash: Optional[str] = None
    explorer_url: Optional[str] = None
    critic_verdict: Optional[str] = None
    critic_reasoning: Optional[str] = None
    tool_call_summary: Optional[str] = None
    wallet_newly_issued: bool = False


def _fake_tx_signature() -> str:
    return "".join(secrets.choice(_B58_ALPHABET) for _ in range(88))


def _rationale_hash(rationale: str) -> str:
    return hashlib.sha256(rationale.encode("utf-8")).hexdigest()


def _build_memo(applicant_id: int, decision: str, rationale_hash: str) -> str:
    """결제 트랜잭션에 함께 새길 온체인 메모. 판정 근거 해시를 포함해 위변조를 검증 가능하게 한다."""
    return f"FundBridge|applicant={applicant_id}|decision={decision}|sha256={rationale_hash}"


def _real_pay_transfer(
    wallet_address: str, amount_usdc: float, memo: Optional[str] = None
) -> tuple[str, str]:
    """devnet_transfer.py를 통해 공개 Solana devnet에 실제 USDC(SPL 토큰) 트랜잭션을 실행한다."""
    record = devnet_transfer.send_devnet_usdc_payment(wallet_address, amount_usdc, memo=memo)
    return record["tx_signature"], record["explorer_url"]


def _mock_pay_transfer(
    wallet_address: str, amount_usdc: float, memo: Optional[str] = None
) -> tuple[str, str]:
    """실제 네트워크 호출 없이 결제 왕복(요청->승인->정산)을 흉내낸다."""
    time.sleep(0.05)  # 정산 지연 시뮬레이션
    sig = _fake_tx_signature()
    return sig, f"https://explorer.solana.com/tx/{sig}?cluster=devnet (MOCK — 실제 tx 아님)"


def _execute_transfer(
    wallet_address: str, amount_usdc: float, memo: Optional[str] = None
) -> tuple[str, str]:
    transfer_fn = _mock_pay_transfer if MOCK_MODE else _real_pay_transfer
    return transfer_fn(wallet_address, amount_usdc, memo=memo)


def disburse_loan(
    applicant_id: int,
    decision: str,
    requested_loan_krw: int,
    rationale: str,
    wallet_address: Optional[str] = None,
    critic_verdict: Optional[str] = None,
    critic_reasoning: Optional[str] = None,
    tool_call_summary: Optional[str] = None,
) -> PaymentResult:
    """판정 결과에 따라 온체인 집행을 수행한다 (최초 판정 시점 집행분).

    승인 -> 지갑 보유 여부 확인 -> 없으면 임베디드(커스터디) 지갑을 즉시 발급 -> 전액
        (devnet 테스트 기준 1.00 USDC) 집행.
    조건부승인 -> 위와 동일한 지갑 확인/발급 후, 정책 3항에 따라 50%(0.50 USDC)만 우선
        집행 (잔여분은 재심사 후 결정).
    거절 -> 지갑을 조회/발급하지 않고, 트랜잭션 없이 판정 근거만 기록.
    """
    now = datetime.now(timezone.utc).isoformat()
    r_hash = _rationale_hash(rationale)
    wallet_newly_issued = False

    if decision in ("approve", "conditional"):
        if wallet_address is None:
            wallet_name = f"applicant_{applicant_id}"
            wallet_newly_issued = not devnet_transfer.wallet_exists(wallet_name)
            wallet_address = devnet_transfer.get_or_create_devnet_wallet(wallet_name)

        amount = DEVNET_TEST_AMOUNT_USDC if decision == "approve" else CONDITIONAL_AMOUNT_USDC
        memo = _build_memo(applicant_id, decision, r_hash)
        tx_signature, explorer_url = _execute_transfer(wallet_address, amount, memo=memo)
        result = PaymentResult(
            applicant_id=applicant_id,
            decision=decision,
            wallet_address=wallet_address,
            requested_loan_krw=requested_loan_krw,
            devnet_test_amount=amount,
            currency=CURRENCY,
            status="EXECUTED",
            tx_signature=tx_signature,
            network="solana-devnet-mock" if MOCK_MODE else "solana-devnet",
            is_mock=MOCK_MODE,
            timestamp=now,
            rationale=rationale,
            rationale_hash=r_hash,
            explorer_url=explorer_url,
            critic_verdict=critic_verdict,
            critic_reasoning=critic_reasoning,
            tool_call_summary=tool_call_summary,
            wallet_newly_issued=wallet_newly_issued,
        )
    else:
        result = PaymentResult(
            applicant_id=applicant_id,
            decision=decision,
            wallet_address=wallet_address,
            requested_loan_krw=requested_loan_krw,
            devnet_test_amount=0.0,
            currency=CURRENCY,
            status="SKIPPED",
            tx_signature=None,
            network="solana-devnet-mock" if MOCK_MODE else "solana-devnet",
            is_mock=MOCK_MODE,
            timestamp=now,
            rationale=rationale,
            rationale_hash=r_hash,
            explorer_url=None,
            critic_verdict=critic_verdict,
            critic_reasoning=critic_reasoning,
            tool_call_summary=tool_call_summary,
        )

    _append_log(result)
    return result


def disburse_remaining(
    applicant_id: int,
    wallet_address: str,
    rationale: str,
) -> PaymentResult:
    """재심사에서 조건부->승인으로 상향된 건의 잔여 50%(0.50 USDC)를 집행한다."""
    now = datetime.now(timezone.utc).isoformat()
    r_hash = _rationale_hash(rationale)
    memo = _build_memo(applicant_id, "approve", r_hash)
    tx_signature, explorer_url = _execute_transfer(
        wallet_address, CONDITIONAL_AMOUNT_USDC, memo=memo
    )
    result = PaymentResult(
        applicant_id=applicant_id,
        decision="approve",
        wallet_address=wallet_address,
        requested_loan_krw=0,  # 잔여분 집행이라 최초 한도 필드는 별도 기록 안 함 (reevaluations 테이블 참고)
        devnet_test_amount=CONDITIONAL_AMOUNT_USDC,
        currency=CURRENCY,
        status="EXECUTED",
        tx_signature=tx_signature,
        network="solana-devnet-mock" if MOCK_MODE else "solana-devnet",
        is_mock=MOCK_MODE,
        timestamp=now,
        rationale=rationale,
        rationale_hash=r_hash,
        explorer_url=explorer_url,
    )
    _append_log(result)
    return result


def _append_log(result: PaymentResult) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    records = []
    if LOG_PATH.exists():
        with open(LOG_PATH, encoding="utf-8") as f:
            records = json.load(f)
    records.append(asdict(result))
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    try:
        bigquery_logger.log_decision(asdict(result))
    except Exception as e:  # noqa: BLE001
        # 로컬 JSON은 이미 저장 완료 — BigQuery는 감사/조회용 보조 기록이라 실패해도 흐름을 막지 않는다.
        print(f"  [경고] BigQuery 기록 실패 (로컬 로그는 정상 저장됨): {e}")
