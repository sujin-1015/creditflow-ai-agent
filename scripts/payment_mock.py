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
REPAY_LOG_PATH = BASE_DIR / "onchain" / "repayments_log.json"

DEVNET_TEST_AMOUNT_USDC = 1.00  # devnet 왕복 증빙용 고정 소액 (실제 대출액과 별개)
CONDITIONAL_AMOUNT_USDC = DEVNET_TEST_AMOUNT_USDC / 2  # 정책 3항: 조건부승인은 한도의 50%만 우선 집행
CURRENCY = "USDC"  # pay.sh/x402 기준 정산 통화에 맞춰 SOL 대신 devnet USDC 사용
_B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"

# --- 자금 통제(Controlled Funds) — 하드 캡 ---
# 정책 3항("소액대출 한도는 연매출의 5%, 최대 500만원")과 정책 6항(운영 안전장치)에 대응하는
# 값을 여기서도 독립적으로 다시 강제한다. decision.py가 계산한 승인 금액이 어떤 이유로든
# (버그, 프롬프트 조작 등) 한도를 넘겨 이 함수까지 들어오더라도, 집행 계층에서 한 번 더
# 막는 이중 방어선(defense-in-depth)이다 — 핫월렛에서 자금이 무제한 자동 송금되는 것을
# 막기 위한 최소한의 브레이크 지점.
PER_TX_HARD_CAP_KRW = 5_000_000  # 건별 한도 (정책 3항과 동일)
DAILY_HARD_CAP_KRW = 20_000_000  # 일별 누적 한도 (신규 — 운영 리스크 통제, 정책 6항)


class FundControlError(RuntimeError):
    """건별/일별 하드 캡 초과로 집행이 차단됐을 때 던진다."""


class DisbursementError(RuntimeError):
    """USDC 송금 단계에서 실패했을 때 던진다.

    지갑 발급은 송금 시도보다 먼저 끝나는 별개의(로컬) 단계라, 송금이 실패해도 지갑은 이미
    만들어져 있는 경우가 많다. 호출부가 "지갑은 발급됐지만 집행은 실패했다"를 구분해서
    보여줄 수 있도록 그 상태를 함께 담아 던진다.
    """

    def __init__(self, message: str, wallet_address: Optional[str] = None, wallet_newly_issued: bool = False):
        super().__init__(message)
        self.wallet_address = wallet_address
        self.wallet_newly_issued = wallet_newly_issued


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


@dataclass
class RepaymentResult:
    """지급의 역방향 — 신청자 지갑에서 agent_treasury로 되돌아오는 상환 트랜잭션 결과.

    devnet PoC에서는 실제 대출 원금/이자 스케줄을 계산하지 않고, 지급 때와 동일한
    devnet 왕복 증빙용 고정 소액(DEVNET_TEST_AMOUNT_USDC)을 상환하는 것으로 시연한다.
    """

    applicant_id: int
    amount_usdc: float
    currency: str
    status: str  # "EXECUTED"
    tx_signature: Optional[str]
    network: str
    is_mock: bool
    timestamp: str
    rationale: str
    rationale_hash: Optional[str] = None
    explorer_url: Optional[str] = None


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


def _real_collect_transfer(
    applicant_id: int, amount_usdc: float, memo: Optional[str] = None
) -> tuple[str, str]:
    """devnet_transfer.py를 통해 신청자 지갑 -> treasury로 실제 상환 트랜잭션을 실행한다."""
    record = devnet_transfer.send_devnet_usdc_repayment(applicant_id, amount_usdc, memo=memo)
    return record["tx_signature"], record["explorer_url"]


def _mock_collect_transfer(
    applicant_id: int, amount_usdc: float, memo: Optional[str] = None
) -> tuple[str, str]:
    time.sleep(0.05)
    sig = _fake_tx_signature()
    return sig, f"https://explorer.solana.com/tx/{sig}?cluster=devnet (MOCK — 실제 tx 아님)"


def _execute_collection(
    applicant_id: int, amount_usdc: float, memo: Optional[str] = None
) -> tuple[str, str]:
    collect_fn = _mock_collect_transfer if MOCK_MODE else _real_collect_transfer
    return collect_fn(applicant_id, amount_usdc, memo=memo)


def _check_hard_caps(applicant_id: int, requested_loan_krw: int) -> None:
    """건별/일별 하드 캡을 강제한다. 위반 시 FundControlError를 던져 집행을 차단한다."""
    if requested_loan_krw > PER_TX_HARD_CAP_KRW:
        raise FundControlError(
            f"건별 한도 초과 — 신청자 {applicant_id}: 요청 금액 {requested_loan_krw:,}원이 "
            f"건별 하드 캡 {PER_TX_HARD_CAP_KRW:,}원을 초과해 집행을 차단합니다."
        )

    today_total = bigquery_logger.get_today_disbursed_total_krw()
    if today_total + requested_loan_krw > DAILY_HARD_CAP_KRW:
        raise FundControlError(
            f"일별 누적 한도 초과 — 오늘 이미 {today_total:,}원 집행됨. 이번 건({requested_loan_krw:,}원)을 "
            f"더하면 일별 하드 캡 {DAILY_HARD_CAP_KRW:,}원을 초과해 집행을 차단합니다."
        )


# 대시보드 "하드 캡 테스트" 버튼용 데모 시나리오. _check_hard_caps()는 지갑 발급/devnet 송금 등
# 부작용이 전혀 없는 순수 검사라, 실제 심사 없이도 캡이 실제로 차단/통과하는지 즉시 보여줄 수 있다.
HARD_CAP_DEMO_SCENARIOS = {
    "violation": 6_000_000,  # 건별 하드 캡(500만원) 초과
    "clean": 100_000,  # 한도 이내 정상 범위
}
HARD_CAP_DEMO_APPLICANT_ID = 999999


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
        _check_hard_caps(applicant_id, requested_loan_krw)

        if wallet_address is None:
            wallet_name = f"applicant_{applicant_id}"
            wallet_newly_issued = not devnet_transfer.wallet_exists(wallet_name)
            wallet_address = devnet_transfer.get_or_create_devnet_wallet(wallet_name)

        amount = DEVNET_TEST_AMOUNT_USDC if decision == "approve" else CONDITIONAL_AMOUNT_USDC
        memo = _build_memo(applicant_id, decision, r_hash)
        try:
            tx_signature, explorer_url = _execute_transfer(wallet_address, amount, memo=memo)
        except Exception as e:  # noqa: BLE001
            raise DisbursementError(
                str(e), wallet_address=wallet_address, wallet_newly_issued=wallet_newly_issued
            ) from e
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


def collect_repayment(
    applicant_id: int,
    amount_usdc: Optional[float] = None,
    rationale: str = "",
) -> RepaymentResult:
    """상환을 처리한다 (신청자 지갑 -> agent_treasury) — 지급의 역방향 흐름.

    지급과 마찬가지로 온체인 메모에 근거 해시를 남겨, 상환이 실제로 어떤 근거(정책/신청자)에
    따라 이뤄졌는지 검증 가능하게 한다. amount_usdc를 지정하지 않으면 지급 때와 동일한
    devnet 왕복 증빙용 고정 소액(DEVNET_TEST_AMOUNT_USDC)을 상환하는 것으로 간주한다
    (실제 대출 원금/이자 스케줄 계산은 이 PoC 범위 밖).
    """
    amount_usdc = DEVNET_TEST_AMOUNT_USDC if amount_usdc is None else amount_usdc
    rationale = rationale or f"신청자 {applicant_id} devnet 상환 시뮬레이션"
    now = datetime.now(timezone.utc).isoformat()
    r_hash = _rationale_hash(rationale)
    memo = f"FundBridge|applicant={applicant_id}|type=repayment|sha256={r_hash}"

    tx_signature, explorer_url = _execute_collection(applicant_id, amount_usdc, memo=memo)

    result = RepaymentResult(
        applicant_id=applicant_id,
        amount_usdc=amount_usdc,
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
    _append_repayment_log(result)
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


def _append_repayment_log(result: RepaymentResult) -> None:
    REPAY_LOG_PATH.parent.mkdir(exist_ok=True)
    records = []
    if REPAY_LOG_PATH.exists():
        with open(REPAY_LOG_PATH, encoding="utf-8") as f:
            records = json.load(f)
    records.append(asdict(result))
    with open(REPAY_LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    try:
        bigquery_logger.log_repayment(asdict(result))
    except Exception as e:  # noqa: BLE001
        print(f"  [경고] BigQuery 상환 기록 실패 (로컬 로그는 정상 저장됨): {e}")
