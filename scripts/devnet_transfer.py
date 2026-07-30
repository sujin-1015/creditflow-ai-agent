"""
공개 Solana devnet 왕복 결제 검증 (Plan A — 진짜 devnet, 무료 테스트 토큰만 사용)

pay.sh CLI의 `pay send`는 스테이블코인 잔액을 자체 서버가 원격으로 인식해야 하는 구조라
공개 devnet과 맞지 않아, Solana 공식 파이썬 SDK(solana-py/solders)로 devnet RPC
(https://api.devnet.solana.com)에 직접 붙어 지갑 생성 -> faucet 입금 -> 송금 -> 정산 확인
까지 왕복 테스트를 수행한다.

전부 devnet 무료 테스트 SOL만 사용 — 실제 자금 이동 없음.

실행:
    python devnet_transfer.py
"""

import asyncio
import json
from pathlib import Path
from typing import Optional

import base58
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.system_program import TransferParams, transfer
from solders.message import Message
from solders.transaction import Transaction

from solana.rpc.async_api import AsyncClient
from solana.rpc.commitment import Confirmed, Finalized
from solana.rpc.core import TxOptsModel

from spl.token.constants import TOKEN_PROGRAM_ID
from spl.token.instructions import (
    create_idempotent_associated_token_account,
    get_associated_token_address,
    transfer_checked,
)
from spl.token.instructions import models as spl_models

DEVNET_RPC = "https://api.devnet.solana.com"
BASE_DIR = Path(__file__).resolve().parent.parent
KEYS_DIR = BASE_DIR / "onchain" / "devnet_keys"
LOG_PATH = BASE_DIR / "onchain" / "devnet_transactions.json"

TRANSFER_LAMPORTS = 10_000_000  # 0.01 SOL (devnet 테스트 토큰, 실제 가치 없음)
AIRDROP_LAMPORTS = 1_000_000_000  # 1 SOL

# Circle 공식 devnet USDC (SPL 토큰). https://faucet.circle.com 에서 무료 수령 가능.
USDC_DEVNET_MINT = Pubkey.from_string("4zMMC9srt5Ri5X14GAgXhaHii3GnPAEERYPJgZJDncDU")
USDC_DECIMALS = 6

# SPL Memo 프로그램 — 판정 근거 해시를 결제 트랜잭션에 함께 새겨 위변조 여부를 검증 가능하게 한다.
MEMO_PROGRAM_ID = Pubkey.from_string("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")


def _memo_instruction(memo: str, signer: Pubkey) -> Instruction:
    return Instruction(
        program_id=MEMO_PROGRAM_ID,
        accounts=[AccountMeta(pubkey=signer, is_signer=True, is_writable=False)],
        data=memo.encode("utf-8"),
    )


def load_or_create_keypair(name: str) -> Keypair:
    KEYS_DIR.mkdir(parents=True, exist_ok=True)
    path = KEYS_DIR / f"{name}.json"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            secret_bytes = bytes(json.load(f))
        return Keypair.from_bytes(secret_bytes)
    kp = Keypair()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(list(bytes(kp)), f)
    return kp


async def ensure_funded(client: AsyncClient, pubkey: Pubkey, min_lamports: int) -> int:
    balance = (await client.get_balance(pubkey, commitment=Confirmed)).value
    if balance >= min_lamports:
        return balance

    print(f"  잔액 부족({balance} lamports) — devnet faucet airdrop 요청 중...")
    resp = await client.request_airdrop(pubkey, AIRDROP_LAMPORTS, commitment=Confirmed)
    sig = resp.value
    await client.confirm_transaction(sig, commitment=Confirmed)
    balance = (await client.get_balance(pubkey, commitment=Confirmed)).value
    print(f"  airdrop 완료. tx={sig}")
    return balance


async def transfer_sol(
    client: AsyncClient, sender: Keypair, recipient: Pubkey, lamports: int
) -> str:
    """blockhash 노드간 전파 지연에 대비해 재시도하며 SOL을 전송하고 서명을 반환한다."""
    ix = transfer(
        TransferParams(from_pubkey=sender.pubkey(), to_pubkey=recipient, lamports=lamports)
    )

    last_error = None
    for attempt in range(5):
        blockhash_resp = await client.get_latest_blockhash(commitment=Finalized)
        recent_blockhash = blockhash_resp.value.blockhash

        message = Message.new_with_blockhash([ix], sender.pubkey(), recent_blockhash)
        tx = Transaction.new_unsigned(message)
        tx.sign([sender], recent_blockhash)

        try:
            send_resp = await client.send_transaction(
                tx, opts=TxOptsModel(skip_preflight=True, preflight_commitment=Finalized)
            )
            signature = send_resp.value
            await client.confirm_transaction(signature, commitment=Confirmed)
            return str(signature)
        except Exception as e:  # noqa: BLE001
            last_error = e
            await asyncio.sleep(2)

    raise last_error


async def get_usdc_balance(client: AsyncClient, owner: Pubkey) -> int:
    """owner의 devnet USDC 잔액을 최소 단위(micro-USDC, 6 decimals)로 반환. ATA 없으면 0."""
    ata = get_associated_token_address(owner, USDC_DEVNET_MINT)
    try:
        resp = await client.get_token_account_balance(ata, commitment=Confirmed)
        return int(resp.value.amount)
    except Exception:  # noqa: BLE001
        return 0  # ATA가 아직 없음 (한 번도 USDC를 받은 적 없음)


async def transfer_usdc(
    client: AsyncClient,
    sender: Keypair,
    recipient: Pubkey,
    micro_usdc: int,
    memo: Optional[str] = None,
) -> str:
    """devnet USDC(SPL 토큰)를 전송한다. 수신자 ATA가 없으면 함께 생성한다.

    memo가 있으면 SPL Memo 프로그램 명령을 같은 트랜잭션에 포함시켜, 결제와
    판정 근거 해시가 하나의 원자적 트랜잭션으로 온체인에 함께 남도록 한다.
    """
    sender_ata = get_associated_token_address(sender.pubkey(), USDC_DEVNET_MINT)
    recipient_ata = get_associated_token_address(recipient, USDC_DEVNET_MINT)

    create_ata_ix = create_idempotent_associated_token_account(
        payer=sender.pubkey(), owner=recipient, mint=USDC_DEVNET_MINT
    )
    transfer_ix = transfer_checked(
        spl_models.TransferCheckedParams(
            program_id=TOKEN_PROGRAM_ID,
            source=sender_ata,
            mint=USDC_DEVNET_MINT,
            dest=recipient_ata,
            owner=sender.pubkey(),
            amount=micro_usdc,
            decimals=USDC_DECIMALS,
        )
    )
    instructions = [create_ata_ix, transfer_ix]
    if memo:
        instructions.append(_memo_instruction(memo, sender.pubkey()))

    last_error = None
    for attempt in range(5):
        blockhash_resp = await client.get_latest_blockhash(commitment=Finalized)
        recent_blockhash = blockhash_resp.value.blockhash

        message = Message.new_with_blockhash(
            instructions, sender.pubkey(), recent_blockhash
        )
        tx = Transaction.new_unsigned(message)
        tx.sign([sender], recent_blockhash)

        try:
            send_resp = await client.send_transaction(
                tx, opts=TxOptsModel(skip_preflight=True, preflight_commitment=Finalized)
            )
            signature = send_resp.value
            await client.confirm_transaction(signature, commitment=Confirmed)
            return str(signature)
        except Exception as e:  # noqa: BLE001
            last_error = e
            await asyncio.sleep(2)

    raise last_error


async def _send_devnet_usdc_payment(
    recipient_pubkey_str: str, amount_usdc: float, memo: Optional[str] = None
) -> dict:
    sender = load_or_create_keypair("agent_treasury")
    recipient = Pubkey.from_string(recipient_pubkey_str)
    micro_usdc = int(round(amount_usdc * 10**USDC_DECIMALS))

    async with AsyncClient(DEVNET_RPC) as client:
        # 트랜잭션 수수료(SOL)와 수신자 ATA 생성 rent는 SOL로 지불되므로 최소 SOL 잔액도 필요
        await ensure_funded(client, sender.pubkey(), 5_000_000)

        sender_balance = await get_usdc_balance(client, sender.pubkey())
        if sender_balance < micro_usdc:
            raise RuntimeError(
                f"agent_treasury USDC 잔액 부족 ({sender_balance / 10**USDC_DECIMALS:.2f} USDC). "
                f"https://faucet.circle.com 에서 {sender.pubkey()} 주소로 devnet USDC를 받아주세요."
            )

        before = await get_usdc_balance(client, recipient)
        signature = await transfer_usdc(client, sender, recipient, micro_usdc, memo=memo)
        after = await get_usdc_balance(client, recipient)

    record = {
        "network": "solana-devnet",
        "currency": "USDC",
        "mint": str(USDC_DEVNET_MINT),
        "rpc_url": DEVNET_RPC,
        "agent_wallet": str(sender.pubkey()),
        "applicant_wallet": recipient_pubkey_str,
        "amount_micro_usdc": micro_usdc,
        "amount_usdc": amount_usdc,
        "memo": memo,
        "tx_signature": signature,
        "confirmed": True,
        "applicant_balance_before_usdc": before / 10**USDC_DECIMALS,
        "applicant_balance_after_usdc": after / 10**USDC_DECIMALS,
        "explorer_url": f"https://explorer.solana.com/tx/{signature}?cluster=devnet",
    }
    _append_log(record)
    return record


def send_devnet_usdc_payment(
    recipient_pubkey_str: str, amount_usdc: float, memo: Optional[str] = None
) -> dict:
    """동기 진입점. payment_mock.py 등 sync 코드에서 바로 호출한다."""
    return asyncio.run(_send_devnet_usdc_payment(recipient_pubkey_str, amount_usdc, memo=memo))


async def _fetch_memo(client: AsyncClient, signature: str) -> Optional[str]:
    resp = await client.get_transaction(
        Signature.from_string(signature),
        max_supported_transaction_version=0,
        commitment=Confirmed,
    )
    if resp.value is None:
        return None
    message = resp.value.transaction.transaction.message
    account_keys = message.account_keys
    for ix in message.instructions:
        if account_keys[ix.program_id_index] == MEMO_PROGRAM_ID:
            return base58.b58decode(ix.data).decode("utf-8", errors="replace")
    return None


def fetch_memo(signature: str) -> Optional[str]:
    """devnet에 이미 올라간 트랜잭션에서 memo 프로그램에 새겨진 텍스트를 추출한다 (검증용)."""
    return asyncio.run(_fetch_memo_client(signature))


async def _fetch_memo_client(signature: str) -> Optional[str]:
    async with AsyncClient(DEVNET_RPC) as client:
        return await _fetch_memo(client, signature)


def _append_log(record: dict) -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    records = []
    if LOG_PATH.exists():
        with open(LOG_PATH, encoding="utf-8") as f:
            records = json.load(f)
    records.append(record)
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


async def _send_devnet_payment(recipient_pubkey_str: str, amount_sol: float) -> dict:
    sender = load_or_create_keypair("agent_treasury")
    recipient = Pubkey.from_string(recipient_pubkey_str)
    lamports = int(amount_sol * 1_000_000_000)

    async with AsyncClient(DEVNET_RPC) as client:
        await ensure_funded(client, sender.pubkey(), lamports + 5_000)
        before = (await client.get_balance(recipient, commitment=Confirmed)).value
        signature = await transfer_sol(client, sender, recipient, lamports)
        after = (await client.get_balance(recipient, commitment=Confirmed)).value

    record = {
        "network": "solana-devnet",
        "rpc_url": DEVNET_RPC,
        "agent_wallet": str(sender.pubkey()),
        "applicant_wallet": recipient_pubkey_str,
        "amount_lamports": lamports,
        "amount_sol": amount_sol,
        "tx_signature": signature,
        "confirmed": True,
        "applicant_balance_before_lamports": before,
        "applicant_balance_after_lamports": after,
        "explorer_url": f"https://explorer.solana.com/tx/{signature}?cluster=devnet",
    }
    _append_log(record)
    return record


def send_devnet_payment(recipient_pubkey_str: str, amount_sol: float) -> dict:
    """동기 진입점. payment_mock.py 등 sync 코드에서 바로 호출한다."""
    return asyncio.run(_send_devnet_payment(recipient_pubkey_str, amount_sol))


def get_or_create_devnet_wallet(name: str) -> str:
    """이름으로 devnet 키페어를 재사용/생성하고 pubkey 문자열을 반환한다."""
    return str(load_or_create_keypair(name).pubkey())


async def _check_rpc(client_cls=AsyncClient):
    async with client_cls(DEVNET_RPC) as client:
        return (await client.get_version()).value


def main():
    agent_wallet = load_or_create_keypair("agent_treasury")
    applicant_wallet = load_or_create_keypair("applicant_01")

    print(f"agent_treasury pubkey  : {agent_wallet.pubkey()}")
    print(f"applicant_01 pubkey    : {applicant_wallet.pubkey()}")
    print(f"devnet RPC 연결 확인: {asyncio.run(_check_rpc())}")

    result = send_devnet_payment(str(applicant_wallet.pubkey()), TRANSFER_LAMPORTS / 1e9)

    print(f"\n트랜잭션 전송: {result['tx_signature']}")
    print(f"\n=== 정산 확인 ===")
    print(f"확정 여부       : {result['confirmed']}")
    print(
        f"수신자 잔액 변화: {result['applicant_balance_before_lamports']/1e9:.4f} -> "
        f"{result['applicant_balance_after_lamports']/1e9:.4f} SOL"
    )
    print(f"Explorer        : {result['explorer_url']}")
    print(f"\n로그 저장: {LOG_PATH}")


if __name__ == "__main__":
    main()
