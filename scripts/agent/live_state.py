"""대시보드 실시간 "심사 중" 표시용 공유 상태 (Firestore).

Cloud Run은 여러 인스턴스로 스케일될 수 있어 프로세스 메모리에 상태를 두면 다른 인스턴스/다른
접속자에게 안 보인다. Firestore에 짧게 살다 사라지는 문서로 저장해 모든 대시보드 접속자가
같은 "심사 중" 상태를 폴링으로 볼 수 있게 한다.

이 상태는 순수 UX용 부가기능이라, 실패해도 심사/집행 흐름 자체를 막으면 안 된다 —
호출부(main.py)에서 항상 try/except로 감싸서 사용한다.

컨테이너가 OOM 등으로 비정상 종료되면 main.py의 clear_in_progress()가 끝내 실행되지
못해 문서가 영원히 안 지워질 수 있다 (실제로 겪음). list_in_progress()가 오래된 문서를
자동으로 걸러내고 정리해서, 정상 종료 경로가 실패해도 화면에 "심사 중"이 영구히 남지 않게 한다.
"""

from datetime import datetime, timedelta, timezone

from google.cloud import firestore

from bigquery_logger import PROJECT_ID

COLLECTION = "in_progress_decisions"
STALE_AFTER = timedelta(minutes=2)

_db = None


def get_db() -> firestore.Client:
    global _db
    if _db is None:
        _db = firestore.Client(project=PROJECT_ID)
    return _db


def mark_in_progress(applicant_id: int) -> None:
    get_db().collection(COLLECTION).document(str(applicant_id)).set(
        {"applicant_id": applicant_id, "started_at": firestore.SERVER_TIMESTAMP}
    )


def clear_in_progress(applicant_id: int) -> None:
    get_db().collection(COLLECTION).document(str(applicant_id)).delete()


def list_in_progress() -> list[dict]:
    now = datetime.now(timezone.utc)
    fresh = []
    for doc in get_db().collection(COLLECTION).stream():
        data = doc.to_dict()
        started_at = data.get("started_at")
        if started_at is not None and now - started_at > STALE_AFTER:
            # 정상 종료 경로(clear_in_progress)가 실행되지 못한 채 남은 문서로 간주하고 정리한다.
            try:
                doc.reference.delete()
            except Exception:  # noqa: BLE001
                pass
            continue
        fresh.append(data)
    return fresh
