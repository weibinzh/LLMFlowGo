import time
import uuid
import logging
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Session, select

from my_meoh_app.database import engine
from my_meoh_app.models.run import Run
from my_meoh_app.core.meoh_runner import run_meoh_optimization
from sqlalchemy.exc import OperationalError

# Configure logging to reduce noise
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)


def _claim_next_pending_run(session: Session) -> Optional[Run]:
    """Fetch and claim the next pending Run to avoid duplicate processing."""
    for attempt in range(3):
        result = session.exec(select(Run).where(Run.status == "pending"))
        run = result.first()
        if not run:
            return None
        # Mark as starting to prevent other executors from picking it
        run.status = "starting"
        if not run.start_time:
            run.start_time = datetime.now(timezone.utc)
        session.add(run)
        try:
            session.commit()
        except Exception as e:
            msg = str(e).lower()
            if "locked" in msg or "busy" in msg or isinstance(e, OperationalError):
                print(f"[executor] DB locked/busy while claiming run {getattr(run,'id',None)}; retry {attempt+1}/3")
                try:
                    session.rollback()
                except Exception:
                    pass
                time.sleep(0.5 * (attempt + 1))
                continue
            print(f"[executor] Error committing claim for run {getattr(run,'id',None)}: {e}")
            try:
                session.rollback()
            except Exception:
                pass
            return None
        session.refresh(run)
        print(f"[executor] Claimed run {run.id} and set status=starting")
        return run
    print("[executor] Failed to claim run due to DB lock after retries.")
    return None


def _mark_failed(session: Session, run_id: uuid.UUID, message: str) -> None:
    run = session.get(Run, run_id)
    if not run:
        return
    run.status = "failed"
    run.end_time = datetime.now(timezone.utc)
    logs = run.logs or ""
    if logs:
        logs += "\n"
    run.logs = logs + f"[executor] {message}"
    session.add(run)
    session.commit()


def executor_loop(poll_interval_seconds: float = 2.0) -> None:
    print(f"[executor] Started. Polling every {poll_interval_seconds}s for pending runs...")
    while True:
        try:
            with Session(engine) as session:
                run = _claim_next_pending_run(session)
                if not run:
                    # Nothing to do; sleep and continue
                    time.sleep(poll_interval_seconds)
                    continue

                print(f"[executor] Picked run {run.id} (problem {run.problem_id})")
                try:
                    run_meoh_optimization(problem_id_str=str(run.problem_id), run_id_str=str(run.id))
                except Exception as e:
                    print(f"[executor] Unhandled error during run {run.id}: {e}")
                    _mark_failed(session, run.id, f"Unhandled exception in executor: {e}")
        except KeyboardInterrupt:
            print("[executor] Stopping on keyboard interrupt.")
            break
        except Exception as loop_err:
            # Unexpected loop error; wait a bit to avoid tight crash loops
            print(f"[executor] Loop error: {loop_err}. Sleeping before retry...")
            time.sleep(5)


if __name__ == "__main__":
    executor_loop()


