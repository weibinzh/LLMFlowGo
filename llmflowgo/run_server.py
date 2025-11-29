import uvicorn
import os
import sys
from multiprocessing import Process, set_start_method

from my_meoh_app.core.run_executor import executor_loop
from my_meoh_app.database import create_db_and_tables


def run_uvicorn_server():
    """Function to run the Uvicorn server."""
    print("Server will run on http://127.0.0.1:8001")

    uvicorn.run(
        "llmflowgo.main:create_app",
        factory=True,
        host="127.0.0.1",
        port=8001,
        reload=False,
        access_log=True,
        log_level="info"
    )

if __name__ == "__main__":
    set_start_method('spawn', force=True)

    project_root = os.path.abspath(os.path.dirname(__file__))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    print("--- Initializing database schema ---")
    create_db_and_tables()

    print("--- Starting Background Run Executor ---")
    executor_process = Process(target=executor_loop, daemon=False)
    executor_process.start()
    print("Executor process started successfully.")
    print("--------------------------------------")

    run_uvicorn_server()
