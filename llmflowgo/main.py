import sys
import os
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# Add the project root to the Python path
# This is necessary for the app to find modules like 'MEoh'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from my_meoh_app.database import create_db_and_tables
from my_meoh_app.api import problems, runs, workflows

APP_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(APP_DIR, "static")

def create_app() -> FastAPI:
    # Ensure the static directory exists before the app tries to mount it.
    if not os.path.exists(STATIC_DIR):
        os.makedirs(STATIC_DIR)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        print("Startup event: Creating database and tables...")
        create_db_and_tables()
        yield
        print("Shutdown event")

    app = FastAPI(lifespan=lifespan)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:3000",
            "http://127.0.0.1:3000",
            "http://localhost:5173",
            "http://127.0.0.1:5173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include API routers
    app.include_router(problems.router, prefix="/api", tags=["problems"])
    app.include_router(runs.router, prefix="/api", tags=["runs"])
    app.include_router(workflows.router, prefix="/api", tags=["workflows"])

    # Mount static files for the frontend, now that we've ensured the dir exists.
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

    @app.get("/")
    async def read_index():
        index_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Frontend not built yet. Run 'npm run build' in the frontend directory."}


    @app.get("/{full_path:path}")
    async def catch_all(full_path: str):
        index_path = os.path.join(STATIC_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return {"message": "Frontend not built yet. Run 'npm run build' in the frontend directory."}
    
    return app

# If you were to run this file directly (e.g., `python my_meoh_app/main.py`),
# you could add the following block for direct execution without uvicorn.
# if __name__ == "__main__":
#     import uvicorn
#     app = create_app()
#     uvicorn.run(app, host="127.0.0.1", port=8000)
