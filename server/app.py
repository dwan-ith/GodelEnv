"""
Production FastAPI Server for Gödel Env.
"""
from __future__ import annotations
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Depends
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv

load_dotenv(override=False)

from server.routers.api import router as api_router
from server.deps import get_ws_manager, ConnectionManager

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)

# Suppress overly verbose litellm logs if needed
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

app = FastAPI(title="Gödel Env Server")

from fastapi.responses import RedirectResponse

# Mount API
app.include_router(api_router)

@app.get("/")
async def root():
    return RedirectResponse(url="/dashboard/")

# Mount Dashboard frontend
app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")

@app.websocket("/ws/stream")
async def websocket_endpoint(
    websocket: WebSocket,
    manager: ConnectionManager = Depends(get_ws_manager)
):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket)

def main():
    """Entry point for the Godel Env server (used by [project.scripts])."""
    import uvicorn
    import os
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)

if __name__ == "__main__":
    main()
