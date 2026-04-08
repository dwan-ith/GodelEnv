"""
server/deps.py — Dependency injection singletons for FastAPI.
The environment and agent now live in the godel_engine package.
"""
from fastapi import WebSocket
from godel_engine.environment import GodelEnvironment


class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        dead = []
        for ws in self.active_connections:
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.active_connections.remove(ws)


# Module-level singletons — shared across all requests
_env = GodelEnvironment()
_ws_manager = ConnectionManager()


def get_env() -> GodelEnvironment:
    return _env


def get_ws_manager() -> ConnectionManager:
    return _ws_manager
