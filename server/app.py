"""
Production FastAPI server for Godel Env.
"""
from __future__ import annotations

import logging
import os

from dotenv import load_dotenv
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server.http_server import create_app

from godel_engine.openenv_environment import GodelOpenEnvEnvironment
from godel_engine.openenv_models import GodelOpenEnvAction, GodelOpenEnvObservation
from server.routers.api import router as demo_router


load_dotenv(override=False)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logging.getLogger("LiteLLM").setLevel(logging.WARNING)

app = create_app(
    env=GodelOpenEnvEnvironment,
    action_cls=GodelOpenEnvAction,
    observation_cls=GodelOpenEnvObservation,
    env_name="GodelEnv",
)
app.include_router(demo_router)
app.mount("/dashboard", StaticFiles(directory="dashboard", html=True), name="dashboard")


@app.get("/")
async def root():
    return RedirectResponse(url="/dashboard/")


def main() -> None:
    """Entry point for the Godel Env server."""
    import uvicorn

    port = int(os.getenv("PORT", 7860))
    uvicorn.run("server.app:app", host="0.0.0.0", port=port, reload=False)


if __name__ == "__main__":
    main()
