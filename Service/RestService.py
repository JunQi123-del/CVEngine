from dataclasses import dataclass

import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel


@dataclass
class RestServiceConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "info"   # "debug", "info", "warning", "error"
    reload: bool = False


class ATDRequest(BaseModel):
    VideoURL: str
    cerebroID: str
    payloadId: str


class RestService:
    """
    FastAPI + Uvicorn REST service.

    Usage:
        config = RestServiceConfig(host="0.0.0.0", port=8000)
        service = RestService(config)
        service.run()

    Endpoints:
        POST /ATD — receive VideoURL, cerebroID, payloadId
    """

    def __init__(self, config: RestServiceConfig):
        self.cfg = config
        self.app = FastAPI(title="CVEngine REST API", version="1.0.0")
        self._register_routes()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the Uvicorn server (blocking)."""
        uvicorn.run(
            self.app,
            host=self.cfg.host,
            port=self.cfg.port,
            log_level=self.cfg.log_level,
            reload=self.cfg.reload,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _register_routes(self) -> None:
        app = self.app

        @app.post("/ATD")
        async def atd(request: ATDRequest):
            """Receive a video task: VideoURL, cerebroID, payloadId."""
             return {
                "VideoURL": request.VideoURL,
                "cerebroID": request.cerebroID,
                "payloadId": request.payloadId,
            }
