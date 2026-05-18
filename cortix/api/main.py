"""
CortiX — Main FastAPI Application Entry Point

Combines routers, handles CORS headers, initializes database connections, 
and supports WebSocket pipelines for streaming live security updates.
"""

import os
import asyncio
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from cortix.config import config
from cortix.database import init_db
from cortix.api.routes import threats, metrics, containment, attackers
from cortix.api.websocket import manager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cortix.api.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup Phase
    logger.info("Initializing CortiX database session engines")
    init_db()
    yield
    # Shutdown Phase
    logger.info("Shutting down API server")


app = FastAPI(
    title="CortiX API Server",
    description="Neuro-inspired Adaptive Firewall Dashboard backend",
    version="1.0.0",
    lifespan=lifespan,
)

# Set up CORS policies
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(threats.router, prefix="/api")
app.include_router(metrics.router, prefix="/api")
app.include_router(containment.router, prefix="/api")
app.include_router(attackers.router, prefix="/api")


@app.get("/api/health")
def health_check():
    """System health check endpoint."""
    return {"status": "HEALTHY", "service": "CORTIX_API"}


@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket connection for real-time threat feed updates."""
    await manager.connect(websocket)
    try:
        # Keep connection open and listen to messages (if any client interaction needed)
        while True:
            data = await websocket.receive_text()
            # Echo or process manual admin updates
            await manager.send_personal_message(f"Acknowledged: {data}", websocket)
    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as exc:
        logger.debug("WebSocket pipeline exception: %s", exc)
        manager.disconnect(websocket)
