"""
CortiX Dashboard WebSockets

Handles live event streaming from Redis pub/sub to active React frontend dashboard sessions.
"""

import json
import logging
from typing import List
from fastapi import WebSocket, WebSocketDisconnect

logger = logging.getLogger("cortix.api.websocket")


class ConnectionManager:
    """
    Manages active websocket client connections.
    """

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info("New WebSocket client connected. Active connections: %d", len(self.active_connections))

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
            logger.info("WebSocket client disconnected. Active connections: %d", len(self.active_connections))

    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)

    async def broadcast(self, message: str):
        """Broadcast live JSON data to all listening React sessions."""
        bad_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                bad_connections.append(connection)
                
        # Clean up failed sessions
        for conn in bad_connections:
            self.disconnect(conn)


# Shared Connection Manager singleton
manager = ConnectionManager()
