from __future__ import annotations

from typing import Any, Dict, List

from fastapi import WebSocket


class ConnectionManager:
    def __init__(self) -> None:
        # key = patient_id or "all"
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, patient_id: str = "all") -> None:
        await websocket.accept()
        self.active_connections.setdefault(patient_id, []).append(websocket)

    def disconnect(self, websocket: WebSocket, patient_id: str = "all") -> None:
        conns = self.active_connections.get(patient_id) or []
        if websocket in conns:
            conns.remove(websocket)
        if not conns and patient_id in self.active_connections:
            self.active_connections.pop(patient_id, None)

    async def send_alert(self, patient_id: str, alert_data: Dict[str, Any]) -> None:
        message = {"type": "CRITICAL_ALERT", "data": alert_data}

        targets: List[WebSocket] = []
        targets.extend(self.active_connections.get(patient_id, []))
        targets.extend(self.active_connections.get("all", []))

        for ws in list(dict.fromkeys(targets)):
            try:
                await ws.send_json(message)
            except Exception:
                # drop broken socket
                self._drop_socket(ws)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        for patient_id, sockets in list(self.active_connections.items()):
            for ws in list(sockets):
                try:
                    await ws.send_json(message)
                except Exception:
                    self.disconnect(ws, patient_id)

    def _drop_socket(self, websocket: WebSocket) -> None:
        for patient_id, sockets in list(self.active_connections.items()):
            if websocket in sockets:
                self.disconnect(websocket, patient_id)
