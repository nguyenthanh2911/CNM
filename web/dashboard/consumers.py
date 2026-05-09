from __future__ import annotations

import json
from typing import Any, Dict

from asgiref.sync import async_to_sync
from channels.generic.websocket import AsyncWebsocketConsumer
from channels.layers import get_channel_layer


class AlertConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.group_name = "alerts"
        await self.channel_layer.group_add(self.group_name, self.channel_name)
        await self.accept()

    async def disconnect(self, close_code):
        await self.channel_layer.group_discard(self.group_name, self.channel_name)

    async def receive(self, text_data=None, bytes_data=None):
        # Optional: forward incoming messages; currently keep-alive only.
        if text_data:
            try:
                payload = json.loads(text_data)
            except Exception:
                payload = {"type": "message", "data": text_data}
            await self.send(text_data=json.dumps(payload))

    async def alert_message(self, event):
        await self.send(text_data=json.dumps({"type": "alert_message", "data": event.get("data")}))

    @classmethod
    def notify(cls, alert_data: Dict[str, Any]) -> None:
        channel_layer = get_channel_layer()
        if channel_layer is None:
            return
        async_to_sync(channel_layer.group_send)("alerts", {"type": "alert_message", "data": alert_data})
