# py/ws_manager.py
import json
import logging
from typing import List, Optional
from fastapi import WebSocket

logger = logging.getLogger("app")

MAX_WS_MESSAGE_SIZE = 5 * 1024 * 1024  # 5MB limit to prevent renderer crash

class ConnectionManager:
    def __init__(self):
        # 维护所有活跃的 WebSocket 连接
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def send_json(self, message: dict, websocket: WebSocket):
        """向特定连接发送消息"""
        try:
            json_str = json.dumps(message, ensure_ascii=False)
            if len(json_str) > MAX_WS_MESSAGE_SIZE:
                logger.warning(f"WebSocket message too large ({len(json_str)} bytes), skipping send")
                return
            await websocket.send_json(message)
        except Exception as e:
            logger.error(f"发送消息失败: {e}")
            self.disconnect(websocket)

    async def broadcast(self, message: dict, exclude: Optional[WebSocket] = None):
        """向所有连接广播消息，可选排除某个连接"""
        try:
            json_str = json.dumps(message, ensure_ascii=False)
            if len(json_str) > MAX_WS_MESSAGE_SIZE:
                logger.warning(f"Broadcast message too large ({len(json_str)} bytes), skipping")
                return
        except Exception:
            return
        # 使用切片副本遍历，防止在循环中删除元素导致报错
        for connection in self.active_connections[:]:
            if connection == exclude:
                continue
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"广播失败，移除失效连接: {e}")
                self.disconnect(connection)

    async def broadcast_settings_update(self, settings: dict, exclude: Optional[WebSocket] = None):
        """快捷函数：广播配置更新"""
        await self.broadcast({
            "type": "settings_update",
            "data": settings
        }, exclude=exclude)

# 创建单例对象
ws_manager = ConnectionManager()