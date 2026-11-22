import asyncio
import base64
import binascii
import json
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

import cv2
import numpy as np
from channels.generic.websocket import AsyncWebsocketConsumer
from django.conf import settings

from .tracking import CrossingCounter


MAX_MESSAGE_SIZE = 1_000_000
inference_worker = ThreadPoolExecutor(max_workers=1)


@lru_cache(maxsize=1)
def person_model():
    from ultralytics import YOLO

    return YOLO(str(settings.BASE_DIR / "yolov8n.pt"))


def detect_people(frame):
    result = person_model().predict(frame, classes=[0], conf=0.5, verbose=False)[0]
    coordinates = result.boxes.xyxy.int().tolist()
    return [{"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1} for x1, y1, x2, y2 in coordinates]


def decode_frame(message):
    if not message or len(message) > MAX_MESSAGE_SIZE:
        return None
    try:
        image = json.loads(message)["image"]
        if not isinstance(image, str) or not image.startswith("data:image/jpeg;base64,"):
            return None
        encoded = base64.b64decode(image.split(",", 1)[1], validate=True)
        return cv2.imdecode(np.frombuffer(encoded, np.uint8), cv2.IMREAD_COLOR) if encoded else None
    except (binascii.Error, KeyError, TypeError, json.JSONDecodeError, cv2.error):
        return None


class VideoConsumer(AsyncWebsocketConsumer):
    async def connect(self):
        self.counter = CrossingCounter()
        await self.accept()

    async def receive(self, text_data=None, bytes_data=None):
        frame = decode_frame(text_data)
        if frame is None:
            await self.send(text_data=json.dumps({"error": "invalid_frame"}))
            return

        height, width = frame.shape[:2]
        boxes = await asyncio.get_running_loop().run_in_executor(inference_worker, detect_people, frame)
        centers = [(box["x"] + box["w"] // 2, box["y"] + box["h"] // 2) for box in boxes]
        self.counter.update(centers, width, height)

        await self.send(text_data=json.dumps({
            "entry": self.counter.entries,
            "exit": self.counter.exits,
            "current_people": len(boxes),
            "boxes": boxes,
        }))
