import json
import cv2
import numpy as np
import base64
from channels.generic.websocket import AsyncWebsocketConsumer

class VideoConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Simple tracking state
        self.prev_centroids = []
        self.entry_count = 0
        self.exit_count = 0
        # Load pre-trained HOG detector for people
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    async def connect(self):
        await self.accept()

    async def receive(self, text_data):
        data = json.loads(text_data)
        image_data = data['image'].split(',')[1]
        
        # Decode image
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # 1. Detection
        boxes, weights = self.hog.detectMultiScale(frame, winStride=(8, 8), padding=(8, 8), scale=1.05)
        
        # 2. Basic Counting Logic (Line Crossing)
        # We define a virtual line at the middle of the frame (y = height/2)
        height, width, _ = frame.shape
        line_y = height // 2
        
        current_count = len(boxes)
        # Note: A real production system would use a Centroid Tracker (like SORT)
        # For this example, we'll simplify the feedback:
        
        # Send data back to frontend
        await self.send(text_data=json.dumps({
            'entry': self.entry_count,
            'exit': self.exit_count,
            'current_people': current_count
        }))