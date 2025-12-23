import json
import cv2
import numpy as np
import base64
from channels.generic.websocket import AsyncWebsocketConsumer
from scipy.spatial import distance as dist
from collections import OrderedDict
from ultralytics import YOLO # Import YOLO

class VideoConsumer(AsyncWebsocketConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 1. Initialize YOLOv8 Nano model (it will download the .pt file on first run)
        self.model = YOLO('yolov8n.pt')
        
        # Tracker settings
        self.next_id = 0
        self.objects = OrderedDict()
        self.entry_count = 0
        self.exit_count = 0
        
    async def receive(self, text_data):
        data = json.loads(text_data)
        image_data = data['image'].split(',')[1]
        nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        height, width = frame.shape[:2]
        line_x = width // 2  # Vertical crossing line

        # 2. Run YOLO Detection
        # classes=[0] ensures we ONLY detect "person" (COCO class 0)
        # conf=0.5 ignores weak detections
        results = self.model(frame, classes=[0], conf=0.5, verbose=False)
        
        detections = []
        input_centroids = []
        
        # 3. Process YOLO results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w, h = x2 - x1, y2 - y1
                
                # Calculate centroid
                cX = int(x1 + w / 2.0)
                cY = int(y1 + h / 2.0)
                
                input_centroids.append((cX, cY))
                detections.append({'x': int(x1), 'y': int(y1), 'w': int(w), 'h': int(h)})

        # 4. Tracking & Counting (Left/Right)
        if len(input_centroids) > 0:
            if len(self.objects) == 0:
                for i in range(len(input_centroids)):
                    self.register(input_centroids[i])
            else:
                object_ids = list(self.objects.keys())
                object_centroids = list(self.objects.values())
                
                D = dist.cdist(np.array(object_centroids), input_centroids)
                rows = D.min(axis=1).argsort()
                cols = D.argmin(axis=1)[rows]

                used_rows, used_cols = set(), set()
                for (row, col) in zip(rows, cols):
                    if row in used_rows or col in used_cols: continue
                    
                    obj_id = object_ids[row]
                    prev_x = self.objects[obj_id][0]
                    curr_x = input_centroids[col][0]
                    
                    # Logic: Left to Right = Entry | Right to Left = Exit
                    if prev_x < line_x and curr_x >= line_x:
                        self.entry_count += 1
                    elif prev_x > line_x and curr_x <= line_x:
                        self.exit_count += 1

                    self.objects[obj_id] = input_centroids[col]
                    used_rows.add(row)
                    used_cols.add(col)

        # 5. Send results to frontend
        await self.send(text_data=json.dumps({
            'entry': self.entry_count,
            'exit': self.exit_count,
            'current_people': len(detections),
            'boxes': detections
        }))

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.next_id += 1