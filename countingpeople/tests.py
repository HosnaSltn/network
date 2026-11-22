import base64
from unittest.mock import patch

import cv2
import numpy as np
from channels.testing import WebsocketCommunicator
from django.test import SimpleTestCase

from .consumers import VideoConsumer
from .tracking import CrossingCounter


class CrossingCounterTests(SimpleTestCase):
    def setUp(self):
        self.counter = CrossingCounter()

    def frame(self, *centers):
        self.counter.update(centers, 400, 300)

    def test_tracks_a_person_who_appears_after_the_first_frame(self):
        self.frame((100, 100))
        self.frame((100, 100), (245, 210))
        self.frame((105, 100), (170, 210))
        self.assertEqual((self.counter.entries, self.counter.exits), (0, 1))

    def test_counts_both_directions_without_counting_line_jitter(self):
        for x in [160, 195, 202, 198, 230, 200, 170]:
            self.frame((x, 100))
        self.assertEqual((self.counter.entries, self.counter.exits), (1, 1))

    def test_expires_missing_tracks_and_rejects_large_jumps(self):
        self.frame((175, 100))
        for _ in range(3):
            self.frame()
        self.frame((225, 100))
        self.frame((20, 100))
        self.assertEqual((self.counter.entries, self.counter.exits), (0, 0))


class VideoSocketTests(SimpleTestCase):
    async def test_bad_frame_does_not_interrupt_the_next_detection(self):
        socket = WebsocketCommunicator(VideoConsumer.as_asgi(), "/ws/video/")
        connected, _ = await socket.connect()
        self.assertTrue(connected)
        try:
            await socket.send_json_to({"image": "data:image/jpeg;base64,not-a-jpeg"})
            self.assertEqual(await socket.receive_json_from(), {"error": "invalid_frame"})

            _, jpeg = cv2.imencode(".jpg", np.zeros((300, 400, 3), dtype=np.uint8))
            frame = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
            box = {"x": 10, "y": 10, "w": 20, "h": 60}
            with patch("countingpeople.consumers.detect_people", return_value=[box]):
                await socket.send_json_to({"image": frame})
                self.assertEqual(await socket.receive_json_from(), {
                    "entry": 0, "exit": 0, "current_people": 1, "boxes": [box],
                })
        finally:
            await socket.disconnect()
