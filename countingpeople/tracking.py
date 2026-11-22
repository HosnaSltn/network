from dataclasses import dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist


@dataclass
class Track:
    center: tuple[int, int]
    side: int
    missed_frames: int = 0


class CrossingCounter:
    def __init__(self):
        self.entries = 0
        self.exits = 0
        self.frame_size = None
        self.tracks = []

    def update(self, centers, width, height):
        if self.frame_size != (width, height):
            self.frame_size = (width, height)
            self.tracks.clear()

        line_x = width // 2
        margin = max(1, round(width * 0.025))

        def side_of_line(center):
            x, _ = center
            return int(x > line_x + margin) - int(x < line_x - margin)

        matched_centers = set()
        for track in self.tracks:
            track.missed_frames += 1

        if self.tracks and centers:
            distances = cdist([track.center for track in self.tracks], centers)
            max_distance = max(width, height) * 0.25
            # Penalize invalid pairs enough to prefer any extra valid match.
            penalty = max_distance * (1 + min(distances.shape))
            cost = np.where(distances <= max_distance, distances, penalty)
            rows, cols = linear_sum_assignment(cost)

            for row, col in zip(rows, cols):
                if distances[row, col] > max_distance:
                    continue
                track = self.tracks[row]
                track.center = centers[col]
                track.missed_frames = 0
                side = side_of_line(track.center)
                if side:
                    if track.side == -1 and side == 1:
                        self.entries += 1
                    elif track.side == 1 and side == -1:
                        self.exits += 1
                    track.side = side
                matched_centers.add(col)

        self.tracks = [track for track in self.tracks if track.missed_frames < 3]
        self.tracks.extend(
            Track(center, side_of_line(center))
            for index, center in enumerate(centers)
            if index not in matched_centers
        )
