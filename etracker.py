import numpy as np
from scipy.optimize import linear_sum_assignment
from collections import deque
import cv2


class EnhancedBagTracker:
    def __init__(self,
                 iou_threshold=0.3,  # Lower for bags that might deform
                 max_missing=15,  # Higher for warehouse scenarios
                 min_hits=3,  # Minimum detections before confirmed
                 max_age=30,  # Maximum frames to keep lost tracks
                 motion_threshold=50):  # Max pixel movement between frames

        self.next_id = 0
        self.tracks = {}  # id -> Track object
        self.iou_threshold = iou_threshold
        self.max_missing = max_missing
        self.min_hits = min_hits
        self.max_age = max_age
        self.motion_threshold = motion_threshold

    def update(self, detections, frame_number=None):
        """
        Update tracker with new detections
        detections: list of (x, y, w, h, confidence) tuples
        """
        if frame_number is None:
            frame_number = getattr(self, 'frame_count', 0)
            self.frame_count = frame_number + 1

        # Predict next positions for existing tracks
        self._predict_tracks()

        # Match detections to tracks
        matches, unmatched_dets, unmatched_trks = self._associate_detections_to_tracks(
            detections, list(self.tracks.keys())
        )

        # Update matched tracks
        for track_id, det_idx in matches:
            self.tracks[track_id].update(detections[det_idx], frame_number)

        # Handle unmatched tracks (missing detections)
        for track_id in unmatched_trks:
            self.tracks[track_id].mark_missed(frame_number)

        # Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            self._create_new_track(detections[det_idx], frame_number)

        # Remove dead tracks
        self._remove_dead_tracks()

        # Return confirmed tracks only
        return self.get_confirmed_tracks()

    def _predict_tracks(self):
        """Predict next position using Kalman filter or simple motion model"""
        for track in self.tracks.values():
            track.predict()

    def _associate_detections_to_tracks(self, detections, track_ids):
        """Hungarian algorithm for optimal assignment"""
        if len(track_ids) == 0:
            return [], list(range(len(detections))), []
        if len(detections) == 0:
            return [], [], track_ids

        # Create cost matrix
        cost_matrix = np.zeros((len(track_ids), len(detections)), dtype=np.float32)

        for t, track_id in enumerate(track_ids):
            track = self.tracks[track_id]
            for d, det in enumerate(detections):
                # Combine IoU and motion consistency
                iou = self.compute_iou(track.get_predicted_bbox(), det[:4])
                motion_cost = self._compute_motion_cost(track, det[:4])

                # Higher weight for IoU, lower for motion (adjust as needed)
                cost = (1 - iou) + 0.3 * motion_cost
                cost_matrix[t, d] = cost

        # Apply Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)

        matches = []
        unmatched_detections = []
        unmatched_tracks = []

        # Check which assignments are valid
        for row, col in zip(row_indices, col_indices):
            track_id = track_ids[row]
            track = self.tracks[track_id]
            det = detections[col]

            iou = self.compute_iou(track.get_predicted_bbox(), det[:4])
            motion_cost = self._compute_motion_cost(track, det[:4])

            # Accept match if IoU is good OR motion is consistent
            if (iou > self.iou_threshold or
                    (iou > 0.1 and motion_cost < 0.5)):  # Relaxed matching
                matches.append((track_id, col))
            else:
                unmatched_tracks.append(track_id)
                unmatched_detections.append(col)

        # Add unmatched tracks and detections
        for t, track_id in enumerate(track_ids):
            if t not in row_indices:
                unmatched_tracks.append(track_id)

        for d in range(len(detections)):
            if d not in col_indices:
                unmatched_detections.append(d)

        return matches, unmatched_detections, unmatched_tracks

    def _compute_motion_cost(self, track, detection_bbox):
        """Compute motion consistency cost"""
        if len(track.history) < 2:
            return 0.0

        # Predict next position based on velocity
        last_center = track.get_center()
        predicted_center = track.get_predicted_center()
        detection_center = self._get_bbox_center(detection_bbox)

        # Distance between predicted and actual
        dist = np.linalg.norm(np.array(predicted_center) - np.array(detection_center))

        # Normalize by motion threshold
        return min(1.0, dist / self.motion_threshold)

    def _create_new_track(self, detection, frame_number):
        """Create new track for unmatched detection"""
        track = Track(self.next_id, detection, frame_number,
                      max_missing=self.max_missing, min_hits=self.min_hits)
        self.tracks[self.next_id] = track
        self.next_id += 1

    def _remove_dead_tracks(self):
        """Remove tracks that have been missing too long"""
        dead_tracks = []
        for track_id, track in self.tracks.items():
            if track.is_dead():
                dead_tracks.append(track_id)

        for track_id in dead_tracks:
            del self.tracks[track_id]

    def get_confirmed_tracks(self):
        """Return only confirmed tracks (minimum hits reached)"""
        confirmed = {}
        for track_id, track in self.tracks.items():
            if track.is_confirmed():
                confirmed[track_id] = track.get_current_bbox()
        return confirmed

    def get_all_tracks_with_status(self):
        """Return all tracks with their status for debugging"""
        result = {}
        for track_id, track in self.tracks.items():
            result[track_id] = {
                'bbox': track.get_current_bbox(),
                'confirmed': track.is_confirmed(),
                'missing_count': track.missing_count,
                'hit_count': track.hit_count,
                'age': track.age
            }
        return result

    @staticmethod
    def compute_iou(boxA, boxB):
        """Compute IoU between two bounding boxes"""
        xA1, yA1, wA, hA = boxA
        xB1, yB1, wB, hB = boxB
        xA2, yA2 = xA1 + wA, yA1 + hA
        xB2, yB2 = xB1 + wB, yB1 + hB

        inter_x1 = max(xA1, xB1)
        inter_y1 = max(yA1, yB1)
        inter_x2 = min(xA2, xB2)
        inter_y2 = min(yA2, yB2)

        inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
        areaA = wA * hA
        areaB = wB * hB

        union_area = areaA + areaB - inter_area
        if union_area == 0:
            return 0

        return inter_area / union_area

    @staticmethod
    def _get_bbox_center(bbox):
        """Get center point of bounding box"""
        x, y, w, h = bbox
        return (x + w / 2, y + h / 2)


class Track:
    """Individual track for a bag"""

    def __init__(self, track_id, initial_detection, frame_number, max_missing=15, min_hits=3):
        self.id = track_id
        self.max_missing = max_missing
        self.min_hits = min_hits

        # State variables
        self.missing_count = 0
        self.hit_count = 1
        self.age = 1
        self.last_frame = frame_number

        # History for motion prediction
        self.history = deque(maxlen=10)  # Keep last 10 positions
        self.history.append(initial_detection[:4])  # Store bbox

        # Simple Kalman-like state (position + velocity)
        x, y, w, h = initial_detection[:4]
        self.state = np.array([x + w / 2, y + h / 2, 0, 0, w, h], dtype=np.float32)  # cx, cy, vx, vy, w, h

    def update(self, detection, frame_number):
        """Update track with new detection"""
        self.missing_count = 0
        self.hit_count += 1
        self.age += 1
        self.last_frame = frame_number

        # Update state
        x, y, w, h = detection[:4]
        new_center = [x + w / 2, y + h / 2]

        if len(self.history) > 0:
            old_center = self._get_bbox_center(self.history[-1])
            velocity = [new_center[0] - old_center[0], new_center[1] - old_center[1]]
        else:
            velocity = [0, 0]

        self.state = np.array([new_center[0], new_center[1], velocity[0], velocity[1], w, h])
        self.history.append(detection[:4])

    def mark_missed(self, frame_number):
        """Mark track as missed in current frame"""
        self.missing_count += 1
        self.age += 1
        self.last_frame = frame_number

    def predict(self):
        """Predict next position based on velocity"""
        if len(self.history) >= 2:
            # Update position based on velocity
            self.state[0] += self.state[2]  # x += vx
            self.state[1] += self.state[3]  # y += vy

            # Decay velocity slightly
            self.state[2] *= 0.9
            self.state[3] *= 0.9

    def get_predicted_bbox(self):
        """Get predicted bounding box"""
        cx, cy, vx, vy, w, h = self.state
        return [cx - w / 2, cy - h / 2, w, h]

    def get_current_bbox(self):
        """Get current bounding box"""
        if self.history:
            return self.history[-1]
        return self.get_predicted_bbox()

    def get_center(self):
        """Get current center"""
        bbox = self.get_current_bbox()
        return self._get_bbox_center(bbox)

    def get_predicted_center(self):
        """Get predicted center"""
        return (self.state[0], self.state[1])

    def is_confirmed(self):
        """Check if track is confirmed (enough hits)"""
        return self.hit_count >= self.min_hits

    def is_dead(self):
        """Check if track should be deleted"""
        return self.missing_count > self.max_missing

    @staticmethod
    def _get_bbox_center(bbox):
        """Get center of bounding box"""
        x, y, w, h = bbox
        return (x + w / 2, y + h / 2)


class LineCounter:
    """Count objects crossing predefined lines"""

    def __init__(self, lines):
        """
        lines: list of ((x1, y1), (x2, y2)) representing counting lines
        """
        self.lines = lines
        self.line_counts = [0] * len(lines)
        self.crossed_tracks = [set() for _ in lines]  # Track which objects crossed each line

    def update_counts(self, tracks_history):
        """
        Update counts based on track trajectories
        tracks_history: dict of {track_id: [list of center points]}
        """
        for track_id, trajectory in tracks_history.items():
            if len(trajectory) < 2:
                continue

            # Check each line
            for line_idx, line in enumerate(self.lines):
                if track_id not in self.crossed_tracks[line_idx]:
                    # Check if trajectory crosses the line
                    if self._trajectory_crosses_line(trajectory, line):
                        self.crossed_tracks[line_idx].add(track_id)
                        self.line_counts[line_idx] += 1

    def _trajectory_crosses_line(self, trajectory, line):
        """Check if trajectory crosses a line"""
        (x1, y1), (x2, y2) = line

        for i in range(len(trajectory) - 1):
            p1 = trajectory[i]
            p2 = trajectory[i + 1]

            if self._segments_intersect(p1, p2, (x1, y1), (x2, y2)):
                return True
        return False

    @staticmethod
    def _segments_intersect(p1, p2, p3, p4):
        """Check if two line segments intersect"""

        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    def get_counts(self):
        """Get current counts for each line"""
        return self.line_counts.copy()


# Usage example
if __name__ == "__main__":
    # Initialize tracker with warehouse-optimized parameters
    tracker = EnhancedBagTracker(
        iou_threshold=0.25,  # Lower for deformable bags
        max_missing=20,  # Allow longer missing periods
        min_hits=2,  # Confirm tracks faster
        motion_threshold=60  # Allow more movement
    )

    # Define counting lines (example)
    counting_lines = [
        ((100, 200), (400, 200)),  # Horizontal line
        ((200, 100), (200, 400))  # Vertical line
    ]
    counter = LineCounter(counting_lines)

    # Track history for counting
    tracks_history = {}

    # Process frames
    frame_number = 0
    while True:  # Your video processing loop
        # Get detections from your detector
        # detections = your_detector.detect(frame)
        # Format: [(x, y, w, h, confidence), ...]

        detections = []  # Placeholder

        # Update tracker
        confirmed_tracks = tracker.update(detections, frame_number)

        # Update trajectory history
        for track_id, bbox in confirmed_tracks.items():
            if track_id not in tracks_history:
                tracks_history[track_id] = []

            center = tracker._get_bbox_center(bbox)
            tracks_history[track_id].append(center)

            # Limit history length
            if len(tracks_history[track_id]) > 50:
                tracks_history[track_id] = tracks_history[track_id][-50:]

        # Update counts
        counter.update_counts(tracks_history)

        # Get current counts
        counts = counter.get_counts()
        print(f"Frame {frame_number}: Line counts: {counts}")

        frame_number += 1
        # break  # Remove this in actual implementation