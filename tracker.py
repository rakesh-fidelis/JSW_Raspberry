# import math
#
#
# class Tracker:
#     def __init__(self):
#         # Store the center positions of the objects
#         self.center_points = {}
#         # Keep the count of the IDs
#         # each time a new object id detected, the count will increase by one
#         self.id_count = 0
#
#
#     def update(self, objects_rect):
#         # Objects boxes and ids
#         objects_bbs_ids = []
#
#         # Get center point of new object
#         for rect in objects_rect:
#             x, y, w, h = rect
#             cx = (x + x + w) // 2
#             cy = (y + y + h) // 2
#
#             # Find out if that object was detected already
#             same_object_detected = False
#             for id, pt in self.center_points.items():
#                 dist = math.hypot(cx - pt[0], cy - pt[1])
#
#                 if dist < 35:
#                     self.center_points[id] = (cx, cy)
#                     objects_bbs_ids.append([x, y, w, h, id])
#                     same_object_detected = True
#                     break
#
#             # New object is detected we assign the ID to that object
#             if same_object_detected is False:
#                 self.center_points[self.id_count] = (cx, cy)
#                 objects_bbs_ids.append([x, y, w, h, self.id_count])
#                 self.id_count += 1
#
#         # Clean the dictionary by center points to remove IDS not used anymore
#         new_center_points = {}
#         for obj_bb_id in objects_bbs_ids:
#             _, _, _, _, object_id = obj_bb_id
#             center = self.center_points[object_id]
#             new_center_points[object_id] = center
#
#         # Update dictionary with IDs not used removed
#         self.center_points = new_center_points.copy()
#         return objects_bbs_ids

# tracker.py (Example/Placeholder)
import numpy as np
import time


class Tracker:
    def __init__(self):
        self.next_id = 0
        self.objects = {}  # Stores {'id': {'bbox': [x,y,w,h], 'last_seen': timestamp}}
        self.max_age = 30  # How many frames to keep an object if not seen (adjust based on FPS/FRAME_SKIP)
        self.iou_threshold = 0.3  # IoU threshold for matching detections to existing objects

    def update(self, detections):
        """
        Updates the tracker with new detections.
        detections: List of [x, y, w, h] for new detections.
        Returns: List of tracked objects, each a tuple (x, y, w, h, id)
        """
        updated_objects = []
        current_object_ids = set()

        # Step 1: Match new detections to existing tracked objects
        matched_detections = set()
        for det_idx, new_det in enumerate(detections):
            best_match_id = -1
            best_iou = -1

            for obj_id, obj_info in self.objects.items():
                tracked_bbox = obj_info['bbox']
                iou = self._calculate_iou(new_det, tracked_bbox)
                if iou > self.iou_threshold and iou > best_iou:
                    best_iou = iou
                    best_match_id = obj_id

            if best_match_id != -1:
                # Update existing object
                self.objects[best_match_id]['bbox'] = new_det
                self.objects[best_match_id]['last_seen'] = time.time()  # Or frame_id
                updated_objects.append((*new_det, best_match_id))
                current_object_ids.add(best_match_id)
                matched_detections.add(det_idx)

        # Step 2: Create new objects for unmatched detections
        for det_idx, new_det in enumerate(detections):
            if det_idx not in matched_detections:
                self.objects[self.next_id] = {'bbox': new_det, 'last_seen': time.time()}  # Or frame_id
                updated_objects.append((*new_det, self.next_id))
                current_object_ids.add(self.next_id)
                self.next_id += 1

        # Step 3: Remove old, unseen objects
        # This basic tracker doesn't use max_age directly for deletion,
        # but a more robust one would. For simplicity, we just return
        # the currently updated/new objects.

        # A more complete tracker would iterate self.objects and remove those
        # not in current_object_ids and whose 'last_seen' is too old.
        # For this basic version, `self.objects` can grow.

        # A simple cleanup if objects haven't been seen for a while (based on time)
        keys_to_delete = []
        for obj_id, obj_info in self.objects.items():
            if obj_id not in current_object_ids and (time.time() - obj_info['last_seen']) > (
                    self.max_age / 15):  # Assuming 15 FPS input, rough estimate
                keys_to_delete.append(obj_id)
        for obj_id in keys_to_delete:
            del self.objects[obj_id]

        return updated_objects

    def _calculate_iou(self, box1, box2):
        # box: [x, y, w, h]
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2

        # Convert to [x1, y1, x2, y2]
        box1_coords = [x1, y1, x1 + w1, y1 + h1]
        box2_coords = [x2, y2, x2 + w2, y2 + h2]

        # Determine the coordinates of the intersection rectangle
        x_left = max(box1_coords[0], box2_coords[0])
        y_top = max(box1_coords[1], box2_coords[1])
        x_right = min(box1_coords[2], box2_coords[2])
        y_bottom = min(box1_coords[3], box2_coords[3])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        box1_area = w1 * h1
        box2_area = w2 * h2

        iou = intersection_area / float(box1_area + box2_area - intersection_area)
        return iou