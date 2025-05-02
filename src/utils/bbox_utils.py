from shapely.geometry import Polygon
import numpy as np

def xywh_to_xyxy(bbox_xywh):
    """Converts bounding box from [x, y, w, h] to [x1, y1, x2, y2] format."""
    x, y, w, h = bbox_xywh
    x1, y1 = x, y
    x2, y2 = x + w, y + h
    return [x1, y1, x2, y2]

def xyxy_to_xywh(bbox_xyxy):
    """Converts bounding box from [x1, y1, x2, y2] to [x, y, w, h] format."""
    x1, y1, x2, y2 = bbox_xyxy
    x, y = x1, y1
    w, h = x2 - x1, y2 - y1
    return [x, y, w, h]

def get_bbox_center(bbox_xywh):
    """Calculates the center point [cx, cy] of a bounding box."""
    x, y, w, h = bbox_xywh
    return [x + w / 2, y + h / 2]

def calculate_iou(bbox1_xywh, bbox2_xywh):
    """Calculates Intersection over Union (IoU) between two bounding boxes."""
    bbox1_xyxy = xywh_to_xyxy(bbox1_xywh)
    bbox2_xyxy = xywh_to_xyxy(bbox2_xywh)

    # Determine the coordinates of the intersection rectangle
    x_left = max(bbox1_xyxy[0], bbox2_xyxy[0])
    y_top = max(bbox1_xyxy[1], bbox2_xyxy[1])
    x_right = min(bbox1_xyxy[2], bbox2_xyxy[2])
    y_bottom = min(bbox1_xyxy[3], bbox2_xyxy[3])

    # Compute the area of intersection rectangle
    intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)

    # Compute the area of both bounding boxes
    bbox1_area = (bbox1_xyxy[2] - bbox1_xyxy[0]) * (bbox1_xyxy[3] - bbox1_xyxy[1])
    bbox2_area = (bbox2_xyxy[2] - bbox2_xyxy[0]) * (bbox2_xyxy[3] - bbox2_xyxy[1])

    # Compute the intersection over union by taking the intersection area and dividing it by the sum of prediction + ground-truth areas - the intersection area
    iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
    return iou

def is_contained(child_bbox_xywh, parent_bbox_xywh, iou_threshold=0.7):
    """Checks if a child bbox is largely contained within a parent bbox."""
    # More robust check using polygon intersection
    child_poly = Polygon(
        [(child_bbox_xywh[0], child_bbox_xywh[1]),
         (child_bbox_xywh[0] + child_bbox_xywh[2], child_bbox_xywh[1]),
         (child_bbox_xywh[0] + child_bbox_xywh[2], child_bbox_xywh[1] + child_bbox_xywh[3]),
         (child_bbox_xywh[0], child_bbox_xywh[1] + child_bbox_xywh[3])]
    )
    parent_poly = Polygon(
         [(parent_bbox_xywh[0], parent_bbox_xywh[1]),
         (parent_bbox_xywh[0] + parent_bbox_xywh[2], parent_bbox_xywh[1]),
         (parent_bbox_xywh[0] + parent_bbox_xywh[2], parent_bbox_xywh[1] + parent_bbox_xywh[3]),
         (parent_bbox_xywh[0], parent_bbox_xywh[1] + parent_bbox_xywh[3])]
    )

    if not parent_poly.intersects(child_poly):
        return False

    intersection_poly = parent_poly.intersection(child_poly)
    # Check if a large percentage of the child area is within the parent
    return intersection_poly.area / child_poly.area > iou_threshold

def calculate_distance(bbox1_xywh, bbox2_xywh):
    """Calculates the distance between the centers of two bounding boxes."""
    center1 = get_bbox_center(bbox1_xywh)
    center2 = get_bbox_center(bbox2_xywh)
    return np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)

# Add other utility functions as needed (e.g., checking overlap percentage)
