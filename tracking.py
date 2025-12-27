import cv2
import numpy as np
from ultralytics import YOLO
import supervision as sv
from collections import deque
from google.colab.patches import cv2_imshow

# Settings
VIDEO_PATH = "/content/videos/Video4.mp4"
OUTPUT_PATH = "/content/annotated_tracking_V4.mp4"
CONF_THRESH = 0.2
IOU_THRESH = 0.5
PREVIEW_EVERY = 30 

# Map dimensions
MAP_WIDTH = 600
MAP_HEIGHT = 300
MAP_PADDING = 55

# Colors
COLOR_TEAM_LEFT = sv.Color.from_hex("#E63946")
COLOR_TEAM_RIGHT = sv.Color.from_hex("#457B9D")
COLOR_REFEREE = sv.Color.from_hex("#FFD700")
COLOR_SPECTATOR = sv.Color.from_hex("#A0A0A0")
COLOR_BALL_PREDICTED = (0, 165, 255)
COLOR_BALL_DETECTED = (0, 255, 0)
COLOR_BALL_TRAIL = (0, 255, 255)

# Homography points
SOURCE_POINTS = np.array([
    [898.0, 783.0], # Top-left
    [2567.0, 812.0], # Top-right
    [2787.0, 1106.0], # Bottom-right
    [746.0, 1059.0] # Bottom-left
], dtype=np.float32)

TARGET_POINTS = np.array([
    [0, 0], [MAP_WIDTH, 0], [MAP_WIDTH, MAP_HEIGHT], [0, MAP_HEIGHT]
], dtype=np.float32)


# TRACKERS 

class ViewTransformer:
    def __init__(self, source: np.ndarray, target: np.ndarray):
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target) # Transformation matrix

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0: return points
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

class VolleyballTracker:
    def __init__(self, max_history=15, max_distance=200):
        self.ball_positions = deque(maxlen=max_history)
        self.frames_since_detection = 0
        self.max_frames_missing = 5
        self.max_distance = max_distance

    def predict_position(self):
        if len(self.ball_positions) < 2: return None
        recent = list(self.ball_positions)[-3:]
        velocities = [(recent[i+1][0] - recent[i][0], recent[i+1][1] - recent[i][1]) for i in range(len(recent) - 1)]
        avg_vx = sum(v[0] for v in velocities) / len(velocities)
        avg_vy = sum(v[1] for v in velocities) / len(velocities)
        return (recent[-1][0] + avg_vx, recent[-1][1] + avg_vy)

    def update(self, ball_detections):
        if len(ball_detections.xyxy) == 0:
            self.frames_since_detection += 1
            if self.frames_since_detection <= self.max_frames_missing:
                return self.predict_position(), True
            return None, False

        centers = [((box[0]+box[2])/2, (box[1]+box[3])/2) for box in ball_detections.xyxy]
        
        if not self.ball_positions:
            self.ball_positions.append(centers[0])
            self.frames_since_detection = 0
            return centers[0], False

        ref = self.predict_position() or self.ball_positions[-1]
        dists = [np.linalg.norm(np.array(ref) - np.array(c)) for c in centers]
        idx = np.argmin(dists)

        if dists[idx] < self.max_distance:
            self.ball_positions.append(centers[idx])
        else:
            self.ball_positions.clear()
            self.ball_positions.append(centers[idx])
            
        self.frames_since_detection = 0
        return centers[idx], False

def draw_court_map(width=600, height=300, padding=50):
    canvas = np.ones((height + 2 * padding, width + 2 * padding, 3), dtype=np.uint8) * 30 
    x0, y0 = padding, padding
    cv2.rectangle(canvas, (x0, y0), (x0 + width, y0 + height), (255, 255, 255), 2)
    mid_x = x0 + width // 2
    cv2.line(canvas, (mid_x, y0 - 10), (mid_x, y0 + height + 10), (0, 255, 255), 2)
    attack_zone_width = width // 6
    cv2.line(canvas, (mid_x - attack_zone_width, y0), (mid_x - attack_zone_width, y0 + height), (180, 180, 180), 1)
    cv2.line(canvas, (mid_x + attack_zone_width, y0), (mid_x + attack_zone_width, y0 + height), (180, 180, 180), 1)
    return canvas

def main():
    model = YOLO("yolov8x.pt") 
    person_tracker = sv.ByteTrack(track_activation_threshold=0.25, lost_track_buffer=40, frame_rate=30)
    ball_tracker = VolleyballTracker()
    view_transformer = ViewTransformer(SOURCE_POINTS, TARGET_POINTS)

    cap = cv2.VideoCapture(VIDEO_PATH)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(cv2.CAP_PROP_FPS)
    writer = cv2.VideoWriter(OUTPUT_PATH, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    base_map = draw_court_map(MAP_WIDTH, MAP_HEIGHT, MAP_PADDING)
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        results = model(frame, conf=CONF_THRESH, iou=IOU_THRESH, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        
        ball_dets = detections[detections.class_id == 32]
        current_ball_pos, is_predicted = ball_tracker.update(ball_dets)
        
        person_dets = detections[detections.class_id == 0]
        person_dets = person_tracker.update_with_detections(person_dets)

        points_camera = person_dets.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
        points_map = view_transformer.transform_points(points_camera)

        left_team_idxs, right_team_idxs, referee_idxs, spectator_idxs = [], [], [], []
        net_x = MAP_WIDTH / 2
        court_buffer_x, court_buffer_y = 20, 60 

        for i, (mx, my) in enumerate(points_map): # mx/my are mapped coordinates
            # Referee check
            if abs(mx - net_x) < 35 and ((my < 0) or (my > MAP_HEIGHT)):
                referee_idxs.append(i)
                continue

            # Spectator check
            if not ((-court_buffer_x <= mx <= MAP_WIDTH + court_buffer_x) and 
                    (-court_buffer_y <= my <= MAP_HEIGHT + court_buffer_y)):
                spectator_idxs.append(i)
                continue

            # Team assignment
            if mx < net_x:
                left_team_idxs.append(i)
            else:
                right_team_idxs.append(i)

        def annotate(scene, idxs, color, label):
            if not idxs: return scene
            subset = person_dets[idxs]
            thickness = 1 if label == "Spec" else 2
            scene = sv.BoxAnnotator(color=color, thickness=thickness).annotate(scene, subset)
            if label != "Spec":
                labels = [f"{label} {tid}" for tid in subset.tracker_id]
                scene = sv.LabelAnnotator(color=color, text_scale=0.4).annotate(scene, subset, labels)
            return scene

        frame = annotate(frame, left_team_idxs, COLOR_TEAM_LEFT, "L")
        frame = annotate(frame, right_team_idxs, COLOR_TEAM_RIGHT, "R")
        frame = annotate(frame, referee_idxs, COLOR_REFEREE, "Ref")
        frame = annotate(frame, spectator_idxs, COLOR_SPECTATOR, "Spec")

        if current_ball_pos:
            cx, cy = int(current_ball_pos[0]), int(current_ball_pos[1])
            for k in range(len(ball_tracker.ball_positions)-1):
                p1, p2 = ball_tracker.ball_positions[k], ball_tracker.ball_positions[k+1]
                cv2.line(frame, (int(p1[0]), int(p1[1])), (int(p2[0]), int(p2[1])), COLOR_BALL_TRAIL, 2)
            cv2.circle(frame, (cx, cy), 10, COLOR_BALL_PREDICTED if is_predicted else COLOR_BALL_DETECTED, -1)

        # Draw Radar
        radar = base_map.copy()
        def plot_points(idxs, color):
            for i in idxs:
                px, py = points_map[i]
                dx, dy = int(px) + MAP_PADDING, int(py) + MAP_PADDING
                if 0 <= dx < radar.shape[1] and 0 <= dy < radar.shape[0]:
                    cv2.circle(radar, (dx, dy), 5, color.as_bgr(), -1)
                    cv2.circle(radar, (dx, dy), 6, (255,255,255), 1)

        plot_points(left_team_idxs, COLOR_TEAM_LEFT)
        plot_points(right_team_idxs, COLOR_TEAM_RIGHT)
        plot_points(referee_idxs, COLOR_REFEREE)
        plot_points(spectator_idxs, COLOR_SPECTATOR)

        if current_ball_pos:
            b_map = view_transformer.transform_points(np.array([[current_ball_pos]]))
            bx, by = int(b_map[0][0]) + MAP_PADDING, int(b_map[0][1]) + MAP_PADDING
            if 0 <= bx < radar.shape[1] and 0 <= by < radar.shape[0]:
                cv2.circle(radar, (bx, by), 6, COLOR_BALL_PREDICTED, -1)

        # Overlay Radar
        h_frame, w_frame, _ = frame.shape
        radar_w = w_frame // 3
        scale = radar_w / radar.shape[1]
        radar_res = cv2.resize(radar, None, fx=scale, fy=scale)
        ry, rx, _ = radar_res.shape
        frame[h_frame - ry - 20 : h_frame - 20, 20 : 20 + rx] = radar_res
        cv2.rectangle(frame, (20, h_frame - ry - 20), (20 + rx, h_frame - 20), (255,255,255), 2)

        writer.write(frame)
        if frame_idx % PREVIEW_EVERY == 0:
            cv2_imshow(cv2.resize(frame, (600, 338)))
        frame_idx += 1

    cap.release()
    writer.release()

if __name__ == "__main__":
    main()
