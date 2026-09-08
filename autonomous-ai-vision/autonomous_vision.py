import cv2
import time
import psutil
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from ultralytics import YOLO

MODEL_NAME = "yolov8n.pt"
BENCHMARK_DURATION = 60
CAMERA_INDEX = 0
CONFIDENCE = 0.3

model = YOLO(MODEL_NAME)
cap = cv2.VideoCapture(CAMERA_INDEX)

data = []
track_history = defaultdict(list)
unique_ids_seen = set()

start_time = time.time()
prev_time = time.time()
heatmap = None

while True:
    success, frame = cap.read()

    if not success:
        print("Camera not working")
        break

    height, width = frame.shape[:2]

    if heatmap is None:
        heatmap = np.zeros((height, width), dtype=np.float32)

    elapsed_time = time.time() - start_time

    inference_start = time.time()
    results = model.track(
        frame,
        persist=True,
        conf=CONFIDENCE,
        verbose=False
    )
    inference_end = time.time()

    annotated_frame = frame.copy()

    current_time = time.time()
    fps = 1 / (current_time - prev_time)
    prev_time = current_time

    cpu_usage = psutil.cpu_percent()
    ram_usage = psutil.virtual_memory().percent
    inference_time_ms = (inference_end - inference_start) * 1000

    tracked_ids = []
    people_count = 0
    object_count = 0
    target_id = None
    target_name = None
    target_offset_x = None
    target_direction = "No target"

    center_screen_x = width // 2
    center_screen_y = height // 2

    cv2.line(annotated_frame, (center_screen_x, 0), (center_screen_x, height), (255, 255, 255), 1)
    cv2.circle(annotated_frame, (center_screen_x, center_screen_y), 6, (255, 255, 255), -1)

    possible_targets = []

    if results[0].boxes.id is not None:
        ids = results[0].boxes.id.cpu().numpy().astype(int).tolist()
        boxes = results[0].boxes.xyxy.cpu().numpy().tolist()
        confs = results[0].boxes.conf.cpu().numpy().tolist()
        classes = results[0].boxes.cls.cpu().numpy().astype(int).tolist()

        for box, track_id, confidence, cls in zip(boxes, ids, confs, classes):
            x1, y1, x2, y2 = box
            class_name = model.names[cls]

            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            tracked_ids.append(track_id)
            unique_ids_seen.add(track_id)
            object_count += 1

            if class_name == "person":
                people_count += 1

            distance_from_center = abs(center_x - center_screen_x)
            possible_targets.append((distance_from_center, track_id, class_name, center_x, center_y))

            track_history[track_id].append((center_x, center_y))

            if len(track_history[track_id]) > 50:
                track_history[track_id].pop(0)

            if 0 <= center_x < width and 0 <= center_y < height:
                heatmap[center_y, center_x] += 1

            points = track_history[track_id]

            for i in range(1, len(points)):
                cv2.line(annotated_frame, points[i - 1], points[i], (0, 255, 255), 2)

            speed = 0

            if len(points) >= 2:
                x_old, y_old = points[-2]
                x_new, y_new = points[-1]
                distance = math.sqrt((x_new - x_old) ** 2 + (y_new - y_old) ** 2)
                speed = distance * fps

                dx = x_new - x_old
                dy = y_new - y_old

                predicted_x = int(x_new + dx * 10)
                predicted_y = int(y_new + dy * 10)

                cv2.circle(annotated_frame, (predicted_x, predicted_y), 7, (0, 0, 255), -1)
                cv2.putText(annotated_frame, "Prediction", (predicted_x + 10, predicted_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
            cv2.circle(annotated_frame, (center_x, center_y), 5, (0, 255, 255), -1)

            cv2.putText(annotated_frame, f"ID {track_id} | {class_name} | {confidence:.2f} | {speed:.0f}px/s",
                        (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 0), 2)

    if possible_targets:
        possible_targets.sort(key=lambda x: x[0])
        _, target_id, target_name, target_x, target_y = possible_targets[0]

        target_offset_x = target_x - center_screen_x

        if target_offset_x < -40:
            target_direction = "Turn LEFT"
        elif target_offset_x > 40:
            target_direction = "Turn RIGHT"
        else:
            target_direction = "CENTERED"

        cv2.circle(annotated_frame, (target_x, target_y), 12, (0, 0, 255), 3)
        cv2.putText(annotated_frame, f"TARGET {target_name} ID {target_id}", (target_x - 70, target_y - 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    dashboard_width = 380
    dashboard = np.zeros((height, dashboard_width, 3), dtype=np.uint8)
    dashboard[:] = (25, 25, 25)

    def write_dash(text, y, color=(255, 255, 255), scale=0.6):
        cv2.putText(dashboard, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2)

    write_dash("AI VISION DASHBOARD", 40, (255, 255, 255), 0.75)
    write_dash(f"Model: {MODEL_NAME}", 85, (200, 200, 200))
    write_dash(f"Confidence: {CONFIDENCE}", 115, (200, 200, 200))

    write_dash(f"FPS: {fps:.1f}", 165, (0, 255, 0))
    write_dash(f"Inference: {inference_time_ms:.1f} ms", 195, (255, 200, 200))
    write_dash(f"CPU: {cpu_usage}%", 225, (0, 255, 255))
    write_dash(f"RAM: {ram_usage}%", 255, (255, 255, 0))

    write_dash(f"Objects visible: {object_count}", 315, (200, 255, 200))
    write_dash(f"People visible: {people_count}", 345, (200, 255, 200))
    write_dash(f"Unique IDs seen: {len(unique_ids_seen)}", 375, (200, 200, 255))

    write_dash("AUTONOMOUS TARGETING", 435, (255, 255, 255), 0.65)
    write_dash(f"Target: {target_name}", 470, (0, 0, 255))
    write_dash(f"Target ID: {target_id}", 500, (0, 0, 255))
    write_dash(f"Direction: {target_direction}", 530, (0, 0, 255))

    if target_offset_x is not None:
        write_dash(f"X offset: {target_offset_x}px", 560, (0, 0, 255))

    combined_display = np.hstack((annotated_frame, dashboard))

    data.append({
        "time_seconds": elapsed_time,
        "model": MODEL_NAME,
        "fps": fps,
        "cpu_usage_percent": cpu_usage,
        "ram_usage_percent": ram_usage,
        "inference_time_ms": inference_time_ms,
        "objects_visible": object_count,
        "people_visible": people_count,
        "tracked_ids": str(tracked_ids),
        "unique_ids_seen": len(unique_ids_seen),
        "target_id": target_id,
        "target_name": target_name,
        "target_offset_x": target_offset_x,
        "target_direction": target_direction
    })

    cv2.imshow("AI Pro Autonomous Vision System", combined_display)

    if elapsed_time >= BENCHMARK_DURATION:
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

df = pd.DataFrame(data)
df.to_csv("pro_tracking_results.csv", index=False)

plt.figure()
plt.plot(df["time_seconds"], df["fps"])
plt.xlabel("Time (seconds)")
plt.ylabel("FPS")
plt.title("FPS Over Time")
plt.savefig("pro_fps_over_time.png")
plt.close()

plt.figure()
plt.plot(df["time_seconds"], df["inference_time_ms"])
plt.xlabel("Time (seconds)")
plt.ylabel("Inference Time (ms)")
plt.title("Inference Time Over Time")
plt.savefig("pro_inference_time.png")
plt.close()

plt.figure()
plt.plot(df["time_seconds"], df["cpu_usage_percent"])
plt.xlabel("Time (seconds)")
plt.ylabel("CPU Usage (%)")
plt.title("CPU Usage Over Time")
plt.savefig("pro_cpu_usage.png")
plt.close()

plt.figure()
plt.plot(df["time_seconds"], df["ram_usage_percent"])
plt.xlabel("Time (seconds)")
plt.ylabel("RAM Usage (%)")
plt.title("RAM Usage Over Time")
plt.savefig("pro_ram_usage.png")
plt.close()

plt.figure()
plt.plot(df["time_seconds"], df["objects_visible"])
plt.xlabel("Time (seconds)")
plt.ylabel("Objects Visible")
plt.title("Objects Visible Over Time")
plt.savefig("pro_objects_visible.png")
plt.close()

plt.figure()
plt.plot(df["time_seconds"], df["unique_ids_seen"])
plt.xlabel("Time (seconds)")
plt.ylabel("Unique IDs Seen")
plt.title("Unique Objects Tracked Over Time")
plt.savefig("pro_unique_ids.png")
plt.close()

# Improved heatmap
improved_heatmap = np.log1p(heatmap)

plt.figure(figsize=(10, 8))
plt.imshow(improved_heatmap, cmap="inferno")
plt.colorbar(label="Movement Intensity")
plt.title("Movement Heatmap")
plt.axis("off")
plt.savefig("pro_movement_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()

summary = f"""
AI Pro Autonomous Vision System - Benchmark Summary

Model used: {MODEL_NAME}
Benchmark duration: {BENCHMARK_DURATION} seconds
Confidence threshold: {CONFIDENCE}

Average FPS: {df["fps"].mean():.2f}
Average inference time: {df["inference_time_ms"].mean():.2f} ms
Average CPU usage: {df["cpu_usage_percent"].mean():.2f}%
Average RAM usage: {df["ram_usage_percent"].mean():.2f}%
Total unique objects tracked: {len(unique_ids_seen)}

Generated files:
- pro_tracking_results.csv
- pro_fps_over_time.png
- pro_inference_time.png
- pro_cpu_usage.png
- pro_ram_usage.png
- pro_objects_visible.png
- pro_unique_ids.png
- pro_movement_heatmap.png
"""

with open("AI_pro_project_summary.txt", "w") as file:
    file.write(summary)

print(summary)