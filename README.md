# Autonomous AI Vision

This is a real-time computer vision project I built using YOLOv8 and OpenCV.

I originally started it as a basic webcam object detector, but kept adding to it because I wanted to experiment with what happens after an object has been detected, such as tracking it over time, keeping persistent IDs, predicting movement and measuring how well the system actually performs.

## What it does

The program detects objects from a live camera feed and tracks them across frames.

I added features including:

- real-time YOLOv8 object detection
- persistent object IDs
- object tracking and movement history
- trajectory visualisation
- motion prediction
- target selection
- detection heatmaps
- live FPS monitoring
- CPU and memory monitoring
- inference-time measurement
- CSV performance logging

## Performance testing

I also built a benchmarking mode instead of relying on how fast the program looked visually.

During one 60-second test, I measured:

- **17.34 FPS average**
- **43.64 ms average inference time**
- **20 unique objects detected/tracked**

The benchmark records performance information to a CSV file so the results can be analysed afterwards.

I used Pandas and Matplotlib to process and visualise the benchmark data.

## Tech used

- Python
- YOLOv8 / Ultralytics
- OpenCV
- NumPy
- Pandas
- Matplotlib
- psutil

## Why I built it

I wanted to get more experience with computer vision beyond simply running a pretrained detector.

A lot of the project ended up being about handling information between frames, measuring performance and turning individual detections into something that behaves more like a continuous vision system.

There are still plenty of things I could improve, especially the tracking and motion prediction when objects overlap, disappear from view or move quickly.

## Running the project

Install the dependencies with:

```bash
pip install -r requirements.txt
```

A webcam or other compatible camera is required for live detection.

---

Built by **Saleh Pour**  
Robotics & Artificial Intelligence, University of Hertfordshire
