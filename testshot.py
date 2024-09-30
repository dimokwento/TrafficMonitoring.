import numpy as np
from supervision.tracker import byte_tracker
from supervision.detection.line_zone import LineZone, LineZoneAnnotator
import supervision as sv
from ultralytics import YOLO
import cv2
from datetime import datetime
import threading
from queue import Queue
from typing import Tuple
#import ffmpeg

# Define the vehicle class IDs
vehicle_class_ids = [0, 1, 2, 3, 5, 7]  # Car, Motorcycle, Bus, Truck, Bicycle
# Load the YOLO model with a lower confidence threshold
model = YOLO("yolov8n.pt")
model.conf = 0.3 
# Create a tracker
tracker = sv.ByteTrack()
# Create a line counter
line_counter = sv.LineZone(start=sv.Point(100, 600), end=sv.Point(1000, 200))
# Create a box annotator
box_annotator = sv.BoxAnnotator()
# Create a label annotator
label_annotator = sv.LabelAnnotator()
# Create a trace annotator
trace_annotator = sv.TraceAnnotator()
# Create a line annotator
line_annotator = sv.LineZoneAnnotator()
# Create a queue to hold frames to be processed
frame_queue = Queue(maxsize=128)
# Create a queue to hold processed frames
annotated_frame_queue = Queue(maxsize=128)
# Create a lock to synchronize access to the counts
count_lock = threading.Lock()

# Define the callback function
def process_frame(frame: np.ndarray, _: int) -> Tuple[np.ndarray, int, int]:
    global line_counter
    print("Callback function called")
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.confidence > 0.3]

    # Filter detections to only include vehicles
    vehicle_mask = np.in1d(detections.class_id, vehicle_class_ids)
    vehicle_detections = sv.Detections(
        xyxy=detections.xyxy[vehicle_mask],
        class_id=detections.class_id[vehicle_mask],
        confidence=detections.confidence[vehicle_mask],
    )

    print("Vehicle detections:", vehicle_detections)

    vehicle_detections = tracker.update_with_detections(vehicle_detections)

    print("Tracker IDs:", vehicle_detections.tracker_id)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(vehicle_detections.class_id, vehicle_detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=vehicle_detections)
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=vehicle_detections, labels=labels)
    annotated_frame = trace_annotator.annotate(
        annotated_frame, detections=vehicle_detections)
    line_counter.trigger(detections=vehicle_detections)
    annotated_frame = line_annotator.annotate(
        annotated_frame, line_counter=line_counter)
    in_count_delta = line_counter.in_count
    out_count_delta = line_counter.out_count

    return annotated_frame, in_count_delta, out_count_delta

# Define a faster processing function
def process_frame_fast(frame: np.ndarray, _: int) -> Tuple[np.ndarray, int, int]:
    global line_counter
    print("Fast processing function called")
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.confidence > 0.3]

    # Filter detections to only include vehicles
    vehicle_mask = np.in1d(detections.class_id, vehicle_class_ids)
    vehicle_detections = sv.Detections(
        xyxy=detections.xyxy[vehicle_mask],
        class_id=detections.class_id[vehicle_mask],
        confidence=detections.confidence[vehicle_mask],
    )

    vehicle_detections = tracker.update_with_detections(vehicle_detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(vehicle_detections.class_id, vehicle_detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=vehicle_detections)
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=vehicle_detections, labels=labels)
    line_counter.trigger(detections=vehicle_detections)
    annotated_frame = line_annotator.annotate(
        annotated_frame, line_counter=line_counter)
    in_count_delta = line_counter.in_count
    out_count_delta = line_counter.out_count

    return annotated_frame, in_count_delta, out_count_delta

# Define a worker function to process frames
def worker():
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        annotated_frame, in_count_delta, out_count_delta = process_frame_fast(frame, 0)
        annotated_frame_queue.put((annotated_frame, in_count_delta, out_count_delta))
        frame_queue.task_done()

# Create multiple worker threads with reduced number of threads
num_workers = 4
threads = []
for _ in range(num_workers):
    t = threading.Thread(target=worker)
    t.start()
    threads.append(t)

# Open the input video file
cap = cv2.VideoCapture("v1.mp4")

# Get the video properties
fps = cap.get(cv2.CAP_PROP_FPS) / 120000  # Reduce the frame rate
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) // 2  # Reduce the video resolution
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) // 2

# Create a VideoWriter object to write the output video with XVID codec
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter("results.mp4", fourcc, fps * 1000, (width, height))

in_count = 0
out_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Put the frame in the frame queue
    frame_queue.put(frame)

    # Get the annotated frame and count deltas from the annotated frame queue
    annotated_frame, in_count_delta, out_count_delta = annotated_frame_queue.get()
    annotated_frame_queue.task_done()

    # Update the counts
    with count_lock:
        in_count += in_count_delta
        out_count += out_count_delta

    # Write the annotated frame to the output video
    out.write(annotated_frame)

    # Display the annotated frame
    cv2.imshow("Counting", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Print the current counts
    print(f"In Count: {line_counter.in_count}, Out Count: {line_counter.out_count}, Total: {line_counter.in_count + line_counter.out_count}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

# Stop the worker threads
for _ in range(num_workers):
    frame_queue.put(None)
for t in threads:
    t.join()

cap.release()
out.release()
cv2.destroyAllWindows()