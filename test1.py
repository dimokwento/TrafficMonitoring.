import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import multiprocessing

# Initialize the YOLO model
yolo_model = YOLO("yolov8x.pt")

# Initialize the ByteTrack tracker
byte_tracker = sv.ByteTrack()

# Initialize annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

# Initialize the line counter
line_counter = sv.LineZone(start=sv.Point(100, 700), end=sv.Point(1050, 300))
line_annotator = sv.LineZoneAnnotator()

# Define the class IDs for vehicles
vehicle_class_ids = [2, 3, 5, 7]  # car, motorbike, bus, truck

# Initialize vehicle counts
vehicle_in_count = multiprocessing.Value('i', 0)
vehicle_out_count = multiprocessing.Value('i', 0)

# Function to process frames
def process_frame(frame: np.ndarray, frame_index: int, video_path: str) -> np.ndarray:
    global vehicle_in_count, vehicle_out_count

    # Detect objects in the frame using YOLO
    detection_results = yolo_model(frame)[0]
    detections = sv.Detections.from_ultralytics(detection_results)

    # Filter detections to only include vehicles
    vehicle_mask = np.in1d(detections.class_id, vehicle_class_ids)
    vehicle_detections = sv.Detections(
        xyxy=detections.xyxy[vehicle_mask],
        class_id=detections.class_id[vehicle_mask],
        confidence=detections.confidence[vehicle_mask],
    )

    # Update the tracker with the vehicle detections
    vehicle_detections = byte_tracker.update_with_detections(vehicle_detections)

    # Create labels for the vehicle detections
    labels = [
        f"#{tracker_id} {detection_results.names[class_id]}"
        for class_id, tracker_id
        in zip(vehicle_detections.class_id, vehicle_detections.tracker_id)
    ]

    # Annotate the frame with boxes, labels, and traces
    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=vehicle_detections)
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=vehicle_detections, labels=labels)
    annotated_frame = trace_annotator.annotate(
        annotated_frame, detections=vehicle_detections)

    # Update the line counter
    line_counter.trigger(detections=vehicle_detections)
    annotated_frame = line_annotator.annotate(
        annotated_frame, line_counter=line_counter)

    # Update the vehicle counts
    with vehicle_in_count.get_lock():
        vehicle_in_count.value += line_counter.in_count
    with vehicle_out_count.get_lock():
        vehicle_out_count.value += line_counter.out_count

    # Print and write the counting results
    date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("Counting:")
    print(f"In Count: {line_counter.in_count}, Out Count: {line_counter.out_count}, Total: {line_counter.in_count + line_counter.out_count}")
    print(f"Date: {date}")

    # Write the counting results to a CSV file
    with open(f'{video_path}.csv', 'a', newline='') as csvfile:
        fieldnames = ['Date', 'In Count', 'Out Count', 'Total']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()
        writer.writerow({
            'Date': date,
            'In Count': line_counter.in_count,
            'Out Count': line_counter.out_count,
            'Total': line_counter.in_count + line_counter.out_count
        })

    return annotated_frame

# Function to process a video
def process_video(video_path: str):
    cap = cv2.VideoCapture(video_path)
    frame_index = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        annotated_frame = process_frame(frame, frame_index, video_path)
        cv2.imshow(f"Advance Counting {video_path}", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break
        frame_index += 1
    cap.release()

# Function to create a bar graph
def create_bar_graph(csv_file):
    days = []
    hours = []
    with open(csv_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            date = row['Date'].split(' ')[0]
            hour = int(row['Date'].split(' ')[1].split(':')[0])
            if date not in days:
                days.append(date)
                hours.append([0]*24)
            hours[days.index(date)][hour] += int(row['Total'])
    
    # Create a bar graph with days on the x-axis and hours on the y-axis
    fig, axs = plt.subplots(len(days), figsize=(10, 7*len(days)))
    if len(days) == 1:  # If there's only one day, axs will be a single Axes object
        axs = [axs]  # Convert it to a list of Axes objects
    for i in range(len(days)):
        axs[i].bar(range(24), hours[i])
        axs[i].set_title(days[i])
        axs[i].set_xlabel('Hours')
        axs[i].set_ylabel('Total Count')
    plt.tight_layout()
    plt.show()

# Process the videos in parallel
if __name__ == "__main__":
    video_paths = ["v2.mp4"]  # Add your video paths here

    # Create a multiprocessing pool with 4 worker processes
    with multiprocessing.Pool(processes=8) as pool:
        # Process each video in parallel
        pool.map(process_video, video_paths)

    for video_path in video_paths:
        create_bar_graph(f'{video_path}.csv')