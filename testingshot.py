import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
from datetime import datetime
import csv
import matplotlib.pyplot as plt
import threading

# Initialize the YOLO model
yolo_model = YOLO("yolov8m.pt")

# Initialize the ByteTrack tracker
byte_tracker = sv.ByteTrack()

# Initialize annotators
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()

# Initialize the line counter
line_counter = sv.LineZone(start=sv.Point(100, 600), end=sv.Point(1000, 300))
line_annotator = sv.LineZoneAnnotator()

# Define the class IDs for vehicles
vehicle_class_ids = [2, 3, 5, 7]  # car, motorbike, bus, truck

# Initialize vehicle counts
vehicle_in_count = 0
vehicle_out_count = 0

# Open the CSV file for writing
with open('counting_results.csv', 'w', newline='') as csvfile:
    fieldnames = ['Date', 'In Count', 'Out Count', 'Total']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    # Write the header
    writer.writeheader()

    # Create a queue to hold the frames
    frame_queue = []

    # Function to process frames
    def process_frame(frame: np.ndarray, frame_index: int) -> np.ndarray:
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
        frame_in_count = line_counter.in_count
        frame_out_count = line_counter.out_count
        vehicle_in_count += frame_in_count
        vehicle_out_count += frame_out_count

        # Print and write the counting results
        date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("Counting:")
        print(f"In Count: {line_counter.in_count}, Out Count: {line_counter.out_count}, Total: {line_counter.in_count + line_counter.out_count}")
        print(f"Date: {date}")
        writer.writerow({
            'Date': date,
            'In Count': line_counter.in_count,
            'Out Count': line_counter.out_count,
            'Total': line_counter.in_count + line_counter.out_count
        })

        return annotated_frame

    # Function to play video
    def play_video():
        global frame_queue
        while True:
            if frame_queue:
                frame = frame_queue.pop(0)
                cv2.imshow("Advance Counting", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    exit()
            else:
                cv2.waitKey(1)

    # Start the video playing thread
    video_thread = threading.Thread(target=play_video)
    video_thread.daemon = True
    video_thread.start()

    # Process the videos
    def process_videos(video_paths):
        for video_path in video_paths:
            cap = cv2.VideoCapture(video_path)
            frame_index = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                annotated_frame = process_frame(frame, frame_index)
                frame_queue.append(annotated_frame)
                frame_index += 1
            cap.release()

    # Process the videos
    video_paths = ["My Video3.mp4"]  # Add your video paths here
    process_videos(video_paths)

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

# Call the function to create a bar graph
create_bar_graph('counting_results.csv')