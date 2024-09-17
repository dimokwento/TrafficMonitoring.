import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
from datetime import datetime

model = YOLO("yolov8x.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
line_counter = sv.LineZone(start=sv.Point(100, 800), end=sv.Point(1300,560))
line_annotator = sv.LineZoneAnnotator()

# Define the class IDs for vehicles
vehicle_class_ids = [2, 3, 5, 7]  # car, motorbike, bus, truck

# Initialize counts
in_count = 0
out_count = 0

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    global in_count, out_count
    print("Callback function called")
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = detections[detections.confidence > 0.5]

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
    frame_in_count = line_counter.in_count
    frame_out_count = line_counter.out_count
    in_count += frame_in_count
    out_count += frame_out_count
    
    print("Counting:")
    print(f"In Count: {line_counter.in_count}, Out Count: {line_counter.out_count}, Total: {line_counter.in_count + line_counter.out_count}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Display the annotated frame
    cv2.imshow("Counting", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()
    return annotated_frame

sv.process_video(
    source_path="v5.mp4",
    target_path="results.mp4",
    callback=callback
)