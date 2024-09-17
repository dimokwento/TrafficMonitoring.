import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2
import multiprocessing

model = YOLO("yolov8x.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator()  # Update to use BoxAnnotator instead of BoundingBoxAnnotator
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
#line_counter = sv.LineZone(start=sv.Point(0, 400), end=sv.Point(1300, 400))
line_annotator = sv.LineZoneAnnotator()

# Define the class IDs for vehicles
vehicle_class_ids = [2, 3, 5, 7]  # car, motorbike, bus, truck

def process_frame(frame: np.ndarray) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)

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
    annotated_frame = trace_annotator.annotate(
        annotated_frame, detections=vehicle_detections)

    #line_counter.trigger(detections=vehicle_detections)
    #annotated_frame = line_annotator.annotate(
     #   annotated_frame, line_counter=line_counter)

    return annotated_frame

def process_video(source_path: str, target_path: str) -> None:
    cap = cv2.VideoCapture(source_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(target_path, fourcc, fps, (width, height))

    with multiprocessing.Pool(processes=4) as pool:  # Adjust the number of processes based on your CPU cores
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            annotated_frame = pool.apply_async(process_frame, (frame,)).get()
            out.write(annotated_frame)
            cv2.imshow("Detection", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break

    cap.release()
    out.release()

if __name__ == '__main__':
    process_video("c1.mp4", "result.mp4")