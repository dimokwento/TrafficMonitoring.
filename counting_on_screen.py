import numpy as np
import supervision as sv
from ultralytics import YOLO
import cv2

model = YOLO("yolov8x.pt")
tracker = sv.ByteTrack()
box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()
trace_annotator = sv.TraceAnnotator()
line_counter = sv.LineZone(start=sv.Point(100, 600), end=sv.Point(1000, 300))
line_annotator = sv.LineZoneAnnotator()

def callback(frame: np.ndarray, _: int) -> np.ndarray:
    results = model(frame)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = tracker.update_with_detections(detections)

    labels = [
        f"#{tracker_id} {results.names[class_id]}"
        for class_id, tracker_id
        in zip(detections.class_id, detections.tracker_id)
    ]

    annotated_frame = box_annotator.annotate(
        frame.copy(), detections=detections)
    annotated_frame = label_annotator.annotate(
        annotated_frame, detections=detections, labels=labels)
    annotated_frame = trace_annotator.annotate(
        annotated_frame, detections=detections)

    line_counter.trigger(detections=detections)
    annotated_frame = line_annotator.annotate(
        annotated_frame, line_counter=line_counter)

    # Display the annotated frame
    cv2.imshow("Screen Count", annotated_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        cv2.destroyAllWindows()
        exit()
    return annotated_frame

sv.process_video(
    source_path="My Video5.mp4",
    target_path="result1.mp4",
    callback=callback
)