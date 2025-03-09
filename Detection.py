import streamlit as st
import cv2
import numpy as np
import torch
import supervision as sv
from PIL import Image

@st.cache_resource
def load_model():
    model = torch.hub.load('ultralytics/yolov5', 'yolov5x6')
    return model

def process_frame(frame: np.ndarray, model) -> np.ndarray:
    results = model(frame, size=1280)
    detections = sv.Detections.from_yolov5(results)
    detections = detections[(detections.class_id == 0) & (detections.confidence > 0.5)]
    
    # Create polygons for zones (same as before)
    frame_width = frame.shape[1]
    frame_height = frame.shape[0]
    grid_width = frame_width // 2
    grid_height = frame_height // 2

    polygons = [
        np.array([[0, 0], [grid_width, 0], [grid_width, grid_height], [0, grid_height]], np.int32),      # Top-left
        np.array([[grid_width, 0], [frame_width, 0], [frame_width, grid_height], [grid_width, grid_height]], np.int32),  # Top-right
        np.array([[0, grid_height], [grid_width, grid_height], [grid_width, frame_height], [0, frame_height]], np.int32), # Bottom-left
        np.array([[grid_width, grid_height], [frame_width, grid_height], [frame_width, frame_height], [grid_width, frame_height]], np.int32)  # Bottom-right
    ]

    colors = sv.ColorPalette.DEFAULT
    zones = [sv.PolygonZone(polygon=polygon) for polygon in polygons]
    zone_annotators = [
        sv.PolygonZoneAnnotator(
            zone=zone,
            color=colors.by_idx(index),
            thickness=3,
            text_thickness=4,
            text_scale=1.5
        ) for index, zone in enumerate(zones)
    ]
    box_annotators = [
        sv.BoxAnnotator(
            color=colors.by_idx(index),
            thickness=2
        ) for index in range(len(polygons))
    ]
    
    # Annotate frame
    for zone, zone_annotator, box_annotator in zip(zones, zone_annotators, box_annotators):
        mask = zone.trigger(detections=detections)
        detections_filtered = detections[mask]
        frame = box_annotator.annotate(scene=frame, detections=detections_filtered)
        frame = zone_annotator.annotate(scene=frame)

    return frame

st.title("YOLOv5 Video Processing")
st.write("Upload a video file and the model will process the video frame by frame.")

uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "mov"])

if uploaded_file is not None:
    model = load_model()
    
    video_path = "/tmp/uploaded_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_file.read())

    video_capture = cv2.VideoCapture(video_path)

    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_rate = video_capture.get(cv2.CAP_PROP_FPS)

    frame_placeholder = st.empty()

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        annotated_frame = process_frame(frame, model)

        pil_img = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB))
        frame_placeholder.image(pil_img, use_container_width=True)

    video_capture.release()
    st.write("Video processing completed.")
