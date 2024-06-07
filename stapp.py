import av
import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Load the trained model
with open('face_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Counters for actions
yawn_counter = 0
sleeping_counter = 0
distracted_counter = 0

class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.yawn_counter = 0
        self.sleeping_counter = 0
        self.distracted_counter = 0

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")

        # Recolor image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(image_rgb)

        # Recolor back to BGR
        image_rgb.flags.writeable = True
        image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        # Draw face landmarks and make predictions
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                mp_drawing.draw_landmarks(image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)

                # Extract landmarks
                face_landmarks_list = []
                for landmark in face_landmarks.landmark:
                    face_landmarks_list.extend([landmark.x, landmark.y, landmark.z])

                # Make predictions
                X = pd.DataFrame([face_landmarks_list])
                face_action_class = model.predict(X)[0]
                face_action_prob = model.predict_proba(X)[0]

                # Display the results
                if face_action_class == 'Yawn' and max(face_action_prob) > 0.75:
                    cv2.putText(image, 'Yawning Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    self.yawn_counter += 1
                if face_action_class == 'Sleeping' and max(face_action_prob) > 0.65:
                    cv2.putText(image, 'Sleeping Detected', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    self.sleeping_counter += 1
                if face_action_class == 'Distracted' and max(face_action_prob) > 0.65:
                    cv2.putText(image, 'Distracted', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                    self.distracted_counter += 1
                if face_action_class == 'Neutral' and max(face_action_prob) > 0.85:
                    cv2.putText(image, 'Neutral', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

        return av.VideoFrame.from_ndarray(image, format="bgr24")

st.title("Sustav za nadzor vozača")

webrtc_ctx = webrtc_streamer(
    key="example",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=VideoProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

if webrtc_ctx.video_processor:
    if st.button("Stop"):
        webrtc_ctx.stop()

    st.text(f"Yawns: {webrtc_ctx.video_processor.yawn_counter}")
    st.text(f"Sleeping: {webrtc_ctx.video_processor.sleeping_counter}")
    st.text(f"Distracted: {webrtc_ctx.video_processor.distracted_counter}")
