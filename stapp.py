import cv2
import mediapipe as mp
import numpy as np
import pickle
import pandas as pd
import time
import streamlit as st

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Load the trained model
with open('face_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Initialize video capture
cap = cv2.VideoCapture(0)
st.title("Sustav za nadzor vozača")
stop_button = st.button("Stop")
frame_placeholder = st.empty()

# Variables to calculate FPS
fps_list = []
start_time = time.time()

# Counters for actions
yawn_counter = 0
sleeping_counter = 0
distracted_counter = 0

# Setup Mediapipe instance
with mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            st.write("The video capture has ended")
            break

        frame_start_time = time.time()



        # Recolor image to RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        # Make detection
        results = face_mesh.process(image)

        # Recolor back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # Draw face landmarks
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

                # You can add conditions to recognize specific actions
                if face_action_class == 'Yawn' and max(face_action_prob) > 0.75:
                    cv2.putText(image, 'Yawning Detected', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    yawn_counter += 1
                if face_action_class == 'Sleeping' and max(face_action_prob) > 0.65:
                    cv2.putText(image, 'Sleeping Detected', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    sleeping_counter += 1
                if face_action_class == 'Distracted' and max(face_action_prob) > 0.65:
                    cv2.putText(image, 'Distracted', (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)
                    distracted_counter += 1
                if face_action_class == 'Neutral' and max(face_action_prob) > 0.85:
                    cv2.putText(image, 'Neutral', (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                                cv2.LINE_AA)

        # Calculate FPS
        frame_end_time = time.time()
        time_elapsed = frame_end_time - frame_start_time
        fps = 1 / time_elapsed
        fps_list.append(fps)

        # Display FPS on the frame
        cv2.putText(image, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Display action counters on the frame
        cv2.putText(image, f'Yawns: {yawn_counter}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Sleeping: {sleeping_counter}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f'Distracted: {distracted_counter}', (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)


        image_copy = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(image_copy, channels="RGB")

        if cv2.waitKey(10) & 0xFF == ord('q') or stop_button:
            break

    cap.release()
    cv2.destroyAllWindows()

# Calculate and print average FPS
average_fps = sum(fps_list) / len(fps_list)
print(f'Average FPS: {average_fps:.2f}')
