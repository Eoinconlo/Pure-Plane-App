import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
from collections import deque

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Pure Plane AI", page_icon="⛳", layout="wide")
st.title("⛳ Pure Plane: Pro Swing Lab")
st.markdown("### Advanced Biometric Analysis")

# Sidebar for Settings
st.sidebar.header("User Settings")
handedness = st.sidebar.radio(
    "Select Handedness:", ("Right-Handed", "Left-Handed"))
is_righty = True if handedness == "Right-Handed" else False

uploaded_file = st.sidebar.file_uploader(
    "Upload Swing Video", type=["mp4", "mov", "avi"])

# --- AI LOGIC SETUP ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Mapping Landmarks based on Handedness
if is_righty:
    LEAD_ARM = [mp_pose.PoseLandmark.LEFT_SHOULDER,
                mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST]
    TRAIL_LEG = [mp_pose.PoseLandmark.RIGHT_HIP,
                 mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE]
    LEAD_HIP = mp_pose.PoseLandmark.LEFT_HIP
else:
    LEAD_ARM = [mp_pose.PoseLandmark.RIGHT_SHOULDER,
                mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST]
    TRAIL_LEG = [mp_pose.PoseLandmark.LEFT_HIP,
                 mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE]
    LEAD_HIP = mp_pose.PoseLandmark.RIGHT_HIP


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
        np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180 else angle


# --- APP STATE ---
swing_stage = "Address"
path_pts = deque(maxlen=20)
head_start_pos = None
tempo_ratio = "N/A"
bs_start = ds_start = 0

# --- MAIN PROCESSING ---
if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap = cv2.VideoCapture(tfile.name)

    # Layout Columns: Video on left, Metrics on right
    col1, col2 = st.columns([2, 1])
    st_frame = col1.empty()
    st_metrics = col2.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        h, w, _ = image.shape

        try:
            landmarks = results.pose_landmarks.landmark

            # Extract Points
            shldr = [landmarks[LEAD_ARM[0].value].x,
                     landmarks[LEAD_ARM[0].value].y]
            elbow = [landmarks[LEAD_ARM[1].value].x,
                     landmarks[LEAD_ARM[1].value].y]
            wrist = [landmarks[LEAD_ARM[2].value].x,
                     landmarks[LEAD_ARM[2].value].y]
            hip = [landmarks[LEAD_HIP.value].x, landmarks[LEAD_HIP.value].y]
            t_hip, t_knee, t_ankl = [landmarks[TRAIL_LEG[0].value].x, landmarks[TRAIL_LEG[0].value].y], \
                                    [landmarks[TRAIL_LEG[1].value].x, landmarks[TRAIL_LEG[1].value].y], \
                                    [landmarks[TRAIL_LEG[2].value].x,
                                        landmarks[TRAIL_LEG[2].value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x,
                    landmarks[mp_pose.PoseLandmark.NOSE.value].y]

            # 1. Math Metrics
            arm_angle = calculate_angle(shldr, elbow, wrist)
            knee_angle = calculate_angle(t_hip, t_knee, t_ankl)
            if head_start_pos is None:
                head_start_pos = nose
            head_dist = np.linalg.norm(
                np.array(nose) - np.array(head_start_pos))

            # 2. Swing Logic & Tempo
            curr_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            if wrist[1] < hip[1] and swing_stage == "Address":
                swing_stage, bs_start = "Backswing", curr_time
            elif swing_stage == "Backswing" and wrist[1] > shldr[1]:
                swing_stage, ds_start = "Downswing", curr_time
            elif swing_stage == "Downswing" and wrist[1] > hip[1]:
                swing_stage = "Impact"
                ds_dur, bs_dur = (curr_time - ds_start), (ds_start - bs_start)
                if ds_dur > 0:
                    tempo_ratio = f"{round(bs_dur/ds_dur, 1)}:1"

            # 3. Visuals (Tracer & Skeleton)
            wrist_px = (int(wrist[0]*w), int(wrist[1]*h))
            path_pts.appendleft(wrist_px)
            for i in range(1, len(path_pts)):
                cv2.line(image, path_pts[i-1], path_pts[i], (255, 255, 0), 3)

            mp.solutions.drawing_utils.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Update Metrics Panel
            with st_metrics:
                st.write(f"**Phase:** {swing_stage}")
                st.write(f"**Tempo:** {tempo_ratio}")
                st.metric("Elbow Angle", f"{int(arm_angle)}°")
                st.metric("Knee Flex", f"{int(knee_angle)}°")
                st.write("✅ Head Stable" if head_dist <
                         0.05 else "❌ Head Sway Detected")

        except:
            pass

        st_frame.image(image, channels="RGB", use_container_width=True)

    cap.release()
else:
    st.info("Please upload a video file in the sidebar to begin analysis.")
