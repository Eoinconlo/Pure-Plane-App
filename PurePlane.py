import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import tempfile
from collections import deque

# --- UI CONFIGURATION ---
st.set_page_config(page_title="Pure Plane AI", page_icon="⛳", layout="wide")
st.title("⛳ Pure Plane: Pro Swing Lab")

# Sidebar for Input Selection
st.sidebar.header("Input Settings")
input_mode = st.sidebar.radio("Select Source:", ("Upload Video", "Live Camera"))
handedness = st.sidebar.radio("Handedness:", ("Right-Handed", "Left-Handed"))
is_righty = True if handedness == "Right-Handed" else False

# --- AI LOGIC SETUP ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7)

if is_righty:
    LEAD_ARM = [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST]
    TRAIL_LEG = [mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE]
    LEAD_HIP = mp_pose.PoseLandmark.LEFT_HIP
else:
    LEAD_ARM = [mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST]
    TRAIL_LEG = [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE]
    LEAD_HIP = mp_pose.PoseLandmark.RIGHT_HIP

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180 else angle

# --- INPUT HANDLING ---
video_data = None
if input_mode == "Upload Video":
    video_data = st.sidebar.file_uploader("Upload Swing Video", type=["mp4", "mov", "avi"])
else:
    video_data = st.camera_input("Record your swing")

# --- MAIN PROCESSING ---
if video_data:
    # Save video to temp file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_data.read())
    cap = cv2.VideoCapture(tfile.name)
    
    # Setup Display
    col1, col2 = st.columns([2, 1])
    st_frame = col1.empty()
    st_metrics = col2.empty()

    # App State
    swing_stage = "Address"
    path_pts = deque(maxlen=20)
    head_start_pos = None
    tempo_ratio = "N/A"
    bs_start = ds_start = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        h, w, _ = image.shape

        try:
            landmarks = results.pose_landmarks.landmark
            
            # Extract Biometric Points
            shldr = [landmarks[LEAD_ARM[0].value].x, landmarks[LEAD_ARM[0].value].y]
            elbow = [landmarks[LEAD_ARM[1].value].x, landmarks[LEAD_ARM[1].value].y]
            wrist = [landmarks[LEAD_ARM[2].value].x, landmarks[LEAD_ARM[2].value].y]
            hip   = [landmarks[LEAD_HIP.value].x, landmarks[LEAD_HIP.value].y]
            t_hip, t_knee, t_ankl = [landmarks[TRAIL_LEG[0].value].x, landmarks[TRAIL_LEG[0].value].y], \
                                    [landmarks[TRAIL_LEG[1].value].x, landmarks[TRAIL_LEG[1].value].y], \
                                    [landmarks[TRAIL_LEG[2].value].x, landmarks[TRAIL_LEG[2].value].y]
            nose = [landmarks[mp_pose.PoseLandmark.NOSE.value].x, landmarks[mp_pose.PoseLandmark.NOSE.value].y]

            # 1. Calculations
            arm_angle = calculate_angle(shldr, elbow, wrist)
            knee_angle = calculate_angle(t_hip, t_knee, t_ankl)
            if head_start_pos is None: head_start_pos = nose
            head_dist = np.linalg.norm(np.array(nose) - np.array(head_start_pos))

            # 2. Swing Sequencing
            curr_time = cap.get(cv2.CAP_PROP_POS_MSEC)
            if wrist[1] < hip[1] and swing_stage == "Address":
                swing_stage, bs_start = "Backswing", curr_time
            elif swing_stage == "Backswing" and wrist[1] > shldr[1]:
                swing_stage, ds_start = "Downswing", curr_time
            elif swing_stage == "Downswing" and wrist[1] > hip[1]:
                swing_stage = "Impact"
                ds_dur, bs_dur = (curr_time - ds_start), (ds_start - bs_start)
                if ds_dur > 0: tempo_ratio = f"{round(bs_dur/ds_dur, 1)}:1"

            # 3. Visuals (Drawing Tracer)
            wrist_px = (int(wrist[0]*w), int(wrist[1]*h))
            path_pts.appendleft(wrist_px)
            for i in range(1, len(path_pts)):
                cv2.line(image, path_pts[i-1], path_pts[i], (0, 255, 255), 3)
            
            mp.solutions.drawing_utils.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # 4. Metrics UI Update
            with st_metrics:
                st.subheader("Swing Metrics")
                st.write(f"**Stage:** {swing_stage}")
                st.write(f"**Tempo:** {tempo_ratio}")
                st.metric("Lead Arm Angle", f"{int(arm_angle)}°")
                st.metric("Trail Knee Flex", f"{int(knee_angle)}°")
                if head_dist > 0.05:
                    st.error("⚠️ Excessive Head Sway")
                else:
                    st.success("✅ Stable Head")

        except: pass

        st_frame.image(image, channels="RGB", use_container_width=True)

    cap.release()
else:
    st.info("👈 Use the sidebar to upload a video or use the camera.")
