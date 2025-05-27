import streamlit as st
import cv2
import numpy as np
import time
import plotly.graph_objects as go
from datetime import datetime
import pandas as pd
from pathlib import Path
import json
import mediapipe as mp
from typing import Dict, List, Tuple, Optional, Any
import logging
import threading
from queue import Queue
import warnings
import os
import pygame
from fpdf import FPDF
warnings.filterwarnings('ignore')
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections import deque
import http.server
import socketserver
import requests
from api_server import start_api_server

# Set page config first, before any other st commands
st.set_page_config(
    page_title="Advanced Proctoring System",
    page_icon="🎓",
    layout="wide"
)

# Initialize API server
if 'api_server_port' not in st.session_state:
    st.session_state.api_server_port = start_api_server(port=5000)
    st.session_state.test_live_data = {
        'tab_switches': 0,
        'fullscreen_exits': 0,
        'duration': 0,
        'test_completed': False
    }
    
    # Verify server is running
    try:
        response = requests.get(f"http://localhost:{st.session_state.api_server_port}/api/ping", timeout=2)
        if response.status_code == 200:
            st.success("API server started successfully")
        else:
            st.error(f"API server returned unexpected status: {response.status_code}")
    except Exception as e:
        st.error(f"Failed to connect to API server: {e}")

# Initialize MediaPipe solutions
mp_face = mp.solutions.face_detection
mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class AdvancedProctorSystem:
    def __init__(self):
        """Initialize the Advanced Proctor System with enhanced face detection"""
        # Initialize MediaPipe solutions
        self.mp_face = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize enhanced face detection
        self.face_detection = self.mp_face.FaceDetection(
            min_detection_confidence=0.7,
            model_selection=1  # Use full-range model for better detection
        )
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=10,  # Increased to detect up to 10 faces
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize other detectors
        self.pose = self.mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

        # Initialize tracking variables
        self.start_time = time.time()
        self.frame_width = 640  # Default width
        self.frame_height = 480  # Default height
        self.face_history = []
        self.face_history_size = 10
        self.face_confidence_threshold = 0.7
        self.looking_away_threshold = 30  # degrees
        self.face_landmarks_history = []
        self.multiple_faces_detected = False

    def setup_advanced_features(self):
        """Initialize advanced tracking features"""
        self.drowsiness_threshold = 0.3
        self.attention_threshold = 0.7
        self.pose_threshold = 0.8
        self.alert_cooldown = 5.0  # seconds
        self.last_alert_time = 0.0

    def setup_sound_alerts(self):
        """Initialize sound alerts"""
        pygame.mixer.init()
        self.alert_sound = None
        
        # Check multiple file formats and locations
        sound_files = ['alert.wav', 'alert.mp3']
        search_paths = [
            os.path.join(os.path.dirname(__file__), 'assets'),
            os.path.join('.', 'assets'),
            'assets'
        ]
        
        for path in search_paths:
            for sound_file in sound_files:
                file_path = os.path.join(path, sound_file)
                if os.path.exists(file_path):
                    try:
                        self.alert_sound = pygame.mixer.Sound(file_path)
                        return
                    except Exception as e:
                        st.warning(f"⚠️ Could not load sound file {file_path}: {e}")
        
        st.warning("⚠️ Alert sound file not found. Alerts will be silent.")

    def _calculate_posture_score(self, shoulders, hip):
        """Calculate posture score based on shoulder and hip positions"""
        # Calculate shoulder alignment
        shoulder_diff = abs(shoulders[0].y - shoulders[1].y)
        
        # Calculate vertical alignment (shoulders to hip)
        vertical_alignment = abs((shoulders[0].y + shoulders[1].y) / 2 - hip.y)
        
        # Normalize scores (lower is better)
        shoulder_score = 1.0 - min(shoulder_diff * 10, 1.0)
        vertical_score = 1.0 - min(vertical_alignment * 5, 1.0)
        
        # Combined score
        return (shoulder_score + vertical_score) / 2.0

    def _analyze_face_orientation(self, face_landmarks, frame_shape):
        """Analyze face orientation using facial landmarks"""
        if not face_landmarks:
            return None, None, None

        h, w = frame_shape[:2]
        
        # Get key facial landmarks
        nose = face_landmarks.landmark[1]  # Nose tip
        left_eye = face_landmarks.landmark[33]  # Left eye
        right_eye = face_landmarks.landmark[263]  # Right eye
        mouth_left = face_landmarks.landmark[61]  # Left mouth corner
        mouth_right = face_landmarks.landmark[291]  # Right mouth corner

        # Convert landmarks to pixel coordinates
        nose_pos = np.array([nose.x * w, nose.y * h])
        left_eye_pos = np.array([left_eye.x * w, left_eye.y * h])
        right_eye_pos = np.array([right_eye.x * w, right_eye.y * h])
        mouth_left_pos = np.array([mouth_left.x * w, mouth_left.y * h])
        mouth_right_pos = np.array([mouth_right.x * w, mouth_right.y * h])

        # Calculate face metrics
        eye_distance = np.linalg.norm(right_eye_pos - left_eye_pos)
        face_width = np.linalg.norm(mouth_right_pos - mouth_left_pos)
        
        # Calculate face orientation
        face_center = (left_eye_pos + right_eye_pos) / 2
        face_direction = nose_pos - face_center
        
        # Calculate angles
        yaw = np.arctan2(face_direction[0], eye_distance) * 180 / np.pi
        pitch = np.arctan2(face_direction[1], eye_distance) * 180 / np.pi

        return yaw, pitch, face_width

    def _calculate_drowsiness(self, face_landmarks):
        """Calculate drowsiness score based on eye aspect ratio"""
        if not face_landmarks:
            return 0.0

        # Eye landmarks indices
        LEFT_EYE = [33, 160, 158, 133, 153, 144]  # Left eye indices
        RIGHT_EYE = [362, 385, 387, 263, 373, 380]  # Right eye indices

        def eye_aspect_ratio(eye_points):
            # Convert landmarks to numpy array
            points = np.array([[p.x, p.y] for p in eye_points])
            
            # Calculate vertical distances
            v1 = np.linalg.norm(points[1] - points[5])
            v2 = np.linalg.norm(points[2] - points[4])
            
            # Calculate horizontal distance
            h = np.linalg.norm(points[0] - points[3])
            
            # Calculate eye aspect ratio
            ear = (v1 + v2) / (2.0 * h + 1e-6)
            return ear

        # Get eye landmarks
        left_eye_points = [face_landmarks.landmark[i] for i in LEFT_EYE]
        right_eye_points = [face_landmarks.landmark[i] for i in RIGHT_EYE]

        # Calculate eye aspect ratios
        left_ear = eye_aspect_ratio(left_eye_points)
        right_ear = eye_aspect_ratio(right_eye_points)

        # Average eye aspect ratio
        avg_ear = (left_ear + right_ear) / 2.0

        # Convert to drowsiness score (lower EAR = higher drowsiness)
        drowsiness_score = 1.0 - min(avg_ear * 6.0, 1.0)
        return drowsiness_score

    def _detect_face_state(self, frame):
        """Enhanced face detection and state analysis"""
        h, w = frame.shape[:2]
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with both face detection and face mesh
        face_results = self.face_detection.process(frame_rgb)
        mesh_results = self.face_mesh.process(frame_rgb)
        
        face_state = {
            'detected': False,
            'looking_away': False,
            'confidence': 0.0,
            'yaw': 0.0,
            'pitch': 0.0,
            'landmarks': None,
            'bbox': None,
            'drowsiness_score': 0.0
        }

        # Check face detection results
        if face_results.detections:
            detection = face_results.detections[0]
            face_state['detected'] = True
            face_state['confidence'] = detection.score[0]
            
            bbox = detection.location_data.relative_bounding_box
            face_state['bbox'] = {
                'x': int(bbox.xmin * w),
                'y': int(bbox.ymin * h),
                'w': int(bbox.width * w),
                'h': int(bbox.height * h)
            }

        # Check face mesh results for detailed analysis
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0]
            face_state['landmarks'] = landmarks
            
            # Calculate drowsiness score
            face_state['drowsiness_score'] = self._calculate_drowsiness(landmarks)
            
            # Analyze face orientation
            yaw, pitch, face_width = self._analyze_face_orientation(landmarks, frame.shape)
            face_state['yaw'] = yaw
            face_state['pitch'] = pitch
            
            face_state['looking_away'] = (
                abs(yaw) > self.looking_away_threshold or 
                abs(pitch) > self.looking_away_threshold
            )

            self.face_landmarks_history.append(landmarks)
            if len(self.face_landmarks_history) > self.face_history_size:
                self.face_landmarks_history.pop(0)

        return face_state

    def _draw_face_annotations(self, frame, face_state):
        """Draw face detection annotations on frame without mesh overlays"""
        if frame is None or not face_state:
            return frame

        # Return the original frame without any annotations
        # Detection still happens in the background
        return frame.copy()

    def process_frame(self, frame):
        """Process frame and return metrics without visual annotations"""
        if frame is None:
            return frame, self._get_default_metrics()

        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe (detection happens here but we don't draw anything)
        face_results = self.face_detection.process(rgb_frame)
        mesh_results = self.face_mesh.process(rgb_frame)
        pose_results = self.pose.process(rgb_frame)
        hands_results = self.hands.process(rgb_frame)

        # Update metrics based on detections
        metrics = self._analyze_frame_data(
            face_results, 
            mesh_results, 
            pose_results, 
            hands_results
        )

        # Draw warning for multiple faces if detected
        if metrics['multiple_faces']:
            cv2.putText(
                frame,
                f"WARNING: {metrics['number_of_faces']} faces detected!",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        # Return original frame without annotations and the calculated metrics
        return frame, metrics

    def _process_pose_detection(self, pose_results, annotated_frame):
        """Process pose detection results"""
        self.mp_drawing.draw_landmarks(
            annotated_frame,
            pose_results.pose_landmarks,
            self.mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
            connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(245,66,230), thickness=2)
        )
        
        # Calculate posture metrics
        if pose_results.pose_landmarks:
            shoulders = [
                pose_results.pose_landmarks.landmark[11],  # Left shoulder
                pose_results.pose_landmarks.landmark[12]   # Right shoulder
            ]
            hip = pose_results.pose_landmarks.landmark[24]  # Hip point
            posture_score = self._calculate_posture_score(shoulders, hip)
            self.pose_scores.append(posture_score)
            if len(self.pose_scores) > self.pose_buffer_size:
                self.pose_scores.pop(0)

    def _process_hand_detection(self, hands_results, annotated_frame):
        """Process hand detection results"""
        for hand_landmarks in hands_results.multi_hand_landmarks:
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                hand_landmarks,
                self.mp_hands.HAND_CONNECTIONS
            )

    def _update_metrics(self, metrics):
        """Update metrics based on current detections"""
        # Calculate accuracy percentage based on recent detections
        recent_scores = self.pose_scores[-10:] if self.pose_scores else [1.0]
        accuracy_score = sum(recent_scores) / len(recent_scores) * 100
        metrics['focus_percentage'] = accuracy_score
        
        # Add event
        self.events.append({
            'timestamp': time.time(),
            'attention_score': accuracy_score / 100,
            'face_detected': metrics['face_detected'],
            'looking_away': metrics['looking_away']
        })

    def _analyze_frame_data(self, face_results, mesh_results, pose_results, hands_results):
        """Analyze detection results and return metrics"""
        metrics = {
            'face_detected': False,
            'looking_away': False,
            'phone_detected': False,
            'focus_percentage': 100.0,
            'drowsiness_score': 0.0,
            'emotion_state': 'Neutral',
            'detected_objects': [],
            'session_duration': time.time() - self.start_time,
            'number_of_faces': 0,
            'multiple_faces': False
        }

        # Count number of faces detected
        if face_results.detections:
            num_faces = len(face_results.detections)
            metrics['number_of_faces'] = num_faces
            metrics['face_detected'] = True
            metrics['multiple_faces'] = num_faces > 1

            if num_faces > 1:
                metrics['focus_percentage'] -= 50.0
                metrics['detected_objects'].append(f'Multiple Faces ({num_faces})')

            # Analyze primary face
            face_state = self._detect_face_state(face_results.detections[0])
            metrics['looking_away'] = face_state['looking_away']
            metrics['drowsiness_score'] = face_state.get('drowsiness_score', 0.0)

        # Analyze hand detection results for phone usage
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                # Check if hands are near face region
                if mesh_results.multi_face_landmarks:
                    wrist = hand_landmarks.landmark[0]  # Wrist landmark
                    face_landmarks = mesh_results.multi_face_landmarks[0].landmark
                    nose = face_landmarks[4]  # Nose landmark
                    
                    # If hand is near face, might indicate phone usage
                    if abs(wrist.y - nose.y) < 0.2:
                        metrics['phone_detected'] = True
                        metrics['detected_objects'].append('Phone')

        # Analyze pose for attention
        if pose_results.pose_landmarks:
            # Calculate attention based on head position
            nose = pose_results.pose_landmarks.landmark[0]
            left_eye = pose_results.pose_landmarks.landmark[2]
            right_eye = pose_results.pose_landmarks.landmark[5]
            
            # Simple head orientation check
            eye_distance = abs(left_eye.x - right_eye.x)
            if eye_distance < 0.05:  # Head turned too much
                metrics['looking_away'] = True
                metrics['focus_percentage'] -= 30.0

        # Update focus percentage based on various factors
        if metrics['looking_away']:
            metrics['focus_percentage'] -= 40.0
        if metrics['phone_detected']:
            metrics['focus_percentage'] -= 50.0
        if not metrics['face_detected']:
            metrics['focus_percentage'] = 0.0

        # Ensure focus percentage stays within 0-100 range
        metrics['focus_percentage'] = max(0.0, min(100.0, metrics['focus_percentage']))

        return metrics

    def _detect_face_state(self, detection):
        """Analyze face detection and return state information"""
        face_state = {
            'looking_away': False,
            'drowsiness_score': 0.0,
            'confidence': detection.score[0]
        }

        # Get face bounding box
        bbox = detection.location_data.relative_bounding_box
        face_state['bbox'] = {
            'x': int(bbox.xmin * self.frame_width),
            'y': int(bbox.ymin * self.frame_height),
            'w': int(bbox.width * self.frame_width),
            'h': int(bbox.height * self.frame_height)
        }

        # Simple looking away detection based on face box position
        if bbox.xmin < 0.1 or bbox.xmin > 0.9:
            face_state['looking_away'] = True

        return face_state

def start_test_server(port=8000):
    """Start a simple HTTP server to serve the test page"""
    # Create a simple HTTP server
    Handler = http.server.SimpleHTTPRequestHandler
    
    # Try to start the server, find an open port if the default is in use
    while True:
        try:
            httpd = socketserver.TCPServer(("", port), Handler)
            break
        except OSError:
            port += 1
            if port > 8100:  # Limit the search to avoid infinite loop
                st.error("Failed to find an available port for the test server")
                return None
    
    # Start the server in a separate thread
    server_thread = threading.Thread(target=httpd.serve_forever)
    server_thread.daemon = True  # So the thread will exit when the main program exits
    server_thread.start()
    
    return port

def main():
    # Initialize ALL session state variables first
    if 'cheating_detected' not in st.session_state:
        st.session_state.cheating_detected = False
    if 'cheating_time' not in st.session_state:
        st.session_state.cheating_time = None
    if 'running' not in st.session_state:
        st.session_state.running = False
    if 'recording' not in st.session_state:
        st.session_state.recording = False
    if 'frame_count' not in st.session_state:
        st.session_state.frame_count = 0
    if 'start_time' not in st.session_state:
        st.session_state.start_time = time.time()
    if 'proctor' not in st.session_state:
        st.session_state.proctor = AdvancedProctorSystem()
    if 'visibilityState' not in st.session_state:
        st.session_state.visibilityState = 'visible'
    if 'metrics_buffer' not in st.session_state:
        st.session_state.metrics_buffer = []
    if 'last_report_time' not in st.session_state:
        st.session_state.last_report_time = time.time()
    if 'test_completed' not in st.session_state:
        st.session_state.test_completed = False
    if 'test_summary' not in st.session_state:
        st.session_state.test_summary = None
    if 'chart_data' not in st.session_state:
        st.session_state.chart_data = pd.DataFrame()

    # Check URL parameters for test completion data
    query_params = st.experimental_get_query_params()
    if 'test_completed' in query_params and query_params['test_completed'][0] == 'true':
        st.session_state.test_completed = True
        st.session_state.test_summary = {
            'duration': float(query_params.get('duration', [0])[0]),
            'fullscreen_exits': int(query_params.get('fullscreen_exits', [0])[0]),
            'tab_switches': int(query_params.get('tab_switches', [0])[0])
        }
        # Clear the URL parameters
        st.experimental_set_query_params()

    st.title("🎓 Advanced Proctoring System")
    
    # Display permanent cheating warning if detected
    if st.session_state.cheating_detected:
        st.error("""
            🚨 CHEATING DETECTED! 
            Tab switching was detected at {}. 
            This incident has been recorded and will be reported.
            """.format(st.session_state.cheating_time.strftime("%H:%M:%S") 
                      if st.session_state.cheating_time else "Unknown time")
        )
        
        # Option to acknowledge and continue
        if st.button("I acknowledge my violation and want to continue"):
            st.session_state.cheating_detected = False
            st.session_state.running = False
            st.rerun()
            
        st.stop()  # Stop further execution until acknowledged

    st.markdown("Tracks **attention**, **distraction**, **phone usage**, and **posture** in real-time.")

    # Create layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        run = st.toggle("▶️ Start Proctoring", key='running')
        record = st.toggle("🔴 Record Session", key='recording')
        
        # Only show Start Test button if proctoring is running
        if st.session_state.running:
            # Add Start Test button that opens a new tab
            if st.button("📝 Start Test", type="primary"):
                # Ensure the test_page.html file exists
                html_path = Path("test_page.html")
                if not html_path.exists():
                    st.error("Test page file not found! Make sure test_page.html is in the same directory.")
                else:
                    # Start the server if not already running
                    if 'test_server_port' not in st.session_state:
                        port = start_test_server()
                        if port:
                            st.session_state.test_server_port = port
                        else:
                            st.error("Failed to start test server")
                            st.stop()
                    
                    # Create the test URL
                    test_url = f"http://localhost:{st.session_state.test_server_port}/test_page.html"
                    
                    # Also provide a direct link button as backup
                    st.link_button("Open Test Page", test_url)
                    
                    # Inform the user
                    st.success("Please click the link above to start your test.")

    # Add JavaScript to listen for test completion message
    st.markdown("""
    <script>
        // Listen for messages from the test window
        window.addEventListener('message', function(event) {
            // Check if it's a test completion message
            if (event.data && event.data.type === 'TEST_COMPLETED') {
                const testData = event.data.data;
                
                // Store test data in localStorage
                localStorage.setItem('testData', JSON.stringify(testData));
                
                // Reload the page to show the test results
                window.location.reload();
            }
        });
        
        // Check if we have test data in localStorage
        document.addEventListener('DOMContentLoaded', function() {
            const testData = localStorage.getItem('testData');
            if (testData) {
                const data = JSON.parse(testData);
                
                // Create a hidden form to submit the data
                const form = document.createElement('form');
                form.method = 'POST';
                form.style.display = 'none';
                
                // Add fields for each piece of data
                for (const key in data) {
                    if (data.hasOwnProperty(key)) {
                        const input = document.createElement('input');
                        input.type = 'hidden';
                        input.name = key;
                        input.value = typeof data[key] === 'object' ? 
                            JSON.stringify(data[key]) : data[key];
                        form.appendChild(input);
                    }
                }
                
                // Add the form to the document and submit it
                document.body.appendChild(form);
                form.submit();
                
                // Clear the localStorage
                localStorage.removeItem('testData');
            }
        });
    </script>
    """, unsafe_allow_html=True)

    # Initialize video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Unable to access camera")
        return

    video_col, metrics_col = st.columns([3, 1])
    
    with video_col:
        video_placeholder = st.empty()
        status_placeholder = st.empty()
        
    with metrics_col:
        metrics_placeholder = st.empty()
        advanced_metrics_placeholder = st.empty()
        
    # Add settings panel
    with st.expander("⚙️ Advanced Settings"):
        col1, col2 = st.columns(2)
        with col1:
            detection_threshold = st.slider(
                "Detection Threshold",
                0.0, 1.0, 0.5,
                help="Adjust the sensitivity of detection algorithms"
            )
            alert_frequency = st.select_slider(
                "Alert Frequency",
                options=["Low", "Medium", "High"],
                value="Medium"
            )
        with col2:
            enable_recording = st.checkbox("Enable Session Recording", value=True)
            enable_analytics = st.checkbox("Enable Advanced Analytics", value=True)
    
    # Add real-time alerts
    alert_placeholder = st.empty()
    
    # Focus chart
    chart_placeholder = st.empty()
    
    if st.session_state.running:
        try:
            video_writer = None
            if st.session_state.recording:
                output_dir = Path("recordings")
                output_dir.mkdir(exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video_writer = cv2.VideoWriter(
                    str(output_dir / f'session_{int(time.time())}.avi'),
                    fourcc, 30.0, (int(cap.get(3)), int(cap.get(4)))
                )
                
            last_update = time.time()
            fps_buffer = []
            
            while st.session_state.running:
                # Check for tab visibility
                if 'visibilityState' in st.session_state and st.session_state.visibilityState == 'hidden':
                    st.session_state.cheating_detected = True
                    st.session_state.cheating_time = datetime.now()
                    st.session_state.running = False
                    st.rerun()

                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read frame from camera")
                    break
                
                # Calculate FPS
                current_time = time.time()
                fps = 1 / (current_time - last_update)
                fps_buffer.append(fps)
                if len(fps_buffer) > 30:
                    fps_buffer.pop(0)
                avg_fps = sum(fps_buffer) / len(fps_buffer)
                last_update = current_time
                
                # Process frame
                frame, metrics = st.session_state.proctor.process_frame(frame)
                
                if st.session_state.recording and video_writer:
                    video_writer.write(frame)
                    
                # Update frame counter
                st.session_state.frame_count += 1
                
                # Add FPS overlay
                cv2.putText(
                    frame,
                    f"FPS: {avg_fps:.1f}",
                    (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    1
                )
                
                # Update video feed
                if st.session_state.frame_count % 2 == 0:
                    video_placeholder.image(
                        cv2.cvtColor(frame, cv2.COLOR_BGR2RGB),
                        channels="RGB",
                        use_column_width=True
                    )
                
                # Update metrics
                if st.session_state.frame_count % 5 == 0:
                    # Determine status icons and colors
                    face_status = '✅' if metrics['face_detected'] else '❌'
                    
                    # Only show distraction status if face is detected
                    if metrics['face_detected']:
                        distraction_status = '❌' if (
                            metrics['looking_away'] or 
                            metrics['phone_detected'] or 
                            metrics['focus_percentage'] < 50
                        ) else '✅'
                        distraction_text = "Distracted" if distraction_status == '❌' else "Attentive"
                    else:
                        distraction_status = '⚠️'
                        distraction_text = "No Face Detected"

                    # Update metrics display with new styling
                    metrics_placeholder.markdown(f"""
                        <div style='background-color: rgba(0,0,0,0.1); padding: 20px; border-radius: 10px;'>
                            <h3>📊 Monitoring Status</h3>
                            <table style='width: 100%;'>
                                <tr>
                                    <td style='padding: 10px;'><b>Face Detection:</b></td>
                                    <td>{face_status} {'Face Detected' if metrics['face_detected'] else 'No Face'}</td>
                                </tr>
                                <tr>
                                    <td style='padding: 10px;'><b>Number of Faces:</b></td>
                                    <td>{'❌' if metrics['multiple_faces'] else '✅'} {metrics['number_of_faces']} {'(Multiple faces detected!)' if metrics['multiple_faces'] else ''}</td>
                                </tr>
                                <tr>
                                    <td style='padding: 10px;'><b>Attention Status:</b></td>
                                    <td>{distraction_status} {distraction_text}</td>
                                </tr>
                                <tr>
                                    <td style='padding: 10px;'><b>Focus Level:</b></td>
                                    <td>{'🎯' if metrics['focus_percentage'] >= 80 else '⚠️'} {metrics['focus_percentage']:.1f}%</td>
                                </tr>
                                <tr>
                                    <td style='padding: 10px;'><b>Session Status:</b></td>
                                    <td>✅ Active - Keep this tab open!</td>
                                </tr>
                            </table>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Show detailed metrics in expandable section
                    with st.expander("📋 Report", expanded=False):
                        # Collect metrics for 10-second summary
                        st.session_state.metrics_buffer.append(metrics)
                        
                        # Generate summary every 10 seconds
                        current_time = time.time()
                        if current_time - st.session_state.last_report_time >= 10:
                            # Create 4 columns for tile layout
                            col1, col2, col3, col4 = st.columns(4)
                            
                            # Calculate summary metrics
                            phone_detected = any(m['phone_detected'] for m in st.session_state.metrics_buffer)
                            avg_faces = sum(m['number_of_faces'] for m in st.session_state.metrics_buffer) / len(st.session_state.metrics_buffer)
                            avg_drowsiness = sum(m['drowsiness_score'] for m in st.session_state.metrics_buffer) / len(st.session_state.metrics_buffer)
                            unique_objects = list(set([obj for m in st.session_state.metrics_buffer for obj in m['detected_objects']]))
                            
                            # Phone Detection Tile
                            with col1:
                                st.markdown("""
                                <div style='background-color: rgba(255,255,255,0.05); 
                                          padding: 10px; 
                                          border-radius: 5px; 
                                          text-align: center;
                                          height: 100px;'>
                                    <div style='font-size: 24px;'>📱</div>
                                    <div style='font-size: 12px; color: #888;'>Phone (10s)</div>
                                    <div style='font-size: 14px; margin-top: 5px;'>
                                        {status}
                                    </div>
                                </div>
                                """.format(
                                    status="📱 Detected" if phone_detected else "✅ Clear"
                                ), unsafe_allow_html=True)

                            # Face Count Tile
                            with col2:
                                st.markdown("""
                                <div style='background-color: rgba(255,255,255,0.05); 
                                          padding: 10px; 
                                          border-radius: 5px; 
                                          text-align: center;
                                          height: 100px;'>
                                    <div style='font-size: 24px;'>👤</div>
                                    <div style='font-size: 12px; color: #888;'>Avg Faces (10s)</div>
                                    <div style='font-size: 14px; margin-top: 5px;'>
                                        {count:.1f}
                                    </div>
                                </div>
                                """.format(
                                    count=avg_faces
                                ), unsafe_allow_html=True)

                            # Drowsiness Score Tile
                            with col3:
                                st.markdown("""
                                <div style='background-color: rgba(255,255,255,0.05); 
                                          padding: 10px; 
                                          border-radius: 5px; 
                                          text-align: center;
                                          height: 100px;'>
                                    <div style='font-size: 24px;'>😴</div>
                                    <div style='font-size: 12px; color: #888;'>Avg Drowsiness (10s)</div>
                                    <div style='font-size: 14px; margin-top: 5px;'>
                                        {score:.2f}
                                    </div>
                                </div>
                                """.format(
                                    score=avg_drowsiness
                                ), unsafe_allow_html=True)

                            # Status Summary Tile
                            with col4:
                                avg_focus = sum(m['focus_percentage'] for m in st.session_state.metrics_buffer) / len(st.session_state.metrics_buffer)
                                status_icon = "✅" if avg_focus >= 50 else "⚠️"
                                
                                st.markdown("""
                                <div style='background-color: rgba(255,255,255,0.05); 
                                          padding: 10px; 
                                          border-radius: 5px; 
                                          text-align: center;
                                          height: 100px;'>
                                    <div style='font-size: 24px;'>{icon}</div>
                                    <div style='font-size: 12px; color: #888;'>10s Summary</div>
                                    <div style='font-size: 14px; margin-top: 5px;'>
                                        Avg Focus: {focus:.1f}%
                                    </div>
                                </div>
                                """.format(
                                    icon=status_icon,
                                    focus=avg_focus
                                ), unsafe_allow_html=True)

                            # Second row
                            col5, col6, col7, col8 = st.columns(4)

                            # Objects Detected Tile
                            with col5:
                                st.markdown("""
                                <div style='background-color: rgba(255,255,255,0.05); 
                                          padding: 10px; 
                                          border-radius: 5px; 
                                          text-align: center;
                                          height: 100px;'>
                                    <div style='font-size: 24px;'>🔍</div>
                                    <div style='font-size: 12px; color: #888;'>Objects (10s)</div>
                                    <div style='font-size: 14px; margin-top: 5px;'>
                                        {objects}
                                    </div>
                                </div>
                                """.format(
                                    objects=', '.join(unique_objects[:2]) or 'None'
                                ), unsafe_allow_html=True)

                            # Tab Switches Tile
                            with col6:
                                # Initialize with live data if available
                                if 'test_live_data' not in st.session_state:
                                    st.session_state.test_live_data = {
                                        'tab_switches': 0,
                                        'fullscreen_exits': 0,
                                        'duration': 0
                                    }
                                
                                tab_switches = st.session_state.test_live_data.get('tab_switches', 0)
                                if 'test_completed' in st.session_state and st.session_state.test_completed:
                                    if 'test_summary' in st.session_state and 'tab_switches' in st.session_state.test_summary:
                                        tab_switches = st.session_state.test_summary['tab_switches']
                                
                                tab_status = "❌" if tab_switches > 0 else "✅"
                                
                                st.markdown(f"""
                                <div style='background-color: rgba(255,255,255,0.05); 
                                          padding: 10px; 
                                          border-radius: 5px; 
                                          text-align: center;
                                          height: 100px;'>
                                <div style='font-size: 24px;'>🔄</div>
                                <div style='font-size: 12px; color: #888;'>Tab Switches</div>
                                <div style='font-size: 14px; margin-top: 5px;' data-test-metric="tab-switches">
                                        {tab_switches} {tab_status}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            # Fullscreen Exits Tile
                            with col7:
                                fullscreen_exits = st.session_state.test_live_data.get('fullscreen_exits', 0)
                                if 'test_completed' in st.session_state and st.session_state.test_completed:
                                    if 'test_summary' in st.session_state and 'fullscreen_exits' in st.session_state.test_summary:
                                        fullscreen_exits = st.session_state.test_summary['fullscreen_exits']
                                
                                fullscreen_status = "❌" if fullscreen_exits > 0 else "✅"
                                
                                st.markdown(f"""
                                <div style='background-color: rgba(255,255,255,0.05); 
                                          padding: 10px; 
                                          border-radius: 5px; 
                                          text-align: center;
                                          height: 100px;'>
                                <div style='font-size: 24px;'>🖥️</div>
                                <div style='font-size: 12px; color: #888;'>Fullscreen Exits</div>
                                <div style='font-size: 14px; margin-top: 5px;' data-test-metric="fullscreen-exits">
                                        {fullscreen_exits} {fullscreen_status}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                            # Time Taken Tile
                            with col8:
                                duration = st.session_state.test_live_data.get('duration', 0)
                                if 'test_completed' in st.session_state and st.session_state.test_completed:
                                    if 'test_summary' in st.session_state and 'duration' in st.session_state.test_summary:
                                        duration = st.session_state.test_summary['duration']
                                
                                minutes = int(duration // 60)
                                seconds = int(duration % 60)
                                time_display = f"{minutes}m {seconds}s"
                                
                                st.markdown(f"""
                                <div style='background-color: rgba(255,255,255,0.05); 
                                          padding: 10px; 
                                          border-radius: 5px; 
                                          text-align: center;
                                          height: 100px;'>
                                <div style='font-size: 24px;'>⏱️</div>
                                <div style='font-size: 12px; color: #888;'>Time Taken</div>
                                <div style='font-size: 14px; margin-top: 5px;' data-test-metric="duration">
                                        {time_display}
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)

                            # Reset buffer and update last report time
                            st.session_state.metrics_buffer = []
                            st.session_state.last_report_time = current_time

                    # Show appropriate alerts based on status
                    if metrics['multiple_faces']:
                        alert_placeholder.error("❌ Multiple faces detected! Only one person should be visible.", icon="⚠️")
                    elif metrics['face_detected']:
                        if metrics['looking_away'] or metrics['phone_detected'] or metrics['focus_percentage'] < 50:
                            alert_placeholder.warning("⚠️ Please maintain focus on the screen!", icon="⚠️")
                        else:
                            alert_placeholder.success("✅ Good! Maintaining attention.", icon="✅")
                    else:
                        alert_placeholder.error("❌ Face not detected! Please return to camera view.", icon="🎥")
                
                # Update focus chart
                if st.session_state.frame_count % 10 == 0:
                    # Create DataFrame for visualization
                    chart_data = pd.DataFrame({
                        'timestamp': [time.time() - st.session_state.start_time],
                        'attention': [metrics['focus_percentage'] / 100],
                        'drowsiness': [metrics['drowsiness_score']]
                    })
                    
                    if 'chart_data' not in st.session_state:
                        st.session_state.chart_data = chart_data
                    else:
                        st.session_state.chart_data = pd.concat([st.session_state.chart_data, chart_data])
                    
                    # Keep last 100 data points
                    if len(st.session_state.chart_data) > 100:
                        st.session_state.chart_data = st.session_state.chart_data.tail(100)
                    
                    fig = go.Figure()
                    
                    # Add attention trace
                    fig.add_trace(go.Scatter(
                        x=st.session_state.chart_data['timestamp'],
                        y=st.session_state.chart_data['attention'],
                        mode='lines',
                        name='Attention Level',
                        line=dict(color='rgb(76, 175, 80)', width=2)
                    ))
                    
                    # Add drowsiness trace
                    fig.add_trace(go.Scatter(
                        x=st.session_state.chart_data['timestamp'],
                        y=st.session_state.chart_data['drowsiness'],
                        mode='lines',
                        name='Drowsiness Level',
                        line=dict(color='rgb(255, 87, 34)', width=2)
                    ))
                    
                    fig.update_layout(
                        title='Attention & Drowsiness Timeline',
                        xaxis_title='Time (seconds)',
                        yaxis_title='Score',
                        height=300,
                        margin=dict(l=10, r=10, t=50, b=10),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='white'),
                        showlegend=True,
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                    
                    fig.update_yaxes(range=[0, 1])
                    
                    chart_placeholder.plotly_chart(
                        fig,
                        use_container_width=True,
                        key=f"attention_chart_{int(time.time())}"
                    )
                
                if st.session_state.frame_count % 10 == 0:  # Fetch every ~1/3 second at 30fps
                    # Placeholder for fetch_test_data function
                    def fetch_test_data():
                        """Fetch test data from the API server"""
                        try:
                            response = requests.get(f"http://localhost:{st.session_state.api_server_port}/api/test-data", timeout=1)
                            if response.status_code == 200:
                                data = response.json()
                                print(f"Received test data: {data}")  # Debug print
                                
                                # Update the live data
                                st.session_state.test_live_data = data
                                
                                # Check if test is completed
                                if data.get('test_completed', False) and not st.session_state.get('test_completed', False):
                                    st.session_state.test_completed = True
                                    st.session_state.test_summary = {
                                        'duration': data.get('duration', 0),
                                        'fullscreen_exits': data.get('fullscreen_exits', 0),
                                        'tab_switches': data.get('tab_switches', 0)
                                    }
                                    st.rerun()
                        except requests.exceptions.RequestException as e:
                            print(f"Error fetching test data: {e}")
                        except Exception as e:
                            print(f"Unexpected error in fetch_test_data: {e}")
                    
                    fetch_test_data()
                
                time.sleep(0.01)
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            logging.error(f"Runtime error: {str(e)}", exc_info=True)
            
        finally:
            if video_writer:
                video_writer.release()
            cap.release()
            
    else:
        st.info("Click 'Start' to begin proctoring")
        
        with st.expander("📖 Instructions", expanded=True):
            st.markdown("""
                ### How to use the Advanced Proctor
                1. Click the "Start" button to begin proctoring.
                2. Ensure your face is visible in the camera view.
                3. **DO NOT switch tabs or leave this window!**
                4. Switching tabs will be recorded as cheating.
                5. Maintain focus on the screen to achieve high focus levels.
                6. The system will alert you if it detects:
                   - Multiple faces
                   - Phone usage
                   - Looking away from screen
                8. The system will also provide real-time feedback on your focus level.
                9. The session must be recorded for review after completion of examination. 
            """)
    # Display test report if test was completed
    if st.session_state.test_completed and st.session_state.test_summary:
        with st.expander("📋 Test Report", expanded=True):
            st.markdown("### 📊 Test Completion Report")
            
            # Format duration
            duration = st.session_state.test_summary['duration']
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            
            # Create columns for metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Test Duration", f"{minutes}m {seconds}s")
            
            with col2:
                fullscreen_exits = st.session_state.test_summary['fullscreen_exits']
                status = "❌ Violation" if fullscreen_exits > 0 else "✅ Good"
                st.metric("Fullscreen Exits", f"{fullscreen_exits} ({status})")
            
            with col3:
                tab_switches = st.session_state.test_summary['tab_switches']
                status = "❌ Violation" if tab_switches > 0 else "✅ Good"
                st.metric("Tab Switches", f"{tab_switches} ({status})")
            
            # Overall assessment
            if fullscreen_exits > 0 or tab_switches > 0:
                st.warning("⚠️ Potential cheating detected during the test. Review the recording for verification.")
            else:
                st.success("✅ No suspicious activity detected during the test.")
if __name__ == "__main__":
    main()


