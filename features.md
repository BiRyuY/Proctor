# 🎓 Advanced Proctoring System

The **Advanced Proctoring System** is an AI-powered monitoring tool built using **OpenCV**, **MediaPipe**, **Streamlit**, and **Plotly**. It provides real-time analysis of student behavior during online assessments or sessions.

---

## 📌 Features

### 📸 Real-Time Monitoring
- **Face Detection:** Detects if the user is present. Alerts when the face is not visible.
- **Posture Detection:** Identifies bad posture (e.g., leaning or slouching) using shoulder width analysis.
- **Phone Usage Detection:** Detects presence of hands near the face indicating potential phone usage.
- **Live Status Overlay:** The system displays the current monitoring status on the webcam feed.

### 🔊 Smart Alerts System
- **Sound Alerts:** Plays an alert beep when a distraction or suspicious activity is detected.
- **Distraction Screenshot Capture:** Saves frames as evidence when the user is distracted or phone is detected.
- **Webcam Offline Detector:** Notifies if the webcam feed fails or stops during the session.

### 📊 Graphical Analytics
- **Line Chart - Focus Timeline:** Displays real-time attention status over time using a clean and clear line plot.
- **Pie Chart - Attention Distribution:** Shows the percentage of time the user was attentive, distracted, using phone, or had posture issues.
- **Attention Score Bar:** Displays an overall attention score in percentage.
- **Live Timer:** Tracks the total duration of the proctoring session.

### 📁 Reports & Exports
- **🧾 Activity Log:** Displays the most recent monitoring logs (time and status).
- **⬇️ Export as CSV:** Download the session logs in CSV format.
- **📤 Export as PDF:** Generate a clean and downloadable PDF report containing the entire session log and results.

### 🌙 Clean UI/UX
- **Dark Theme UI:** Consistent theme across the app, with matching backgrounds for graphs and charts.
- **Responsive Layout:** Camera feed, graphs, and reports are all arranged cleanly and clearly below each other.

---

## 🛠️ How to Run

### 1. Install Dependencies
```bash
pip install opencv-python mediapipe streamlit plotly pygame fpdf
