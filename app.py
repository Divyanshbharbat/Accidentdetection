import streamlit as st
import cv2
import numpy as np
import os
import time
import base64
import tempfile
import threading
from sound_utils import generate_chime_wav
from accident_detector import AccidentDetector

# Page configuration
st.set_page_config(
    page_title="AI-Based Road Accident Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create folders for saved screenshots
ACCIDENTS_DIR = "accidents"
os.makedirs(ACCIDENTS_DIR, exist_ok=True)

# Generate alarm sound if not exists
SOUND_FILE = "notification.wav"
if not os.path.exists(SOUND_FILE):
    generate_chime_wav(SOUND_FILE)

# Custom premium styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .stApp {
        background-color: #0E1117;
        color: #FAFAFA;
    }
    
    /* Metrics Card Styles */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(255, 255, 255, 0.2);
    }
    
    .metric-value {
        font-size: 32px;
        font-weight: 800;
        margin-top: 6px;
    }
    
    .metric-label {
        font-size: 14px;
        color: #A0AEC0;
        text-transform: uppercase;
        letter-spacing: 1.5px;
    }
    
    /* Warning/Alert Banner */
    .alert-banner {
        background: linear-gradient(135deg, #FF4B4B 0%, #D80000 100%);
        color: white;
        padding: 18px;
        border-radius: 10px;
        font-weight: 600;
        font-size: 20px;
        text-align: center;
        margin-bottom: 20px;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 15px rgba(255, 75, 75, 0.35);
        animation: pulse 1.5s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.015); }
        100% { transform: scale(1); }
    }
    </style>
""", unsafe_allow_html=True)

# Helper function to play sound in the browser via HTML/JS autoplay
def play_sound_html(sound_path=SOUND_FILE):
    if os.path.exists(sound_path):
        with open(sound_path, "rb") as f:
            data = f.read()
        b64 = base64.b64encode(data).decode()
        audio_html = f"""
            <audio autoplay="true" style="display:none;">
                <source src="data:audio/wav;base64,{b64}" type="audio/wav">
            </audio>
        """
        # Inject HTML to trigger autoplay
        st.components.v1.html(audio_html, height=0, width=0)

# Initialize Session State variables
if "accident_logs" not in st.session_state:
    st.session_state.accident_logs = []
if "accident_count" not in st.session_state:
    st.session_state.accident_count = 0
if "playing_sound" not in st.session_state:
    st.session_state.playing_sound = False
if "training_logs" not in st.session_state:
    st.session_state.training_logs = "Not started yet."
if "is_training" not in st.session_state:
    st.session_state.is_training = False

# Sidebar Config
st.sidebar.markdown("<h2 style='text-align: center; color: #FF4B4B;'>⚙️ ADVANCED SETTINGS</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<p style='text-align: center; color: #888;'>Tune detection and collision sensitivity parameters here.</p>", unsafe_allow_html=True)
st.sidebar.markdown("---")

# 1. Advanced Parameters in Sidebar
conf_threshold = st.sidebar.slider("YOLOv8 Detection Confidence", 0.10, 0.90, 0.25, 0.05)
overlap_threshold = st.sidebar.slider("Collision Overlap (IoM Threshold)", 0.20, 0.80, 0.45, 0.05)
speed_drop_ratio = st.sidebar.slider("Sudden Speed Drop Ratio (%)", 10, 90, 60, 5) / 100.0
speed_threshold = st.sidebar.slider("Min Vehicle Speed before Collision (px/f)", 0.5, 10.0, 3.0, 0.5)
alert_cooldown = st.sidebar.slider("Alert Cooldown (Seconds)", 1, 30, 5, 1)

st.sidebar.markdown("---")
# Test Audio in Sidebar
if st.sidebar.button("Test Notification Chime 🔊", use_container_width=True):
    play_sound_html()
    st.sidebar.success("Playing chime! Verify your speaker volume.")

# Main Application Layout
st.markdown("<h1 style='text-align: center; color: #FFFFFF;'>🚨 AI-Based Road Accident Detection System</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888888; font-size: 16px;'>Real-time vehicle detection, tracking, collision physics analysis, and notification alert systems</p>", unsafe_allow_html=True)

# 2. Main Panel Controls: Model & Source Selection placed in columns for high visibility!
col_ctrl1, col_ctrl2 = st.columns(2)
with col_ctrl1:
    st.markdown("<h4 style='color: #FFBB00;'>🧠 Active AI Model</h4>", unsafe_allow_html=True)
    model_option = st.selectbox(
        "Choose YOLOv8 Model",
        ["yolov8n.pt (Default vehicle detector)", "best.pt (Custom trained classification)"],
        label_visibility="collapsed"
    )
    model_path = "yolov8n.pt" if "yolov8n.pt" in model_option else "best.pt"

with col_ctrl2:
    st.markdown("<h4 style='color: #FFBB00;'>📹 Input Source</h4>", unsafe_allow_html=True)
    source_option = st.selectbox(
        "Select Input Source",
        ["📁 Upload Image/Video File", "💻 Live Laptop Camera", "🌐 RTSP IP Camera Stream"],
        label_visibility="collapsed"
    )

# Metric layout banner
col_m1, col_m2, col_m3 = st.columns(3)
with col_m1:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">System Status</div>
            <div class="metric-value" style="color: #00FF66;">ACTIVE</div>
        </div>
    """, unsafe_allow_html=True)
with col_m2:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Accidents Logged</div>
            <div class="metric-value" style="color: #FF4B4B;">{st.session_state.accident_count}</div>
        </div>
    """, unsafe_allow_html=True)
with col_m3:
    st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Model Config</div>
            <div class="metric-value" style="color: #33B5E5;">{model_path}</div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Create two Main columns: Live View and Sidebar Logs
col_left, col_right = st.columns([2, 1])

def get_model_mtime(model_path):
    if os.path.exists(model_path):
        return os.path.getmtime(model_path)
    return 0

# Initialize the detector
@st.cache_resource
def get_detector(model_file, mtime, overlap, speed_drop, min_speed, cooldown):
    return AccidentDetector(
        model_path=model_file,
        overlap_threshold=overlap,
        speed_drop_ratio=speed_drop,
        speed_threshold=min_speed,
        alert_cooldown=cooldown
    )

# Instantiate / Update detector parameters
try:
    mtime = get_model_mtime(model_path)
    detector = get_detector(model_path, mtime, overlap_threshold, speed_drop_ratio, speed_threshold, alert_cooldown)
    # Update properties in case user changes sliders in sidebar
    detector.overlap_threshold = overlap_threshold
    detector.speed_drop_ratio = speed_drop_ratio
    detector.speed_threshold = speed_threshold
    detector.alert_cooldown = alert_cooldown
except Exception as e:
    st.error(f"Failed to load YOLOv8 model from path: '{model_path}'. Check if the file path is correct or wait for it to finish training. Error: {e}")
    st.stop()

# Right panel placeholder for real-time scrolling logs
with col_right:
    st.markdown("<h3 style='color: #FF4B4B;'>🔔 Real-Time Alert Log</h3>", unsafe_allow_html=True)
    log_container = st.empty()
    
    # Render logs
    def refresh_log_ui():
        if not st.session_state.accident_logs:
            log_container.info("No accidents detected yet. Monitor active...")
        else:
            log_html = "<div style='max-height: 400px; overflow-y: auto; padding: 10px; background-color: rgba(255,255,255,0.02); border-radius: 8px; border: 1px solid rgba(255,255,255,0.05);'>"
            for log in reversed(st.session_state.accident_logs):
                log_html += f"""
                    <div style='border-left: 4px solid #FF4B4B; padding-left: 10px; margin-bottom: 12px;'>
                        <strong style='color: #FF4B4B;'>🚨 Accident #{log['id']}</strong> ({log['timestamp']})<br/>
                        <span style='font-size: 14px; color: #DDD;'>Target: {log['vehicles']}</span><br/>
                        <span style='font-size: 12px; color: #888;'>Reason: {log['trigger']}</span>
                    </div>
                """
            log_html += "</div>"
            log_container.markdown(log_html, unsafe_allow_html=True)

    refresh_log_ui()

# Left panel for media loading & running inference loop
with col_left:
    st.markdown("<h3>📹 Live CCTV Monitor</h3>", unsafe_allow_html=True)
    video_feed = st.empty()
    
    if source_option == "📁 Upload Image/Video File":
        uploaded_file = st.file_uploader("Upload Image or Video File", type=["mp4", "avi", "jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            file_name = uploaded_file.name.lower()
            is_image = file_name.endswith((".jpg", ".jpeg", ".png"))
            
            # Action button
            submit_btn = st.button("🚀 Submit & Process File", use_container_width=True)
            
            if submit_btn:
                detector.clear()
                st.session_state.accident_logs = []
                st.session_state.accident_count = 0
                refresh_log_ui()
                
                if is_image:
                    # 1. PROCESS STATIC IMAGE
                    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                    
                    # Check task type
                    if getattr(detector.model, "task", "detect") == "classify":
                        annotated_img, accident_detected, details = detector.track_and_detect(img, conf_threshold=conf_threshold)
                        
                        if accident_detected and details:
                            st.session_state.accident_count = detector.accident_count
                            st.session_state.accident_logs.append(details)
                            
                            img_filename = f"accident_img_cls_{detector.accident_count}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                            img_path = os.path.join(ACCIDENTS_DIR, img_filename)
                            cv2.imwrite(img_path, img)
                            details["screenshot"] = img_path
                    else:
                        # Bounding box detection mode for static images
                        orig_speed_threshold = detector.speed_threshold
                        detector.speed_threshold = -1.0  # bypass speed check
                        
                        results = detector.model(img, conf=conf_threshold)
                        annotated_img = img.copy()
                        accident_detected = False
                        details = None
                        
                        if results and results[0].boxes:
                            boxes = results[0].boxes
                            xyxy_list = boxes.xyxy.cpu().numpy()
                            cls_list = boxes.cls.cpu().numpy().astype(int)
                            
                            vehicles = []
                            for i, box in enumerate(xyxy_list):
                                cls_id = cls_list[i]
                                if cls_id in detector.vehicle_classes:
                                    vehicles.append(box)
                                    
                            for idxA in range(len(vehicles)):
                                for idxB in range(idxA + 1, len(vehicles)):
                                    boxA = vehicles[idxA]
                                    boxB = vehicles[idxB]
                                    
                                    overlap = detector.calculate_overlap(boxA, boxB)
                                    if overlap > detector.overlap_threshold:
                                        accident_detected = True
                                        detector.accident_count += 1
                                        details = {
                                            "id": detector.accident_count,
                                            "timestamp": time.strftime("%H:%M:%S"),
                                            "vehicles": "Vehicle A and Vehicle B",
                                            "location": "Static Image",
                                            "trigger": "Static image bounding box overlap (IoM)"
                                        }
                                        st.session_state.accident_count = detector.accident_count
                                        st.session_state.accident_logs.append(details)
                                        
                                        cv2.rectangle(annotated_img, (int(boxA[0]), int(boxA[1])), (int(boxA[2]), int(boxA[3])), (0, 0, 255), 3)
                                        cv2.rectangle(annotated_img, (int(boxB[0]), int(boxB[1])), (int(boxB[2]), int(boxB[3])), (0, 0, 255), 3)
                                        
                                        img_filename = f"accident_img_{detector.accident_count}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                                        img_path = os.path.join(ACCIDENTS_DIR, img_filename)
                                        cv2.imwrite(img_path, img)
                                        details["screenshot"] = img_path
                                        break
                                        
                            if not accident_detected:
                                for box in vehicles:
                                    cv2.rectangle(annotated_img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 255, 0), 2)
                                    
                        detector.speed_threshold = orig_speed_threshold
                        
                    # Show results
                    video_feed.image(annotated_img, channels="BGR", use_container_width=True)
                    
                    if accident_detected and details:
                        st.markdown("""
                            <div class="alert-banner">
                                ⚠️ ACCIDENT DETECTED IN IMAGE!
                            </div>
                        """, unsafe_allow_html=True)
                        play_sound_html()
                        refresh_log_ui()
                    else:
                        st.success("No accidents detected in image.")
                        
                else:
                    # 2. PROCESS VIDEO FILE
                    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    tfile.write(uploaded_file.read())
                    video_path = tfile.name
                    
                    cap = cv2.VideoCapture(video_path)
                    
                    if not cap.isOpened():
                        st.error("Error opening video file.")
                    else:
                        st.toast("Processing video stream...")
                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break
                            
                            annotated_frame, triggered, details = detector.track_and_detect(frame, conf_threshold=conf_threshold)
                            
                            if triggered and details:
                                st.session_state.accident_count = detector.accident_count
                                st.session_state.accident_logs.append(details)
                                
                                img_filename = f"accident_{details['id']}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                                img_path = os.path.join(ACCIDENTS_DIR, img_filename)
                                cv2.imwrite(img_path, frame)
                                details["screenshot"] = img_path
                                
                                play_sound_html()
                                st.toast(f"🚨 Accident detected: {details['vehicles']}!", icon="🚨")
                                refresh_log_ui()
                                
                            display_frame = cv2.resize(annotated_frame, (854, 480))
                            video_feed.image(display_frame, channels="BGR", use_container_width=True)
                            time.sleep(0.01)
                            
                        cap.release()
                        st.success("Video processing complete!")
                        
    elif source_option == "💻 Live Laptop Camera":
        webcam_idx = st.number_input("Laptop Camera Index (Integrated webcam is usually 0)", min_value=0, max_value=5, value=0, step=1)
        
        col_wb1, col_wb2 = st.columns(2)
        start_wb = col_wb1.button("▶️ Start Laptop Camera", use_container_width=True)
        stop_wb = col_wb2.button("⏹️ Stop Camera", use_container_width=True)
        
        if start_wb:
            cap = cv2.VideoCapture(webcam_idx)
            detector.clear()
            st.session_state.accident_logs = []
            st.session_state.accident_count = 0
            refresh_log_ui()
            
            if not cap.isOpened():
                st.error(f"Laptop Camera (Index {webcam_idx}) could not be opened.")
            else:
                st.toast("Accessing laptop camera...")
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    annotated_frame, triggered, details = detector.track_and_detect(frame, conf_threshold=conf_threshold)
                    
                    if triggered and details:
                        st.session_state.accident_count = detector.accident_count
                        st.session_state.accident_logs.append(details)
                        
                        img_filename = f"camera_accident_{details['id']}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                        img_path = os.path.join(ACCIDENTS_DIR, img_filename)
                        cv2.imwrite(img_path, frame)
                        details["screenshot"] = img_path
                        
                        play_sound_html()
                        st.toast("🚨 ACCIDENT DETECTED!", icon="🚨")
                        refresh_log_ui()
                        
                    display_frame = cv2.resize(annotated_frame, (854, 480))
                    video_feed.image(display_frame, channels="BGR", use_container_width=True)
                    time.sleep(0.01)
                    
                cap.release()
                
    elif source_option == "🌐 RTSP IP Camera Stream":
        rtsp_url = st.text_input("RTSP Stream URL (e.g., rtsp://username:password@ip_address:554/stream1)", value="rtsp://")
        
        col_rt1, col_rt2 = st.columns(2)
        start_rt = col_rt1.button("▶️ Connect to Camera", use_container_width=True)
        stop_rt = col_rt2.button("⏹️ Disconnect", use_container_width=True)
        
        if start_rt and rtsp_url != "rtsp://":
            cap = cv2.VideoCapture(rtsp_url)
            detector.clear()
            st.session_state.accident_logs = []
            st.session_state.accident_count = 0
            refresh_log_ui()
            
            if not cap.isOpened():
                st.error("Could not connect to RTSP Camera Stream. Please check URL and network connectivity.")
            else:
                st.toast("Connecting to IP camera...")
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        st.warning("Video buffer empty or RTSP stream disconnected.")
                        break
                    
                    annotated_frame, triggered, details = detector.track_and_detect(frame, conf_threshold=conf_threshold)
                    
                    if triggered and details:
                        st.session_state.accident_count = detector.accident_count
                        st.session_state.accident_logs.append(details)
                        
                        img_filename = f"rtsp_accident_{details['id']}_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                        img_path = os.path.join(ACCIDENTS_DIR, img_filename)
                        cv2.imwrite(img_path, frame)
                        details["screenshot"] = img_path
                        
                        play_sound_html()
                        st.toast("🚨 ACCIDENT DETECTED!", icon="🚨")
                        refresh_log_ui()
                        
                    display_frame = cv2.resize(annotated_frame, (854, 480))
                    video_feed.image(display_frame, channels="BGR", use_container_width=True)
                    time.sleep(0.01)
                    
                cap.release()

# Bottom Tabs: Evidence Center and Custom Model Trainer
st.markdown("---")
tab_gallery, tab_training = st.tabs(["🖼️ Evidence Gallery", "⚙️ Custom Model Training"])

with tab_gallery:
    st.subheader("📁 Accident Evidence Files")
    
    screenshot_files = [f for f in os.listdir(ACCIDENTS_DIR) if f.endswith((".jpg", ".png"))]
    
    if not screenshot_files:
        st.info("No saved accident screenshots in the evidence directory.")
    else:
        cols = st.columns(4)
        for idx, filename in enumerate(screenshot_files):
            file_path = os.path.join(ACCIDENTS_DIR, filename)
            col = cols[idx % 4]
            with col:
                image = cv2.imread(file_path)
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                col.image(image_rgb, caption=filename, use_container_width=True)
                
                with open(file_path, "rb") as file:
                    col.download_button(
                        label=f"💾 Download Image",
                        data=file,
                        file_name=filename,
                        mime="image/jpeg",
                        key=f"dl_{filename}_{idx}"
                    )

with tab_training:
    st.subheader("🏋️ Train Custom Classification Model")
    st.write("Since you provided a classification dataset (with `Accident` and `Non Accident` subfolders) at:")
    st.code("C:\\Users\\bharb\\Downloads\\archive (1)\\data")
    st.write("You can train a YOLOv8 classification model using the form below. Training will run in a separate background thread so it won't crash your Streamlit interface.")
    
    dataset_path = st.text_input("Dataset Directory Path", value="C:\\Users\\bharb\\Downloads\\archive (1)\\data")
    epochs = st.number_input("Number of Epochs", min_value=1, max_value=100, value=5, step=1)
    batch_size = st.selectbox("Batch Size", [8, 16, 32, 64], index=1)
    
    def run_training_thread(path, eps, batch):
        st.session_state.is_training = True
        st.session_state.training_logs = "Starting YOLOv8 classification training..."
        
        try:
            from ultralytics import YOLO
            model = YOLO("yolov8n-cls.pt")
            
            st.session_state.training_logs += f"\nLoading pretrained yolov8n-cls.pt...\nTraining path: {path}\nEpochs: {eps}, Batch Size: {batch}\nTraining started..."
            
            results = model.train(data=path, epochs=eps, batch=batch, imgsz=224, workers=0, verbose=True)
            
            st.session_state.training_logs += f"\n\n🎉 Training Completed Successfully!"
            st.session_state.training_logs += f"\nSaved weights location: runs/classify/train/weights/best.pt"
            
            best_weights = "runs/classify/train/weights/best.pt"
            if os.path.exists(best_weights):
                import shutil
                shutil.copy(best_weights, "best.pt")
                st.session_state.training_logs += "\nSuccessfully copied trained 'best.pt' to project root directory!"
                
        except Exception as e:
            st.session_state.training_logs += f"\n\n❌ Error during training: {e}"
        finally:
            st.session_state.is_training = False
            
    if st.session_state.is_training:
        st.warning("Training is currently in progress. Please do not close the browser.")
        st.spinner("Training model in background...")
    else:
        if st.button("🚀 Start YOLOv8 Classification Training"):
            if not os.path.exists(dataset_path):
                st.error("Specified dataset path does not exist. Please double-check.")
            else:
                t = threading.Thread(target=run_training_thread, args=(dataset_path, epochs, batch_size))
                t.daemon = True
                t.start()
                st.success("Background training thread started!")
                
    st.markdown("### Training Logs:")
    st.code(st.session_state.training_logs, language="text")
