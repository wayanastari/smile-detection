import streamlit as st
import tempfile
import os
import time
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from smile import detect_smiles
from streamlit_webrtc import webrtc_streamer 
from streamlit_webrtc.webrtc_utils import VideoTransformerBase

# Load model
model = tf.keras.models.load_model("model/smile-model.h5", compile=False)

st.title("SMILE DETECTION")
os.makedirs("output", exist_ok=True) # Ensure output directory exists

container = st.container()
st.caption("Smile — it's the simplest way to brighten the world.")
st.write("Selamat datang di Smile Detection, sebuah aplikasi cerdas yang bisa mengenali senyum secara otomatis dari foto, video, bahkan secara real-time lewat webcam!🥰")


# Pilihan mode
mode = st.radio("Pilih Mode:", ["Upload File", "Open Webcam"])

uploaded_file = None
video_file_path = None

if mode == "Upload File":
    uploaded_file = st.file_uploader("Upload Foto atau Video", type=["jpg", "jpeg", "png", "mp4", "avi"])
    if uploaded_file:
        file_type = uploaded_file.type
        if "image" in file_type:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            uploaded_image = cv2.imdecode(file_bytes, 1)
        elif "video" in file_type:
            # Create a temporary file for the video
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
                tfile.write(uploaded_file.read())
                video_file_path = tfile.name

# --- Webcam Processing Class (for streamlit-webrtc) ---
class SmileVideoProcessor(VideoTransformerBase):
    def __init__(self, model):
        self.model = model

    def transform(self, frame):
        # Convert the WebRTC frame (RGB) to OpenCV BGR format
        img = frame.to_ndarray(format="bgr24")
        processed_img = detect_smiles(img.copy(), self.model)
        # Return the processed image
        return processed_img
# --------------------------------------------------------

# Removed the "Proses" button for webcam, as webrtc_stream starts immediately.
# The "Proses" button is now only for "Upload File" mode.
if mode == "Upload File":
    if st.button("Proses File"): # Renamed button for clarity
        progress_text = "Sedang memproses. Mohon tunggu..."
        my_bar = st.progress(0, text=progress_text)

        for percent_complete in range(100):
            time.sleep(0.005)
            my_bar.progress(percent_complete + 1, text=progress_text)
        time.sleep(0.5)
        my_bar.empty()

        if uploaded_file is None:
            st.warning("Silakan upload file terlebih dahulu.")
        elif "image" in uploaded_file.type:
            result = detect_smiles(uploaded_image.copy(), model)
            output_path = "output/labeled_image.jpg"
            cv2.imwrite(output_path, result)
            st.image(result, channels="BGR", caption="Hasil Deteksi")
            with open(output_path, "rb") as f:
                st.download_button("Download Gambar", f, file_name="labeled_image.jpg")

        elif "video" in uploaded_file.type:
            stframe = st.empty()
            cap = cv2.VideoCapture(video_file_path)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            output_path = "output/labeled_video.avi"
            # It's good practice to ensure the directory exists before writing
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (width, height))

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                result = detect_smiles(frame.copy(), model)
                out.write(result)
                stframe.image(result, channels="BGR")

            cap.release()
            out.release()
            os.remove(video_file_path) # Clean up temporary video file
            st.success("Video selesai diproses.")
            with open(output_path, "rb") as f:
                st.download_button("Download Video", f, file_name="labeled_video.avi")

elif mode == "Open Webcam":
    st.info("Akses webcam Anda diizinkan melalui browser. Pastikan untuk memberikan izin jika diminta.")
    # Use streamlit-webrtc for live webcam feed
    webrtc_ctx = webrtc_stream(
        key="smile-detector-webcam",
        video_processor_factory=lambda: SmileVideoProcessor(model), # Pass the model to the processor
        media_stream_constraints={"video": True, "audio": False}, # Only video, no audio
        async_processing=True, # Process frames asynchronously
    )

    if webrtc_ctx.video_receiver:
        st.write("Mendeteksi senyum secara real-time...")
        # You could add a stop button here if needed for the stream,
        # but the webrtc_stream component usually has its own stop/start controls.
        # Note: Recording the output of webrtc_stream is more complex and typically
        # requires client-side recording or more advanced server-side handling.
        # The current download button for webcam output won't work directly here.
