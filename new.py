import streamlit as st
import cv2
import numpy as np
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import tempfile
import os
from sgp4.api import Satrec, jday
import pandas as pd
from datetime import datetime, timedelta
from datetime import time as dt_time
import time
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import pytz
import time
import plotly.graph_objects as go

# Load YOLOv8 model
model = YOLO('best.pt')

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.capture_image = False

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img, _ = detect_objects(img)
        if self.capture_image:
            self.captured_image = img.copy()
            self.capture_image = False
        return img

    def get_captured_image(self):
        return getattr(self, 'captured_image', None)

def get_image_download_link(img, filename, text):
    buffered = BytesIO()
    img.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}">{text}</a>'
    return href

def detect_objects(image):
    results = model(image)
    for result in results:
        for bbox in result.boxes:
            x1, y1, x2, y2 = map(int, bbox.xyxy[0])
            cv2.rectangle(image, (x1, y1), (x2, y2), (95, 207, 30), 3)
            label = f"{result.names[int(bbox.cls[0])]}: {bbox.conf[0]:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    return image, results

def run_object_detection():
    st.sidebar.image('img_000002.jpg', use_column_width=True)
    st.sidebar.title("CubeSat Detection")
    st.sidebar.markdown("### Choose Input Source")
    activities = ["Image", "Webcam", "Video"]
    choice = st.sidebar.selectbox("Choose among the given options:", activities)

    st.markdown("<h1 style='text-align: center; color: #FF5733;'>Cubesat 1U & 3U & 6U Detection using YOLOv8</h1>", unsafe_allow_html=True)
 
    if choice == 'Image':
        st.markdown("### Upload Images for Detection")
        img_files = st.file_uploader("Choose Images", type=['jpg', 'jpeg', 'jfif', 'png'], accept_multiple_files=True)
        
        if img_files:
            for img_file in img_files:
                img = np.array(Image.open(img_file))
                original_img = img.copy()
                processed_img, results = detect_objects(img)

                col1, col2 = st.columns(2)
                with col1:
                    st.image(original_img, caption='Original Image', use_column_width=True)
                with col2:
                    st.image(processed_img, caption='Processed Image', use_column_width=True)

                with st.expander("Zoom Processed Image"):
                    st.image(processed_img, caption='Zoomed Processed Image', use_column_width=True)

                result_image = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
                st.markdown(get_image_download_link(result_image, img_file.name, 'Download Image'), unsafe_allow_html=True)
                for result in results:
                    for bbox in result.boxes:
                        st.markdown(f"<p style='color: #4CAF50;'>Class: {result.names[int(bbox.cls[0])]}, Confidence: {bbox.conf[0]:.2f}</p>", unsafe_allow_html=True)

    elif choice == 'Webcam':
        st.markdown("### Real-time Object Detection with Webcam")
        ctx = webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)

        if st.button("Capture Image"):
            ctx.video_transformer.capture_image = True

        if ctx.video_transformer and ctx.video_transformer.get_captured_image() is not None:
            captured_image = ctx.video_transformer.get_captured_image()
            original_img = captured_image.copy()
            processed_img, results = detect_objects(captured_image)

            col1, col2 = st.columns(2)
            with col1:
                st.image(original_img, caption='Captured Image', use_column_width=True)
            with col2:
                st.image(processed_img, caption='Processed Image', use_column_width=True)

            with st.expander("Zoom Processed Image"):
                st.image(processed_img, caption='Zoomed Processed Image', use_column_width=True)

            result_image = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
            st.markdown(get_image_download_link(result_image, 'captured_image.jpg', 'Download Image'), unsafe_allow_html=True)
            for result in results:
                for bbox in result.boxes:
                    st.markdown(f"<p style='color: #4CAF50;'>Class: {result.names[int(bbox.cls[0])]}, Confidence: {bbox.conf[0]:.2f}</p>", unsafe_allow_html=True)

    elif choice == 'Video':
        st.markdown("### Upload a Video for Detection")
        video_file = st.file_uploader("Choose a Video", type=['mp4', 'mov', 'avi', 'mkv'])
        
        if video_file:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(video_file.read())
            cap = cv2.VideoCapture(tfile.name)

            stframe = st.empty()
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                frame, results = detect_objects(frame)
                stframe.image(frame, channels='BGR')
            cap.release()
            os.remove(tfile.name)

    if st.button("Back"):
        st.session_state.option = None
        st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### About")
    st.sidebar.info("This application uses YOLOv8 for object detection in images, webcam, and video files. Upload your files or use the webcam for real-time detection.")

def run_prediction_model():
    st.markdown("<h1 style='text-align: center; color: #FF5733;'>Prediction Model</h1>", unsafe_allow_html=True)

    # Function to calculate satellite positions and velocity
    def get_satellite_positions_and_velocity(satellite, start_time, end_time, delta_t):
        positions = []
        velocities = []
        times = []

        t = start_time
        while t <= end_time:
            jd, fr = jday(t.year, t.month, t.day, t.hour, t.minute, t.second + t.microsecond * 1e-6)
            e, r, v = satellite.sgp4(jd, fr)
            if e == 0:  # Check for no error in computation
                positions.append((r[0], r[1], r[2]))
                velocities.append((v[0], v[1], v[2]))
                times.append(t)
            t += delta_t

        return positions, velocities, times

    # رسم الخريطة
    def plot_ground_track(positions, times, current_time=None):
        lons = []
        lats = []

        for r in positions:
            x, y, z = r
            lon = np.degrees(np.arctan2(y, x))
            lat = np.degrees(np.arctan2(z, np.sqrt(x * x + y * y)))
            lons.append(lon)
            lats.append(lat)

        fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={'projection': ccrs.PlateCarree()})
        ax.stock_img()
        ax.coastlines()
        ax.gridlines(draw_labels=True)
        ax.plot(lons, lats, 'c', transform=ccrs.Geodetic(), markersize=1)

        # إضافة موقع القمر الصناعي الحالي
        if current_time:
            index = times.index(current_time)
            current_lon = lons[index]
            current_lat = lats[index]
            ax.plot(current_lon, current_lat, 'bo', markersize=8, transform=ccrs.Geodetic())
            plt.text(current_lon, current_lat, ' CubeSat1U(CZ-6A DEB)', fontsize=12, ha='left', color='cyan', transform=ccrs.Geodetic())

        st.pyplot(fig)

    # Streamlit interface
    st.title('Satellite Ground Track')
    st.sidebar.title('Settings')

    # Path to the TLE file
    tle_file_path = 'sat54277.txt'
    with open(tle_file_path, 'r') as file:
        tle_lines = file.readlines()
        tle_line1 = tle_lines[0].strip()
        tle_line2 = tle_lines[1].strip()

    # Time inputs
    start_date = st.sidebar.date_input('Start Date', datetime.utcnow().date())
    start_time = st.sidebar.time_input('Start Time', datetime.utcnow().time())
    end_date = st.sidebar.date_input('End Date', datetime.utcnow().date() + timedelta(days=1))
    end_time = st.sidebar.time_input('End Time', (datetime.utcnow() + timedelta(days=1)).time())

    # Convert time to datetime
    start_datetime = datetime.combine(start_date, start_time).replace(tzinfo=pytz.UTC)
    end_datetime = datetime.combine(end_date, end_time).replace(tzinfo=pytz.UTC)

    # Update interval in minutes
    delta_t_minutes = st.sidebar.slider('Delta Time (minutes)', 1, 60, 10)
    delta_t = timedelta(minutes=delta_t_minutes)

    # Create the satellite
    satellite = Satrec.twoline2rv(tle_line1, tle_line2)

    # Calculate positions and velocities
    positions, velocities, times = get_satellite_positions_and_velocity(satellite, start_datetime, end_datetime, delta_t)

    # Paths to the CSV files
    csv_file_path1 = 'C54277 future prediction using LSTM_magnitudes (1).csv'
    csv_file_path2 = 'tle-54277_54277_with_magnitudes (1).csv'

    # Read data from the files
    data1 = pd.read_csv(csv_file_path1)
    data2 = pd.read_csv(csv_file_path2)

    # Attempt to find the time column automatically
    def find_time_column(data):
        for col in data.columns:
            if 'time' in col.lower() or 'date' in col.lower():
                return col
        return None

    time_column1 = find_time_column(data1)
    time_column2 = find_time_column(data2)

    if time_column1 is None or time_column2 is None:
        st.error("Could not automatically detect time column in one or both CSV files. Please check the column names.")
    else:
        # Convert the time column to datetime
        data1[time_column1] = pd.to_datetime(data1[time_column1], errors='coerce')
        data2[time_column2] = pd.to_datetime(data2[time_column2], errors='coerce')

        # Localize the time values in the DataFrame to UTC
        data1[time_column1] = data1[time_column1].dt.tz_localize('UTC')
        data2[time_column2] = data2[time_column2].dt.tz_localize('UTC')

        # Filter data based on the selected time range
        mask1 = (data1[time_column1] >= start_datetime) & (data1[time_column1] <= end_datetime)
        mask2 = (data2[time_column2] >= start_datetime) & (data2[time_column2] <= end_datetime)

        data1_filtered = data1.loc[mask1]
        data2_filtered = data2.loc[mask2]

        # Options to display data
        st.sidebar.subheader("Data Display Options")
        show_data1 = st.sidebar.checkbox('Show Data from pred model data')
        show_data2 = st.sidebar.checkbox('Show Data from stk data')

        # Display data based on user options
        if show_data1:
            st.subheader("Filtered Data from pred model data")
            st.dataframe(data1_filtered)

        if show_data2:
            st.subheader("Filtered Data from stk data")
            st.dataframe(data2_filtered)

        # Display position and velocity data for each time
        data = {
            'Time': times,
            'Longitude': [np.degrees(np.arctan2(r[1], r[0])) for r in positions],
            'Latitude': [np.degrees(np.arctan2(r[2], np.sqrt(r[0] * r[0] + r[1] * r[1]))) for r in positions],
            'Velocity (km/s)': velocities
        }
        df = pd.DataFrame(data)

        # Convert velocity to printable format
        df['Velocity (km/s)'] = df['Velocity (km/s)'].apply(lambda v: f'({v[0]:.2f}, {v[1]:.2f}, {v[2]:.2f})')

        # Interactive table using st.experimental_data_editor
        st.subheader("Satellite Positions and Velocities")
        selected_rows = st.data_editor(df, use_container_width=True)

        # Create a container to update the plot
        plot_container = st.empty()

        # Loop to update the plot every second
        for current_time in times:
            with plot_container:
                plot_ground_track(positions, times, current_time)
                time.sleep(1)




def main():
    if "option" not in st.session_state:
        st.session_state.option = None

    if st.session_state.option is None:
        st.markdown("<h1 style='text-align: center; color: #FF5733;'>Choose Operation</h1>", unsafe_allow_html=True)
        options = ["Object Detection", "Prediction Model"]
        choice = st.selectbox("Select an option:", options, key="option_select")
        if st.button("Proceed"):
            st.session_state.option = choice
            st.experimental_rerun()

    if st.session_state.option == "Object Detection":
        run_object_detection()
    elif st.session_state.option == "Prediction Model":
        run_prediction_model()


if __name__ == "__main__":
    main()

