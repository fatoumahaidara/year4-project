from pathlib import Path
from datetime import datetime
import pickle

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# Optional import so the app shows an error instead of a blank page if dlib/face_recognition
# is not installed correctly.
try:
    import face_recognition
    FACE_LIB_OK = True
    FACE_LIB_ERROR = ""
except Exception as e:
    face_recognition = None
    FACE_LIB_OK = False
    FACE_LIB_ERROR = str(e)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
ENCODINGS_FILE = BASE_DIR / "encodings.pickle"
CSV_FILE = BASE_DIR / "attendance.csv"

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Attendance System for Students",
    page_icon="🎓",
    layout="wide",
)

st.title("🎓 Smart Attendance System for Students")
st.caption("Upload a photo or use your webcam to mark attendance automatically.")

# ── Startup checks ───────────────────────────────────────────────────────────
if not FACE_LIB_OK:
    st.error("Face recognition library could not be loaded.")
    st.code(FACE_LIB_ERROR)
    st.info("This usually means dlib / face_recognition is not installed correctly in your environment.")
    st.stop()


# ── Load encodings ───────────────────────────────────────────────────────────
@st.cache_resource
def load_encodings():
    if not ENCODINGS_FILE.exists():
        return [], []

    try:
        with ENCODINGS_FILE.open("rb") as f:
            data = pickle.load(f)
        encodings = data.get("encodings", [])
        names = data.get("names", [])
        return encodings, names
    except Exception as e:
        st.sidebar.error(f"Could not read encodings.pickle: {e}")
        return [], []


known_encodings, known_names = load_encodings()
st.sidebar.success(f"✅ Loaded {len(known_names)} face encodings")


def save_encodings(encodings, names):
    data = {"encodings": encodings, "names": names}
    with ENCODINGS_FILE.open("wb") as f:
        pickle.dump(data, f)


# ── Attendance log ───────────────────────────────────────────────────────────
def load_attendance():
    if CSV_FILE.exists():
        try:
            return pd.read_csv(CSV_FILE)
        except Exception:
            pass
    return pd.DataFrame(columns=["Name", "Date", "Time", "Confidence"])



def log_attendance(name, confidence):
    df = load_attendance()
    today = datetime.now().strftime("%Y-%m-%d")
    now = datetime.now().strftime("%H:%M:%S")

    already = ((df["Name"] == name) & (df["Date"] == today)).any()
    if not already:
        new_row = pd.DataFrame([
            {
                "Name": name,
                "Date": today,
                "Time": now,
                "Confidence": f"{confidence:.1f}%",
            }
        ])
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)
        return True
    return False


# ── Face recognition ─────────────────────────────────────────────────────────
def recognise_faces(image_array):
    """Detect and recognise faces in an RGB numpy array."""
    locations = face_recognition.face_locations(image_array)
    encodings = face_recognition.face_encodings(image_array, locations)

    results = []
    annotated = image_array.copy()

    for enc, (top, right, bottom, left) in zip(encodings, locations):
        name = "Unknown"
        confidence = 0.0

        if known_encodings:
            matches = face_recognition.compare_faces(known_encodings, enc, tolerance=0.5)
            if True in matches:
                distances = face_recognition.face_distance(known_encodings, enc)
                best = int(np.argmin(distances))
                if matches[best]:
                    name = known_names[best]
                    confidence = (1 - distances[best]) * 100

        color = (0, 200, 0) if name != "Unknown" else (200, 0, 0)
        cv2.rectangle(annotated, (left, top), (right, bottom), color, 2)
        cv2.rectangle(annotated, (left, bottom - 30), (right, bottom), color, cv2.FILLED)
        cv2.putText(
            annotated,
            f"{name} {confidence:.0f}%",
            (left + 4, bottom - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
        )

        results.append({"name": name, "confidence": confidence})

        if name != "Unknown" and log_attendance(name, confidence):
            st.toast(f"✅ {name} marked present!", icon="🎓")

    return annotated, results


if not ENCODINGS_FILE.exists():
    st.warning("encodings.pickle was not found. Register students before recognising faces.")


tab1, tab2, tab3, tab4 = st.tabs([
    "📷 Upload Photo",
    "📹 Webcam",
    "📋 Attendance Log",
    "📝 Register Student",
])

# ── Tab 1 ────────────────────────────────────────────────────────────────────
with tab1:
    st.subheader("Upload a photo to identify faces")
    uploaded_file = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png"], key="identify_upload"
    )

    if uploaded_file:
        image_pil = Image.open(uploaded_file).convert("RGB")
        image_array = np.array(image_pil)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image_pil, caption="Original", use_container_width=True)

        with st.spinner("Recognising faces..."):
            annotated, results = recognise_faces(image_array)

        with col2:
            st.image(annotated, caption="Result", use_container_width=True)

        st.divider()
        if results:
            st.subheader(f"Found {len(results)} face(s):")
            for r in results:
                if r["name"] != "Unknown":
                    st.success(f"✅ **{r['name']}** — {r['confidence']:.1f}% confidence")
                else:
                    st.error("❌ Unknown person")
        else:
            st.warning("No faces detected in this image.")

# ── Tab 2 ────────────────────────────────────────────────────────────────────
with tab2:
    st.subheader("Use webcam to capture and identify")
    st.info("Allow camera access in your browser, then take a snapshot.")

    snap = st.camera_input("Take a photo", key="identify_camera")

    if snap:
        image_pil = Image.open(snap).convert("RGB")
        image_array = np.array(image_pil)

        with st.spinner("Recognising faces..."):
            annotated, results = recognise_faces(image_array)

        st.image(annotated, caption="Result", use_container_width=True)

        if results:
            for r in results:
                if r["name"] != "Unknown":
                    st.success(f"✅ **{r['name']}** — {r['confidence']:.1f}% confidence")
                else:
                    st.error("❌ Unknown person")
        else:
            st.warning("No faces detected.")

# ── Tab 3 ────────────────────────────────────────────────────────────────────
with tab3:
    st.subheader("Attendance Records")

    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("🔄 Refresh", key="refresh_attendance"):
            st.rerun()
        if st.button("🗑️ Clear Today", key="clear_today"):
            df = load_attendance()
            today = datetime.now().strftime("%Y-%m-%d")
            df = df[df["Date"] != today]
            df.to_csv(CSV_FILE, index=False)
            st.rerun()

    df = load_attendance()

    if df.empty:
        st.info("No attendance recorded yet. Upload a photo or use the webcam.")
    else:
        today = datetime.now().strftime("%Y-%m-%d")
        today_df = df[df["Date"] == today]

        m1, m2, m3 = st.columns(3)
        m1.metric("Present Today", len(today_df))
        m2.metric("Total Records", len(df))
        m3.metric("Unique People", df["Name"].nunique())

        st.divider()

        dates = sorted(df["Date"].unique(), reverse=True)
        selected_date = st.selectbox("Filter by date", ["All"] + list(dates), key="date_filter")

        if selected_date != "All":
            df = df[df["Date"] == selected_date]

        st.dataframe(df, use_container_width=True, hide_index=True)

        csv_data = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            data=csv_data,
            file_name=f"attendance_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            key="download_attendance",
        )

# ── Tab 4 ────────────────────────────────────────────────────────────────────
with tab4:
    st.subheader("Register New Student")

    name = st.text_input("Enter student name", key="student_name")
    option = st.radio("Choose input method:", ["Upload Image", "Use Webcam"], key="register_method")

    image = None

    if option == "Upload Image":
        uploaded = st.file_uploader(
            "Upload student photo", type=["jpg", "jpeg", "png"], key="register_upload"
        )
        if uploaded:
            image = Image.open(uploaded).convert("RGB")

    else:
        snap = st.camera_input("Take a photo", key="register_camera")
        if snap:
            image = Image.open(snap).convert("RGB")

    if image is not None:
        st.image(image, caption="Captured Image", use_container_width=True)

        if st.button("Register Student", key="register_student"):
            if name.strip() == "":
                st.warning("⚠️ Please enter a name")
            else:
                image_array = np.array(image)
                locations = face_recognition.face_locations(image_array)

                if len(locations) == 0:
                    st.error("❌ No face detected. Try again.")
                elif len(locations) > 1:
                    st.error("❌ Multiple faces detected. Use a single-person image.")
                else:
                    enc = face_recognition.face_encodings(image_array, locations)[0]
                    known_encodings.append(enc)
                    known_names.append(name)
                    save_encodings(known_encodings, known_names)
                    st.success(f"✅ {name} registered successfully!")
                    st.cache_resource.clear()
