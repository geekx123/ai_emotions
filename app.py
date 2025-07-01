import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image, UnidentifiedImageError
import random

# --- Configuration ---
MODEL_PATH = "emotion_model.h5"
HAARCASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define filters. Each emotion can have a list of transparent PNG filter paths.
FILTERS = {
    'Happy': [
        'filters/sunglasses.png',
        'filters/sunglasses_cool.png',
        'filters/happy_crown.png'
    ],
    'Angry': [
        'filters/devil_horn.png',
        'filters/angry_mask.png',
        'filters/red_eyes.png'
    ],
    'Sad': [
        'filters/tears.png',
        'filters/sad_cloud.png',
        'filters/rain_effect.png'
    ],
    'Surprise': [
        'filters/shock_effect.png',
        'filters/exclamation_mark.png'
    ],
    'Neutral': [
        'filters/monocle.png',
        'filters/glasses.png'
    ],
    'Fear': [
        'filters/ghost_overlay.png',
        'filters/scared_eyes.png'
    ],
    'Disgust': [
        'filters/sick_face.png',
        'filters/green_slime.png'
    ],
}

# --- Asset Loading ---
@st.cache_resource
def get_emotion_model(model_path):
    """Loads the pre-trained Keras emotion detection model."""
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading emotion model from **{model_path}**. Ensure the file exists and is a valid Keras model. Details: {e}")
        st.stop()

@st.cache_resource
def get_face_cascade(cascade_path):
    """Loads the OpenCV Haar Cascade classifier for face detection."""
    try:
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            raise IOError(f"Haar Cascade XML file not loaded. Check the path or OpenCV installation.")
        return face_cascade
    except Exception as e:
        st.error(f"Error loading face detection cascade from **{cascade_path}**. Details: {e}")
        st.stop()

def load_filter_image(filter_path):
    """Loads a single filter image and returns its RGB and Alpha channels."""
    try:
        filter_img_pil = Image.open(filter_path).convert("RGBA")
        filter_img_np = np.array(filter_img_pil)
        filter_rgb = filter_img_np[:, :, :3]
        filter_alpha = filter_img_np[:, :, 3]
        return filter_rgb, filter_alpha
    except FileNotFoundError:
        st.warning(f"Filter image not found: **{filter_path}**. Check 'filters/' folder and filename.")
        return None, None
    except Exception as e:
        st.warning(f"Could not load filter image **{filter_path}**. Details: {e}")
        return None, None

# --- Image Processing Utilities ---
def overlay_image_alpha(base_img, overlay_img, x, y, alpha_mask):
    """Overlays an image with an alpha channel onto a base image."""
    x, y = int(x), int(y)
    h, w = overlay_img.shape[0], overlay_img.shape[1]

    # Calculate boundaries for slicing to prevent out-of-bounds errors
    y1, y2 = max(0, y), min(base_img.shape[0], y + h)
    x1, x2 = max(0, x), min(base_img.shape[1], x + w)

    y1_overlay, y2_overlay = max(0, -y), min(h, base_img.shape[0] - y)
    x1_overlay, x2_overlay = max(0, -x), min(w, base_img.shape[1] - x)

    if x1 >= x2 or y1 >= y2:
        return base_img

    roi = base_img[y1:y2, x1:x2]
    overlay_cropped = overlay_img[y1_overlay:y2_overlay, x1_overlay:x2_overlay]
    alpha_cropped = alpha_mask[y1_overlay:y2_overlay, x1_overlay:x2_overlay]

    alpha_factor = np.expand_dims(alpha_cropped / 255.0, axis=2)
    blended_region = (overlay_cropped * alpha_factor + roi * (1 - alpha_factor)).astype(np.uint8)

    base_img[y1:y2, x1:x2] = blended_region
    return base_img

def get_emotion_display_color(emotion):
    """Returns a BGR color for drawing based on the detected emotion."""
    if emotion == 'Angry': return (0, 0, 255)
    elif emotion == 'Sad': return (255, 0, 0)
    elif emotion == 'Happy': return (0, 255, 255)
    elif emotion == 'Surprise': return (255, 165, 0)
    elif emotion == 'Fear': return (128, 0, 128)
    elif emotion == 'Disgust': return (0, 128, 0)
    else: return (0, 255, 0)

# --- Main Processing Function ---
def analyze_and_filter_image(image_source, emotion_model, face_cascade):
    """Processes an image to detect faces, predict emotions, and apply filters."""
    if image_source is None:
        st.info("Upload an image or take a picture with the webcam.")
        return

    try:
        img_pil = Image.open(image_source).convert("RGB")
        img_np = np.array(img_pil)
        display_img = img_np.copy()

        gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) == 0:
            st.warning("No face detected. Ensure a clear face is visible.")
            st.image(img_np, caption="Original Image (No face detected)", use_column_width=True)
            return

        for (x, y, w, h) in faces:
            try:
                # Extract and validate face region
                face_roi_gray = gray_img[max(0, y):min(img_np.shape[0], y + h), max(0, x):min(img_np.shape[1], x + w)]

                if face_roi_gray.size == 0 or w <= 0 or h <= 0:
                    st.warning(f"Skipping invalid face detection (x={x}, y={y}, w={w}, h={h}).")
                    continue

                # Prepare face for emotion prediction
                face_roi_resized = cv2.resize(face_roi_gray, (64, 64), interpolation=cv2.INTER_AREA)
                model_input = img_to_array(face_roi_resized.astype("float") / 255.0)
                model_input = np.expand_dims(model_input, axis=0)

                # Predict emotion
                predictions = emotion_model.predict(model_input, verbose=0)[0]
                emotion_idx = np.argmax(predictions)
                detected_emotion = EMOTIONS[emotion_idx]
                confidence_percent = predictions[emotion_idx] * 100
            except Exception as e:
                st.warning(f"Could not predict emotion for a face. Skipping filter. Details: {e}")
                detected_emotion = "Unknown"
                confidence_percent = 0.0

            # Get color for drawing
            draw_color = get_emotion_display_color(detected_emotion)

            # Draw rectangle around the face
            cv2.rectangle(display_img, (x, y), (x + w, y + h), draw_color, 2)

            # Display emotion and accuracy
            text_label = f"{detected_emotion} ({confidence_percent:.2f}%)"
            text_y_pos = y - 40 if y - 40 > 10 else y + h + 40 # Position text above or below face
            cv2.putText(display_img, text_label, (x, text_y_pos), cv2.FONT_HERSHEY_DUPLEX, 0.9, draw_color, 2)

            # Apply filter
            if detected_emotion in FILTERS and FILTERS[detected_emotion]:
                chosen_filter_path = random.choice(FILTERS[detected_emotion])
                filter_rgb, filter_alpha = load_filter_image(chosen_filter_path)

                if filter_rgb is not None and filter_alpha is not None:
                    filter_width = int(w * 1.2)
                    filter_height = int(h * 1.2)

                    if filter_width > 0 and filter_height > 0:
                        filter_rgb_resized = cv2.resize(filter_rgb, (filter_width, filter_height), interpolation=cv2.INTER_AREA)
                        filter_alpha_resized = cv2.resize(filter_alpha, (filter_width, filter_height), interpolation=cv2.INTER_AREA)

                        # Calculate filter position (slightly adjust y for better placement)
                        filter_x_pos = x - int((filter_width - w) / 2)
                        filter_y_pos = y - int(filter_height * 0.15) # Shift up by 15% of filter height

                        display_img = overlay_image_alpha(display_img, filter_rgb_resized, filter_x_pos, filter_y_pos, filter_alpha_resized)
                    else:
                        st.warning(f"Could not resize filter for '{detected_emotion}' due to invalid dimensions. Skipping.")

        st.image(display_img, caption="Processed Image with Emotion and Filter", use_column_width=True)

    except UnidentifiedImageError:
        st.error("The uploaded file is not a valid image. Please upload a JPG, JPEG, or PNG file.")
    except Exception as e:
        st.error(f"An unexpected error occurred during image processing. Details: {e}")
        st.info("If this persists, check your terminal for more specific error messages, or try a different image/webcam setup.")

# --- Streamlit UI ---
st.set_page_config(
    page_title="AI Mirror: Emotion Detector",
    page_icon="🪞",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.title("🪞 AI Mirror: Emotion Detector")
st.markdown("""
    <style>
    /* General app background and text colors */
    .stApp {
        background-color: #0d1117; /* Dark blue-black for a techy feel */
        color: #e0e6f2; /* Light blue-grey for text */
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #5d9cec; /* A vibrant, yet subtle blue for headings */
    }

    /* Markdown text */
    .stMarkdown p {
        color: #c9d1d9; /* Slightly lighter text for readability */
    }

    /* Radio button labels */
    .stRadio > label {
        color: #e0e6f2;
    }

    /* Buttons (File Uploader, Camera Input, etc.) */
    .stFileUploader > div > button,
    .stCameraInput > div > button,
    .stButton > button {
        background-color: #1f6feb; /* A strong, clear blue for buttons */
        color: white;
        border-radius: 8px;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        cursor: pointer;
        transition: background-color 0.3s ease, transform 0.2s ease;
    }

    /* Button hover effect */
    .stFileUploader > div > button:hover,
    .stCameraInput > div > button:hover,
    .stButton > button:hover {
        background-color: #2f81ff; /* Lighter blue on hover */
        transform: translateY(-2px); /* Slight lift effect */
    }

    /* Text input, selectbox, number input backgrounds */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > select,
    .stNumberInput > div > div > input {
        background-color: #161b22; /* Darker input fields */
        color: #e0e6f2;
        border: 1px solid #30363d;
        border-radius: 5px;
    }

    /* Info, warning, error message boxes */
    .stAlert {
        border-left: 5px solid;
        border-radius: 5px;
    }
    .stAlert.info { border-color: #5d9cec; background-color: rgba(93, 156, 236, 0.1); color: #c9d1d9; }
    .stAlert.warning { border-color: #d1b600; background-color: rgba(209, 182, 0, 0.1); color: #c9d1d9; }
    .stAlert.error { border-color: #e65252; background-color: rgba(230, 82, 82, 0.1); color: #c9d1d9; }

    /* Custom CSS to hide the Streamlit menu and footer for a cleaner look */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """, unsafe_allow_html=True)

st.markdown("""
    Use this app to **detect emotions** from faces in images and apply **fun filters**!
    Choose between uploading an image file or using your webcam to take a picture.
    """, unsafe_allow_html=True)

# Load assets once at the start of the app using st.cache_resource for performance
emotion_model = get_emotion_model(MODEL_PATH)
face_cascade = get_face_cascade(HAARCASCADE_PATH)

# Option selection: File Upload or Webcam
st.markdown("---")
option = st.radio("### Choose your input method:", ("Upload an Image", "Use Webcam"), horizontal=True)
st.markdown("---")

if option == "Upload an Image":
    uploaded_file = st.file_uploader("Upload an image file (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        analyze_and_filter_image(uploaded_file, emotion_model, face_cascade)
elif option == "Use Webcam":
    st.info("Make sure your browser has granted permission to access your webcam.")
    camera_image = st.camera_input("Click 'Take Photo' to capture an image")
    if camera_image is not None:
        analyze_and_filter_image(camera_image, emotion_model, face_cascade)

st.markdown("---")
st.markdown("""
    **Note:** Emotion detection accuracy may vary. This app is for entertainment purposes.
    Ensure `emotion_model.h5`, `haarcascade_frontalface_default.xml`, and the `filters/` folder with transparent PNGs are correctly placed.
    """)
