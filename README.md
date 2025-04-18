# Project Driver Drowsiness Detection
import cv2
import dlib
from scipy.spatial import distance

# Function to calculate the eye aspect ratio (EAR)
def calculate_ear(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds for EAR and consecutive frames
EAR_THRESHOLD = 0.25  # Eye Aspect Ratio threshold
CONSECUTIVE_FRAMES = 20  # Number of consecutive frames for drowsy detection

# Initialize frame counter
frame_counter = 0

# Initialize Dlib's face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # Download this file from Dlib's repository

# Indexes for the left and right eye landmarks
LEFT_EYE_INDEXES = list(range(36, 42))
RIGHT_EYE_INDEXES = list(range(42, 48))

# Initialize webcam
video_capture = cv2.VideoCapture(0)

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Convert frame to grayscale for better performance
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = detector(gray)

    for face in faces:
        # Get facial landmarks
        landmarks = predictor(gray, face)

        # Extract eye landmarks
        left_eye = [landmarks.part(i) for i in LEFT_EYE_INDEXES]
        right_eye = [landmarks.part(i) for i in RIGHT_EYE_INDEXES]

        # Convert landmarks to coordinates
        left_eye_coords = [(point.x, point.y) for point in left_eye]
        right_eye_coords = [(point.x, point.y) for point in right_eye]

        # Calculate EAR for both eyes
        left_ear = calculate_ear(left_eye_coords)
        right_ear = calculate_ear(right_eye_coords)
        avg_ear = (left_ear + right_ear) / 2.0

        # Check if EAR is below the threshold
        if avg_ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= CONSECUTIVE_FRAMES:
                cv2.putText(frame, "DROWSINESS DETECTED!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        else:
            frame_counter = 0

        # Draw eye contours on the frame
        for (x, y) in left_eye_coords + right_eye_coords:
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

    # Display the frame
    cv2.imshow("Driver Drowsiness Detection", frame)

    # Break loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
video_capture.release()
cv2.destroyAllWindows()
