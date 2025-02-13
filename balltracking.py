import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import torch
# Initialize MediaPipe Pose model
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose()

# Define the drawing specifications for landmarks and connections
landmark_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2)
connection_spec = mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=2)  

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Load the Keras model
cnn_model = tf.keras.models.load_model('my_model.h5')

# Open video capture
cap = cv2.VideoCapture('videos/Roger Federer Serves from Back Perpsective in HD.mp4')

# Class IDs for YOLO (0 for person, 32 for sports ball)
person_class_id = 0
ball_class_id = 32

# Define a distance threshold for proximity detection
distance_threshold = 50  

# Action class labels
class_labels = ['backhand', 'forehand', 'neutral', 'serve']

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Run YOLOv5 to detect persons and balls
    results_yolo = model(frame)

    player_bbox = None
    ball_bbox = None
    best_conf_person = 0
    best_conf_ball = 0

    for detection in results_yolo.xyxy[0]:
        x1, y1, x2, y2, conf, cls = detection

        if int(cls) == person_class_id and conf > best_conf_person:
            best_conf_person = conf
            player_bbox = (int(x1), int(y1), int(x2), int(y2))
        elif int(cls) == ball_class_id and conf > best_conf_ball:
            best_conf_ball = conf
            ball_bbox = (int(x1), int(y1), int(x2), int(y2))

    if player_bbox is not None:
        x1, y1, x2, y2 = player_bbox
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Person {best_conf_person:.2f}', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        person_frame = rgb_frame[y1:y2, x1:x2]

        results_pose = pose.process(person_frame)

        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                person_frame, 
                results_pose.pose_landmarks, 
                mp_pose.POSE_CONNECTIONS, 
                landmark_drawing_spec=landmark_spec,
                connection_drawing_spec=connection_spec
            )

            landmarks = results_pose.pose_landmarks.landmark
            features = np.array([
                landmarks[mp_pose.PoseLandmark.NOSE].y, landmarks[mp_pose.PoseLandmark.NOSE].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y, landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].y, landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].y, landmarks[mp_pose.PoseLandmark.RIGHT_WRIST].x,
                landmarks[mp_pose.PoseLandmark.LEFT_HIP].y, landmarks[mp_pose.PoseLandmark.LEFT_HIP].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y, landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x,
                landmarks[mp_pose.PoseLandmark.LEFT_KNEE].y, landmarks[mp_pose.PoseLandmark.LEFT_KNEE].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].y, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x,
                landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x,
            ])

            features = np.expand_dims(features, axis=0)  # Add batch dimension
            features = features.reshape((features.shape[0], 1, 26, 1))  # Adjust shape

            if ball_bbox is not None:
                bx1, by1, bx2, by2 = ball_bbox
                cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                cv2.putText(frame, f'Ball {best_conf_ball:.2f}', (bx1, by1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

                # Calculate center points of the player and ball
                player_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                ball_center = ((bx1 + bx2) // 2, (by1 + by2) // 2)

                # Calculate the distance between player and ball
                distance = np.linalg.norm(np.array(player_center) - np.array(ball_center))

                
                    # Run the model prediction when the ball is close
                output = cnn_model.predict(features)
                class_label_index = np.argmax(output, axis=1)[0]
                action = class_labels[class_label_index]

                cv2.putText(frame, f'Action: {action}', (x1, y1 - 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

        frame[y1:y2, x1:x2] = cv2.cvtColor(person_frame, cv2.COLOR_RGB2BGR)

    cv2.imshow('YOLO and Mediapipe Pose Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
