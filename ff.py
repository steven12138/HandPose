import cv2
import mediapipe as mp

# Set up camera input
cap = cv2.VideoCapture(0)

# Choose pre-trained model
model = mp.solutions.hands.Hands()

# Initialize hand tracking module
with model:
    while True:
        # Read frame from camera
        ret, image = cap.read()
        if not ret:
            break

        # Process the camera frame
        results = model.process(image)

        # Mark recognized hands
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for idx, landmark in enumerate(hand_landmarks.landmark):
                    # Draw a circle at each landmark point
                    cv2.circle(image, (int(landmark.x * image.shape[1]), int(landmark.y * image.shape[0])), 5,
                               (0, 255, 0), -1)

        # Display the output
        cv2.imshow("MediaPipe Hand Tracking", image)
        if cv2.waitKey(1) == ord("q"):
            break

# Release resources
cap.release()
cv2.destroyAllWindows()
