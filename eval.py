import cv2
import numpy as np
import tensorflow as tf
import time

model = tf.keras.models.load_model('model.h5')

emotion_labels = ["happy", "sad", "neutral"]
running = False
emotion = "None"


def predict_emotion(frame):
    # Resize to 48x48
    resized = cv2.resize(frame, (48, 48))
    
    # Convert to grayscale
    grayscale = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    
    # Normalize the pixel values
    normalized = grayscale / 255.0  # Model expects values between 0 and 1
    
    # Expand dimensions to match the input shape of the model
    input_data = np.expand_dims(normalized, axis=0)  # Add batch dimension
    input_data = np.expand_dims(input_data, axis=-1)  # Add channel dimension
    
    # Pass the image through the model
    prediction = model.predict(input_data)[0].argmax()
    return emotion_labels[prediction]


def run_camera():
    global running, emotion
    
    cap = cv2.VideoCapture(0)  # Initialize camera

    while True:
        if running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break

            # Display the camera feed
            cv2.imshow('Camera Feed', frame)

            emotion = predict_emotion(frame)

        # Display emotion on a blank UI
        ui = np.zeros((300, 600, 3), dtype="uint8")
        cv2.putText(ui, "Press 'S' to Start/Stop Camera", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(ui, "Press 'C' to Close", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(ui, f"Emotion: {emotion}", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('Emotion UI', ui)

        # Key press handling
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):  # Start/Stop camera
            running = not running
            if running:
                print("Camera started")
            else:
                print("Camera stopped")
        elif key == ord('c'):  # Close application
            print("Closing application")
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera()
