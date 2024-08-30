import os
import cv2
import numpy as np
import tensorflow as tf
import argparse

# Parse command-line arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Run webcam-based liveness detection.')
    parser.add_argument('-m', '--model', type=str, required=True, help='Path to the trained model file')
    return parser.parse_args()

# Load your trained model
def load_model(model_path):
    return tf.keras.models.load_model(model_path)

# Function to preprocess the frame before feeding into the model
def preprocess_frame(frame):
    # Resize the frame to the input size expected by the model (e.g., 150x150)
    resized_frame = cv2.resize(frame, (150, 150))
    # Convert to array and normalize (scale pixel values between 0 and 1)
    resized_frame = np.array(resized_frame) / 255.0
    # Expand dimensions to fit model input (batch size, height, width, channels)
    resized_frame = np.expand_dims(resized_frame, axis=0)
    return resized_frame

# Function to capture video from the webcam and make predictions
def capture_webcam_and_predict(model):
    # Initialize webcam (0 is the default device index for the primary webcam)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Check if the webcam is opened correctly
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("[INFO] loading face detector...")
    protoPath = os.path.sep.join(["./face_detector", "deploy.prototxt"])
    modelPath = os.path.sep.join(["./face_detector", "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

    # Best values
    tolerance_x = 320  # Tolerance in the x-direction
    tolerance_y = 180  # Tolerance in the y-direction


    while True:
        # Capture frame-by-frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                     (300, 300), (104.0, 177.0, 123.0))

        # Pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)
        detections = net.forward()

        out_of_limits = False  # Flag to indicate if any face is out of limits

        for i in range(0, detections.shape[2]):
            # Extract the confidence (i.e., probability) associated with the prediction
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections
            if confidence > 0.6:
                # Compute the (x, y)-coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Apply tolerance to the bounding box
                startX = max(0, startX - tolerance_x)
                startY = max(0, startY - tolerance_y)
                endX = min(w, endX + tolerance_x)
                endY = min(h, endY + tolerance_y)

                # Check if the bounding box is out of limits
                if startX <= 0 or startY <= 0 or endX >= w or endY >= h:
                    out_of_limits = True
                    color = (0, 0, 255)  # Red if out of limits
                    cv2.putText(frame, "Out of limits", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    # Draw red rectangle indicating out of limits, skip prediction
                    cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                    continue  # Skip predictions if out of limits

                # Preprocess the frame for the model
                face_region = frame[startY:endY, startX:endX]
                preprocessed_frame = preprocess_frame(face_region)

                # Make predictions using the model
                preds = model.predict(preprocessed_frame)
                prediction = np.argmax(preds, axis=1).astype(int)

                # Depending on the model output, interpret the prediction
                label = 'spoof' if prediction else 'real'
                color = (0, 0, 255) if prediction else (0, 255, 0)

                # Draw the bounding box around the face
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

                # Display the label on the frame (in the bottom-left corner of the frame)
                text = f"Prediction: {label}"
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                bottom_left = (10, h - 10)
                cv2.putText(frame, text, bottom_left, cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # Display the resulting frame
        cv2.imshow('Webcam - Press "q" to exit', frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture when done
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    model = load_model(args.model)
    capture_webcam_and_predict(model)
