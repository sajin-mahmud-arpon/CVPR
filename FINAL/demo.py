import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained MNIST model
model = load_model("mnist_cnn_model.keras")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Define ROI
    x1, y1, x2, y2 = 100, 100, 300, 300
    roi = gray[y1:y2, x1:x2]

    # Preprocess ROI
    roi_blur = cv2.GaussianBlur(roi, (5, 5), 0)
    _, roi_thresh = cv2.threshold(roi_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours
    contours, _ = cv2.findContours(roi_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        digit_crop = roi_thresh[y:y+h, x:x+w]

        # Resize and pad
        digit_resized = cv2.resize(digit_crop, (20, 20))
        canvas = np.zeros((28, 28), dtype=np.uint8)
        x_offset = (28 - 20) // 2
        y_offset = (28 - 20) // 2
        canvas[y_offset:y_offset+20, x_offset:x_offset+20] = digit_resized

        # Normalize and reshape
        roi_input = canvas.astype("float32") / 255.0
        roi_input = roi_input.reshape(1, 28, 28, 1)

        # Predict
        pred = model.predict(roi_input, verbose=0)
        digit = np.argmax(pred)
        confidence = np.max(pred)

        # Display prediction
        cv2.putText(frame, f"Predicted: {digit} ({confidence:.2f})", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

    # Draw ROI box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.putText(frame, "Draw digit inside box", (x1, y2 + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    cv2.imshow("Digit Recognition", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()