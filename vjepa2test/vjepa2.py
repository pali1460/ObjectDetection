import cv2
import torch
from collections import deque
from transformers import AutoVideoProcessor, VJEPA2ForVideoClassification

# --- Configuration ---
NUM_FRAMES = 64  # Number of consecutive frames to consider
FRAME_WIDTH, FRAME_HEIGHT = 256, 256  # Model expects 256x256 input

# --- Initialize Webcam Display ---
cv2.namedWindow("Display", cv2.WINDOW_AUTOSIZE)

# --- Load Pretrained V-JEPA 2 Model and Processor ---
processor = AutoVideoProcessor.from_pretrained("facebook/vjepa2-vitl-fpc16-256-ssv2")
model = VJEPA2ForVideoClassification.from_pretrained(
    "facebook/vjepa2-vitl-fpc16-256-ssv2"
).to("cuda").eval()

# --- Open Webcam Stream ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()

# --- Store Frames for Inference ---
frame_buffer = deque(maxlen=NUM_FRAMES)

def preprocess_clip(frames):
    """
    Resize and process frames for input to the model.
    """
    resized = [cv2.resize(f, (FRAME_WIDTH, FRAME_HEIGHT)) for f in frames]
    return processor(resized, return_tensors="pt").to("cuda")

text_to_display = "No Predictions Yet"

# --- Main Loop ---
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        org = (50, 50) # (x, y) coordinates
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = (0, 255, 0) # Green color (BGR)
        thickness = 1
        cv2.putText(frame, text_to_display, org, font, font_scale, color, thickness)
        # Display current frame
        cv2.imshow("Display", frame)

        # Append frame to buffer
        frame_buffer.append(frame)

        # Run prediction when we have enough frames
        if len(frame_buffer) == NUM_FRAMES:
            inputs = preprocess_clip(list(frame_buffer))
            with torch.no_grad():
                outputs = model(**inputs)
            logits = outputs.logits
            predicted_label = logits.argmax(-1).item()
            label = model.config.id2label[predicted_label]
            print(f"Prediction: {label}")
            text_to_display = "Preduction: " + label
            frame_buffer.clear()

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()