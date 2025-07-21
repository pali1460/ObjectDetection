import cv2
import torch
from collections import deque
from transformers import AutoVideoProcessor, VJEPA2ForVideoClassification

# --- Configuration ---
NUM_FRAMES = 16  # Number of consecutive frames to consider
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
padding = 5
background_color = (0, 0, 0)  # Black background
org = (50, 50) # (x, y) coordinates
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
color = (255, 255, 255) # White Text
thickness = 1
# --- Main Loop ---
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break        
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(text_to_display, font, font_scale, thickness)

        # Calculate rectangle coordinates
        # The rectangle's top-left corner
        rect_x1 = 50 - padding
        rect_y1 = 400 - text_height - padding
        # The rectangle's bottom-right corner
        rect_x2 = 50 + text_width + padding
        rect_y2 = 400 + baseline + padding

        # Draw the filled rectangle (background)
        cv2.rectangle(frame, (rect_x1, rect_y1), (rect_x2, rect_y2), background_color, -1)

        # Draw the text on top of the background
        cv2.putText(frame, text_to_display, (50, 400), font, font_scale, color, thickness)

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
            text_to_display = "Prediction: " + label
            frame_buffer.clear()

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
