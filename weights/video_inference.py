import cv2
from ultralytics import YOLO

MODEL_PATH = "best.pt"              # path to trained model
VIDEO_PATH = "manhole_video.webm"      # input road video
OUTPUT_PATH = "output_manhole_video.mp4"

CONFIDENCE = 0.40                   # optimal from F1 curve
DEVICE = "cpu"                     # "cuda" or "cpu"

model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise IOError("❌ Error opening video file")

width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = cap.get(cv2.CAP_PROP_FPS)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(
        frame,
        conf=CONFIDENCE,
        device=DEVICE,
        verbose=False
    )

    # YOLO built-in visualization (masks + boxes + labels)
    annotated_frame = results[0].plot()

    out.write(annotated_frame)

cap.release()
out.release()

print("✅ Video inference completed successfully")
print(f"📁 Output saved as: {OUTPUT_PATH}")
