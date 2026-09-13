import os
import cv2

# Anchor paths directly relative to backend directory
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BACKEND_DIR)

# Update the video filename/path if it is in data/videos or project root
# Common locations to check:
possible_paths = [
    os.path.join(PROJECT_ROOT, "drone_flight.mp4"),
    os.path.join(PROJECT_ROOT, "data", "videos", "drone_flight.mp4"),
    os.path.join(PROJECT_ROOT, "data", "drone_flight.mp4")
]

video_path = None
for p in possible_paths:
    if os.path.exists(p):
        video_path = p
        break

if not video_path:
    raise FileNotFoundError(
        f"Video file not found. Checked:\n" + "\n".join(possible_paths)
    )

output_dir = os.path.join(PROJECT_ROOT, "data", "workspace", "images")
os.makedirs(output_dir, exist_ok=True)

# Clean out stale frames
for f in os.listdir(output_dir):
    file_path = os.path.join(output_dir, f)
    if os.path.isfile(file_path):
        os.remove(file_path)

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise RuntimeError(f"OpenCV could not open video: {video_path}")

total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
print(f"Loaded: {video_path} ({total_video_frames} frames, {fps:.1f} FPS)")

# Step = 2 extracts ~120 to 150 frames from a 10s clip
step = 2
saved = 0
count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    if count % step == 0:
        frame_filename = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
        cv2.imwrite(frame_filename, frame)
        saved += 1
    count += 1

cap.release()
print(f"Extracted {saved} frames to {output_dir}")