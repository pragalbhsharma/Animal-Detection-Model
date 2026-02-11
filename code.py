!pip install ultralytics opencv-python-headless pandas matplotlib
import pandas as pd
from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from google.colab import files

csv_path = "/content/dataset2_labels.csv"
df = pd.read_csv(csv_path)

CARNIVORES = [
    "lion", "tiger", "bear", "wolf", "leopard",
    "dog", "cat", "fox"
]
model = YOLO("yolov8n.pt")

img_path = df.iloc[0]["image_path"]
img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

results = model(img)

carnivore_count = 0

for r in results:
    for box in r.boxes:
        cls_id = int(box.cls)
        label = r.names[cls_id]

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if label in CARNIVORES:
            color = (255, 0, 0)  # RED
            carnivore_count += 1
        else:
            color = (0, 255, 0)  # GREEN

        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        cv2.putText(
            img, label,
            (x1, y1-5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6, color, 2
        )

# Show image
plt.figure(figsize=(10,10))
plt.imshow(img)
plt.axis("off")
plt.show()

print("Carnivorous Animals Detected:", carnivore_count)
uploaded = files.upload()

video_path = list(uploaded.keys())[0]
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    carnivore_count = 0

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            label = r.names[cls_id]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if label in CARNIVORES:
                color = (0,0,255)
                carnivore_count += 1
            else:
                color = (255,0,0)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(
                frame, label,
                (x1, y1-5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, color, 2
            )

    cv2.putText(
        frame,
        f"Carnivores: {carnivore_count}",
        (20,40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1, (0,0,255), 2
    )

    cv2.imshow("Animal Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Total carnivorous animals detected:", carnivore_count)
print(f"Number of carnivorous animals detected: {carnivore_count}")
