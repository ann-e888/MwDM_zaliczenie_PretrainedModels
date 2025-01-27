from ultralytics import YOLO

# Source path
source_path = "./test-images/family.jpg"

# Load pre-trained model
model = YOLO("pre-trained-models/yolo11n-cls.pt")
results = model(
    source=source_path,
    show=True,
    task="classification",
    save=True,
    save_dir="./results",
)


for result in results:
    classes = []
    for cls_idx, data in enumerate(result.probs.top5):
        classes.append((result.names[data], result.probs.top5conf[cls_idx]))

    for cls in classes:
        print(f"Class: {cls[0]}, Confidence: {cls[1]:.2f}")
