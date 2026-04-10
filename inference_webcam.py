import cv2
import torch
import time
from PIL import Image
from torchvision import transforms
from facenet_pytorch import MTCNN
from models.multitask_model import MultiTaskModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model
model = MultiTaskModel().to(device)
checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# MTCNN
mtcnn = MTCNN(keep_all=True, device=device)

mask_labels = ["No Mask", "Mask"]
emotion_labels = ["Happy", "Neutral"]

cap = cv2.VideoCapture(0)

prev_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # FPS
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    prev_time = curr_time

    boxes, _ = mtcnn.detect(frame)

    if boxes is not None:
        for box in boxes:
            x1, y1, x2, y2 = map(int, box)

            face = frame[y1:y2, x1:x2]

            if face.size == 0:
                continue

            face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
            face_tensor = transform(face_pil).unsqueeze(0).to(device)

            with torch.no_grad():
                mask_out, emotion_out = model(face_tensor)

                mask_pred = mask_out.argmax(1).item()
                emotion_pred = emotion_out.argmax(1).item()

            label = f"{mask_labels[mask_pred]} | {emotion_labels[emotion_pred]}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.putText(frame, f"FPS: {int(fps)}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("AI Camera", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()