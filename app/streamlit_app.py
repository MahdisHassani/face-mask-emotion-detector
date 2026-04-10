import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from models.multitask_model import MultiTaskModel

# load model
device = torch.device("cpu")

model = MultiTaskModel().to(device)
checkpoint = torch.load("best_model.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

mask_labels = ["No Mask", "Mask"]
emotion_labels = ["Happy", "Neutral"]

st.title("Face Mask + Emotion Detector 😎")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        mask_out, emotion_out = model(img_tensor)

        mask_pred = mask_out.argmax(1).item()
        emotion_pred = emotion_out.argmax(1).item()

    st.write("### Prediction:")
    st.write(f"Mask: {mask_labels[mask_pred]}")
    st.write(f"Emotion: {emotion_labels[emotion_pred]}")