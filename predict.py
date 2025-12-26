import sys
import torch
from torchvision import transforms
from PIL import Image

from model import CNN

MODEL_PATH = "bird_cnn_better.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------
# Load model + classes
# --------------------
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
class_names = checkpoint["class_names"]

model = CNN(num_classes=len(class_names))
model.load_state_dict(checkpoint["model_state"])
model.to(DEVICE)
model.eval()

# --------------------
# Image preprocessing
# --------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# --------------------
# Predict
# --------------------
image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    outputs = model(image)
    _, pred = torch.max(outputs, 1)

print("Predicted species:", class_names[pred.item()])
