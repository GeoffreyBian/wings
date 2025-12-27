import sys
import torch
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F

from model import CNN

MODEL_PATH = "bird_cnn_pretrained.pth"
TOP_K = 5
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
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# --------------------
# Predict
# --------------------
image_path = sys.argv[1]
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    outputs = model(image)
    probs = F.softmax(outputs, dim=1)

    top_probs, top_idxs = probs.topk(TOP_K, dim=1)

print("\nTop predictions:")
for prob, idx in zip(top_probs[0], top_idxs[0]):
    print(f"{class_names[idx]}: {prob.item() * 100:.2f}%")
