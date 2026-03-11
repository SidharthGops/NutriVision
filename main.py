from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import timm
import json

with open("nutrition.json") as f:
    nutrition_db = json.load(f)

app = FastAPI()

# Load checkpoint
checkpoint = torch.load("model.pth", map_location="cpu")

classes = checkpoint["classes"]
NUM_CLASSES = len(classes)

# Build model using timm
model = timm.create_model(
    "efficientnet_b0",
    pretrained=False,
    num_classes=NUM_CLASSES
)

# Load weights
model.load_state_dict(checkpoint["model_state"])
model.eval()


transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = Image.open(file.file).convert("RGB")
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)

        predicted = probs.argmax(1).item()
        confidence = probs[0][predicted].item()

    food_name = classes[predicted]

    nutrition = nutrition_db.get(food_name, None)

    return {
        "food": food_name,
        "confidence": float(confidence),
        "nutrition": nutrition
    }