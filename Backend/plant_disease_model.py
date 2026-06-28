from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image
import torch

# Load model and processor once
processor = AutoImageProcessor.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")
model = AutoModelForImageClassification.from_pretrained("linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification")

# Predict function
def predict_disease(image_path: str):
    image = Image.open(image_path).convert("RGB")

    inputs = processor(images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class = model.config.id2label[predicted_class_idx]

    return {
        "class": predicted_class,
        "confidence": torch.softmax(logits, dim=1)[0][predicted_class_idx].item()
    }
