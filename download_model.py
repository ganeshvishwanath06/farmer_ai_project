import torch
import torchvision.models as models

# Use MobileNetV2 (lightweight, good for hackathons)
model = models.mobilenet_v2(pretrained=True)

# Modify the last layer for 38 crop disease classes
model.classifier[1] = torch.nn.Linear(model.last_channel, 38)

# Download trained weights (from PlantVillage dataset)
url = "https://github.com/AakashKumarNain/Crop-Disease-Classification/releases/download/v1/plant_disease_model.pth"
model.load_state_dict(torch.hub.load_state_dict_from_url(url, progress=True))

# Save locally
torch.save(model.state_dict(), "plant_disease_model.pth")

print("✅ Model downloaded and saved as plant_disease_model.pth")
