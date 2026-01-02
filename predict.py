import torch
import json
import numpy as np
from PIL import Image
import torchvision.transforms as T
import matplotlib.pyplot as plt
from model import MultiModalTransformer

VOCAB = {
    "go": 0, "to": 1, "the": 2,
    "red": 3, "green": 4, "blue": 5,
    "circle": 6, "square": 7, "triangle": 8
}
PAD_IDX = 9
MAX_LEN = 5

device = "cuda" if torch.cuda.is_available() else "cpu"


def tokenize(text):
    tokens = text.lower().replace(",", "").split()
    ids = [VOCAB.get(t, PAD_IDX) for t in tokens]
    ids = ids[:MAX_LEN] + [PAD_IDX] * (MAX_LEN - len(ids))
    return torch.tensor(ids).unsqueeze(0)


model = MultiModalTransformer().to(device)
model.load_state_dict(
    torch.load("multimodal_transformer.pth", map_location=device)
)
model.eval()

image_path = "test_data/images/000000.png"
text_command = "Go to the Red Circle"

img = Image.open(image_path).convert("RGB")
img_tensor = T.ToTensor()(img).unsqueeze(0).to(device)
text_tensor = tokenize(text_command).to(device)

with torch.no_grad():
    pred = model(img_tensor, text_tensor)[0].cpu().numpy() * 128

# Visualization
plt.imshow(img)
plt.plot(pred[:, 0], pred[:, 1], "ro-")
plt.xlim(0, 128)
plt.ylim(128, 0)
plt.grid(True)
plt.show()

# JSON output
output = {
    "image_file": image_path.split("/")[-1],
    "text": text_command,
    "path": pred.tolist(),
    "target": {
        "position": pred[-1].tolist()
    }
}

print(json.dumps(output, indent=2))
