import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from data_loader import NavigationDataset
from model import MultiModalTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = NavigationDataset("data")
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = MultiModalTransformer().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss_fn = torch.nn.MSELoss()

losses = []

EPOCHS = 40
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for img, text, path in loader:
        img, text, path = img.to(device), text.to(device), path.to(device)

        pred = model(img, text)
        loss = loss_fn(pred, path)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    losses.append(avg_loss)
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {avg_loss:.4f}")

# Save model
torch.save(model.state_dict(), "multimodal_transformer.pth")

# Plot loss
plt.plot(losses)
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("Training Loss Curve")
plt.savefig("loss_curve.png")
plt.show()
