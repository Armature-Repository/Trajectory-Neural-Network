import torch
import numpy as np
import matplotlib.pyplot as plt
from model import ProjectileNet

# Load dataset
data = np.load("projectile_data.npy")
inputs = torch.tensor(data[:, :3], dtype=torch.float32)
targets = torch.tensor(data[:, 3:], dtype=torch.float32)

# Create model
model = ProjectileNet()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_fn = torch.nn.MSELoss()

epochs = 20000
loss_history = []

print("Training started...")

for epoch in range(epochs):
    optimizer.zero_grad()
    pred = model(inputs)
    loss = loss_fn(pred, targets)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    # Progress indicator
    if epoch % 100 == 0:
        print(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.6f}")

print("Training complete.")

# Save trained model
torch.save(model.state_dict(), "model.pth")
print("Saved model.pth")

# Plot loss curve
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.grid(True)
plt.savefig("loss.png")
print("Saved loss.png")