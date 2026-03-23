import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from model import ProjectileNet

# Load model
model = ProjectileNet()
model.load_state_dict(torch.load("model.pth"))
model.eval()

def predict(v0, angle_deg, t):
    angle = angle_deg * math.pi / 180
    inp = torch.tensor([[v0, angle, t]], dtype=torch.float32)
    out = model(inp).detach().numpy()[0]
    return out  # (x, y)

def true_trajectory(v0, angle_deg, t):
    angle = angle_deg * math.pi / 180
    g = 9.81
    x = v0 * math.cos(angle) * t
    y = v0 * math.sin(angle) * t - 0.5 * g * t**2
    return x, y

# Compare predicted vs true trajectory
v0 = 23
angle = 43

times = np.linspace(0, 3, 100)
true_x, true_y = [], []
pred_x, pred_y = [], []

for t in times:
    tx, ty = true_trajectory(v0, angle, t)
    px, py = predict(v0, angle, t)

    true_x.append(tx)
    true_y.append(ty)
    pred_x.append(px)
    pred_y.append(py)

# Plot comparison
plt.plot(true_x, true_y, label="True Trajectory")
plt.plot(pred_x, pred_y, label="Predicted Trajectory")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Model Prediction vs True Physics")
plt.legend()
plt.grid(True)
plt.savefig("trajectory_comparison.png")
print("Saved trajectory_comparison.png")