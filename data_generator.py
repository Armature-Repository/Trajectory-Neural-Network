import numpy as np

def generate_data(n=20000):
    g = 9.81
    data = []

    for _ in range(n):
        v0 = np.random.uniform(5, 50)
        angle = np.random.uniform(10, 80) * np.pi / 180
        t = np.random.uniform(0, 5)

        x = v0 * np.cos(angle) * t
        y = v0 * np.sin(angle) * t - 0.5 * g * t**2

        data.append([v0, angle, t, x, y])

    return np.array(data)

if __name__ == "__main__":
    data = generate_data()
    np.save("projectile_data.npy", data)
    print("Saved projectile_data.npy")