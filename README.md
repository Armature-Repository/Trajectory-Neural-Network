Neural Network Learning Projectile Motion
This project trains a small neural network to learn the underlying physical law of projectile motion using only simulated data. The model receives initial velocity, launch angle, and time as inputs, and predicts the projectile’s x and y positions. The goal is to demonstrate how machine learning can approximate smooth physical functions and to visualize how well the model learns over time.


Features
- Synthetic dataset generated from real projectile equations
- Fully connected neural network implemented in PyTorch
- Training loop with loss visualization
- Automatic saving of trained model weights
- Prediction script that compares learned trajectory vs. true physics
- Plots for both training loss and trajectory comparison


Technologies Used
- Python
- NumPy for data generation
- PyTorch for neural network modeling and training
- Matplotlib for visualization
- Standard physics equations for projectile motion


Project Structure
data_generator.py
model.py
train.py
predict.py


How to Run
- Generate the dataset:
python data_generator.py
- Train the model:
python train.py
- This produces:
- model.pth — trained weights
- loss.png — training loss curve
- Visualize predictions:
python predict.py
- This produces:
- trajectory_comparison.png — predicted vs. true trajectory


Training Loss Curve

<img width="640" height="480" alt="loss" src="https://github.com/user-attachments/assets/93b34a06-bab9-444e-a25a-aa6ce4cd1dc5" />

This plot shows how the model’s mean squared error decreases over time. As the number of epochs increases, the network gradually learns the smooth mapping from (v_0,\theta ,t) to (x,y).

Predicted vs. True Trajectory

<img width="640" height="480" alt="trajectory_comparison" src="https://github.com/user-attachments/assets/83059720-7f1b-430a-b5ec-3edea50c0a68" />
This plot compares the neural network’s predicted projectile path with the true analytical solution. A close match indicates that the model has successfully learned the underlying physical law.


Future Improvements
- Normalize inputs for faster convergence
- Add drag or wind resistance to create a more complex physical system
- Train on multiple physical regimes (e.g., springs, orbits)
- Add a GUI to visualize trajectories in real time
- Experiment with larger or deeper neural networks
