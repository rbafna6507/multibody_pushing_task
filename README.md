### Testing

Running `install.sh` will install all required libraries. 

Running `demo.py` will run a simple demo that demonstrates the neural network in action using our MPPI strategy to learn multibody indirect manipulation.

All assets required are included in this repo.

### Project Overview
This project focuses on indirect robotic manipulation, where a robot arm learns to push a target cube to a goal using an intermediate cube. It utilizes a data-driven approach where a neural network first learns the complex physics of the multi-body interactions. This learned model then guides a Model Predictive Path Integral (MPPI) controller to plan a sequence of pushes within a PyBullet simulation.

### Simulation Environment
The virtual workspace contains a Panda robot arm, a yellow intermediate cube, and a red target cube. The system state is a 6D vector representing the poses of both cubes, while the robot's push action is a 3D vector defining the contact location, angle, and distance. To ensure indirect manipulation, collision between the robot arm and the target cube was disabled.


### Dynamics Learning and Data
Analytical modeling is challenging due to complex contact forces. Data was collected using a biased strategy that increased the probability of contact between the cubes, enriching the dataset with informative interaction events. This data was then processed for supervised learning and split into 80% for training and 20% for validation.


### Model and Control Algorithm
A residual neural network was designed to predict the  change in state resulting from an action, a method that often yields more stable learning. This learned dynamics model serves as the predictive core for an MPPI controller, which plans over a 40-step horizon by sampling 1000 action sequences at each step. The controller optimizes a cost function that heavily penalizes the target's distance from the goal while also encouraging the intermediate object to stay close to the target to facilitate pushing.