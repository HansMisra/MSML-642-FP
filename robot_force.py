import pybullet as p
import pybullet_data
import numpy as np
import time
import pickle as pk
import random
#from bipedal.robot import Bipedal

# Initialize Pybullet
#p.connect(p.GUI)  # Use p.DIRECT for headless
p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
#p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)  # Disable rendering
#p.setTimeStep(1 / 500)

start_position = [0,0,0.55]
start_orientation = [0,0,0]
# Load the plane and robot
plane_id = p.loadURDF("plane.urdf")
robot_id = p.loadURDF("./bipedal/urdf/bipedal.urdf", start_position, p.getQuaternionFromEuler(start_orientation))#, [0, 0, 0.2], useFixedBase= False)  # Replace with your robot's URDF
# Define joints
joint_indices = [3, 4, 5, 9, 10, 11]  # Replace with your robot's joint indices
action_space = [-0.5, 0, 0.5]  # Joint action steps (torques or positions)
state_size = 177147  # Discretized state space size
action_size = len(action_space) ** len(joint_indices)

# Initialize Q-Table
Q = np.zeros((state_size, action_size), dtype=np.float32)

# Training Parameters
alpha = 0.01  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Exploration rate
max_force = 10 # Maximum external force


# Helper functions
def get_robot_state(robot_id):
    """Discretize robot state for Q-Learning."""
    base_position, base_orientation = p.getBasePositionAndOrientation(robot_id)
    joint_states = [p.getJointState(robot_id, i)[0] for i in joint_indices]
    return int(sum(abs(j) for j in joint_states) * 10)  # Example: use summed joint angles

def calculate_reward(robot_id, previous_time_upright):
    """Reward for balancing."""
    base_position, base_orientation = p.getBasePositionAndOrientation(robot_id)
    tilt = abs(base_orientation[1])  # Roll angle
    if base_position[2] < 0.05:  # Fallen
        return -100, previous_time_upright
    upright_time_reward = 1  # Reward for each timestep upright
    return 10 - tilt * 10 + previous_time_upright, previous_time_upright + 1

def apply_random_force(robot_id, max_force):
    """Apply random external force to the robot."""
    sign_x = random.choice([-1, 0, 1])
    sign_y = random.choice([-1, 0, 1])
    force = [sign_x*0.1*max_force, sign_y*0.1*max_force, 0]
    #force = [np.random.uniform(-max_force, max_force), np.random.uniform(-max_force, max_force), 0]
    position = [0, 0, 0.2]  # Center of mass
    p.applyExternalForce(robot_id, -1, force, position, p.WORLD_FRAME)

def reset_robot(robot_id):
    """Reset the robot to its initial position."""
    p.resetBasePositionAndOrientation(robot_id, start_position, p.getQuaternionFromEuler(start_orientation))
    for joint in joint_indices:
        p.resetJointState(robot_id, joint, 0)

for joint_index in range(p.getNumJoints(robot_id)):
    joint_info = p.getJointInfo(robot_id, joint_index)
    print(f"Joint Index: {joint_index}")
    print(f"Joint Name: {joint_info[1].decode('utf-8')}")
    print(f"Joint Type: {joint_info[2]}")  # 0 = Revolute, 1 = Prismatic, etc.
    print(f"Parent Link Index: {joint_info[16]}")
    print(f"Lower Limit: {joint_info[8]}, Upper Limit: {joint_info[9]}")
    print(f"----------------------------")

# Training Loop
episodes = 20000
MAX_TIMESTEP = 5000
try:
    for episode in range(episodes):
        reset_robot(robot_id)
        state = get_robot_state(robot_id)
        done = False
        episode_force = min(episode*2, episodes)*(max_force)/10000  # Gradual force increase
        timestep = 0
        upright_time = 0

        time.sleep(0.01)
        while not done:
            # Choose action
            if np.random.rand() < epsilon:
                action_index = np.random.randint(action_size)
            else:
                action_index = np.argmax(Q[state])
            
            # Map action index to joint actions
            action_values = [action_space[(action_index // (len(action_space) ** i)) % len(action_space)]
                            for i in range(len(joint_indices))]

            # Apply actions to joints
            p.setJointMotorControlArray(robot_id, joint_indices, p.TORQUE_CONTROL, forces=action_values)
            p.stepSimulation()

            # Apply random external force
            if timestep % 100 == 0:  # Every 100 steps
                apply_random_force(robot_id, max_force=episode_force)
            
            # Observe new state and calculate reward
            new_state = get_robot_state(robot_id)
            reward, upright_time = calculate_reward(robot_id, upright_time)
            if reward == -100 or timestep == MAX_TIMESTEP:  # Robot fell
                done = True

            # Update Q-Table
            Q[state, action_index] += alpha * (reward + gamma * np.max(Q[new_state]) - Q[state, action_index])
            state = new_state
            timestep += 1

        # Decay epsilon
        epsilon = max(0.1, epsilon * 0.9999)

        print(f"Episode {episode + 1}/{episodes} complete. Epsilon: {epsilon:.3f}. Reward: {reward:.3f}. Upright time: {upright_time}")
except KeyboardInterrupt:
    reset_robot(robot_id)
    state = get_robot_state(robot_id)
    action_index = np.argmax(Q[state])  # Exploit learned policy
    action_values = [action_space[(action_index // (len(action_space) ** i)) % len(action_space)]
                 for i in range(len(joint_indices))]
    with open('action_values.pkl', 'wb') as f:
        pk.dump(action_values, f)
    print("Action values saved")
    exit()


    #np.save("q_table.npy", Q)
    #print("Q-Table saved to q_table.npy")

# Testing
print("Testing trained policy...")
reset_robot(robot_id)
state = get_robot_state(robot_id)
action_index = np.argmax(Q[state])  # Exploit learned policy
action_values = [action_space[(action_index // (len(action_space) ** i)) % len(action_space)]
                 for i in range(len(joint_indices))]
with open('action_values.pkl', 'wb') as f:
    pk.dump(action_values, f)
print("Action values saved")

done = False
while not done:
    action_index = np.argmax(Q[state])  # Exploit learned policy
    action_values = [action_space[(action_index // (len(action_space) ** i)) % len(action_space)]
                     for i in range(len(joint_indices))]
    p.setJointMotorControlArray(robot_id, joint_indices, p.TORQUE_CONTROL, forces=action_values)
    p.stepSimulation()

    # Apply increasing force
    apply_random_force(robot_id, max_force=max_force)

    # Check if robot falls
    base_position, _ = p.getBasePositionAndOrientation(robot_id)
    if base_position[2] < 0.05:  # Fallen
        print("Robot fell.")
        done = True
