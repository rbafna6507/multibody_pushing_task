import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import torch.optim as optim

try:
    from panda_pushing_env import TARGET_POSE_MULTI, BOX_SIZE
except ImportError:
    print("Warning: Could not import from panda_pushing_env. Using default values.")
    TARGET_POSE_MULTI = np.array([0.75, 0.0, 0.0])
    BOX_SIZE = 0.1

print("loaded from /again")

def collect_data_random(env, num_trajectories=100, trajectory_length=10):
    collected_data = []
    pbar = tqdm(range(num_trajectories))
    for i in pbar:
        state = env.reset()
        states = [state]
        actions = []
        for t in range(trajectory_length):
            action = env.action_space.sample()
            try:
                next_state, _, done, _ = env.step(action)
                states.append(next_state)
                actions.append(action)
                state = next_state
                if done:
                    break
            except Exception as e:
                 print(f"Error during env.step in trajectory {i}, step {t}: {e}")
                 break

        if len(actions) > 0:
            states_array = np.array(states, dtype=np.float32)
            actions_array = np.array(actions, dtype=np.float32)
            collected_data.append({'states': states_array, 'actions': actions_array})

        pbar.set_description(f"Collected {len(collected_data)} trajectories")

    if not collected_data:
         print("Warning: No valid trajectories were collected. Check environment interaction and step function.")
    return collected_data


def collect_data_with_contacts(env,
                               num_trajectories: int = 500,
                               trajectory_length: int = 15,
                               contact_bias: float = 0.8,
                               noise_std: np.ndarray = np.array([0.15, 0.05, 0.15])):
    assert env.is_multi_object, "Use the multi-object environment!"
    collected = []

    low  = env.action_space.low
    high = env.action_space.high
    dtype = low.dtype

    for _ in tqdm(range(num_trajectories), desc="collecting"):
        state = env.reset()
        states = [state]
        actions = []
        contact_occurred = False

        for t in range(trajectory_length):
            if np.random.rand() < contact_bias:
                yellow, red = state[:3], state[3:]
                vec = red[:2] - yellow[:2]
                push_loc = 0.0
                push_ang = np.clip(
                    np.arctan2(vec[1], vec[0]) - yellow[2],
                    -np.pi/2, np.pi/2
                )
                desired_dist = np.linalg.norm(vec) - env.block_size
                push_len = np.clip(desired_dist/env.push_length, 0.1, 1.0)
                base_action = np.array([push_loc, push_ang, push_len], dtype=dtype)
            else:
                base_action = env.action_space.sample()

            noisy = base_action + (np.random.randn(3) * noise_std)
            clipped = np.clip(noisy, low, high)
            action = clipped.astype(dtype)

            next_state, _, done, _ = env.step(action)
            if np.linalg.norm(next_state[:2] - next_state[3:5]) < env.block_size * 1.1:
                contact_occurred = True

            states.append(next_state)
            actions.append(action)
            state = next_state
            if done:
                break

        if contact_occurred:
            collected.append({
                "states":  np.asarray(states,  dtype=np.float32),
                "actions": np.asarray(actions, dtype=np.float32)
            })

    return collected


def process_data_single_step(collected_data, batch_size=500):
    if not collected_data:
        raise ValueError("collected_data is empty in process_data_single_step")
    dataset = SingleStepDynamicsDataset(collected_data)
    if len(dataset) == 0:
        raise ValueError("SingleStepDynamicsDataset is empty after processing.")
    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    val_len = total_len - train_len
    if val_len == 0: # Ensure val_set is not empty
        if train_len > 0:
            train_len -= 1
            val_len = 1
        else:
            raise ValueError("Dataset too small to create train/val split.")
    train_data, val_data = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_data, batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size, shuffle=False)
    return train_loader, val_loader

def process_data_multiple_step(collected_data, batch_size=500, num_steps=4):
    if not collected_data:
        raise ValueError("collected_data is empty in process_data_multiple_step")
    dataset = MultiStepDynamicsDataset(collected_data, num_steps=num_steps)
    if len(dataset) == 0:
        raise ValueError(f"MultiStepDynamicsDataset is empty for num_steps={num_steps}")
    total_len = len(dataset)
    train_len = int(total_len * 0.8)
    val_len = total_len - train_len
    if val_len == 0:
        if train_len > 0:
            train_len -= 1
            val_len = 1
        else:
             raise ValueError("Dataset too small to create train/val split.")
    train_data, val_data = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_data, batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_data, batch_size, shuffle=False)
    return train_loader, val_loader


class SingleStepDynamicsDataset(Dataset):
    def __init__(self, collected_data):
        self.samples = []
        for traj in collected_data:
            states = traj['states']
            actions = traj['actions']
            num_transitions = min(len(states) - 1, len(actions))
            for i in range(num_transitions):
                self.samples.append({
                    'state': torch.tensor(states[i], dtype=torch.float32),
                    'action': torch.tensor(actions[i], dtype=torch.float32),
                    'next_state': torch.tensor(states[i+1], dtype=torch.float32),
                })
        if not self.samples:
             print("Warning: SingleStepDynamicsDataset is empty after processing collected_data.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]


class MultiStepDynamicsDataset(Dataset):
    def __init__(self, collected_data, num_steps=4):
        self.samples = []
        self.num_steps = num_steps
        for traj in collected_data:
            states = traj['states']
            actions = traj['actions']
            traj_len = len(actions)
            if traj_len >= num_steps:
                for i in range(traj_len - num_steps + 1):
                    self.samples.append({
                        'state': torch.tensor(states[i], dtype=torch.float32),
                        'action': torch.tensor(actions[i:i+num_steps], dtype=torch.float32),
                        'next_state': torch.tensor(states[i+1 : i+num_steps+1], dtype=torch.float32)
                    })
        if not self.samples:
             print(f"Warning: MultiStepDynamicsDataset is empty for num_steps={num_steps}. Check data collection or num_steps.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]

class SE2PoseLoss(nn.Module):
    def __init__(self, block_width, block_length):
        super().__init__()
        self.w = block_width
        self.l = block_length
        self.rg_sq = torch.tensor((self.l**2 + self.w**2)/12.0, dtype=torch.float32)

    def forward(self, pose_pred, pose_target):
        rg = torch.sqrt(self.rg_sq).to(pose_pred.device)
        position_loss = F.mse_loss(pose_pred[..., :2], pose_target[..., :2])
        # Angle difference handling wraparound
        angle_diff = pose_pred[..., 2] - pose_target[..., 2]
        angle_diff = torch.atan2(torch.sin(angle_diff), torch.cos(angle_diff))
        orientation_loss = F.mse_loss(angle_diff, torch.zeros_like(angle_diff))
        se2_pose_loss = position_loss + rg * orientation_loss
        return se2_pose_loss

class SingleStepLoss(nn.Module):
    def __init__(self, loss_fn_3d):
        super().__init__()
        self.loss = loss_fn_3d

    def forward(self, model, state, action, target_state):
        predicted_state = model(state, action)
        state_dim = predicted_state.shape[-1]

        if state_dim == 6:
            loss_intermediate = self.loss(predicted_state[..., :3], target_state[..., :3])
            loss_target = self.loss(predicted_state[..., 3:], target_state[..., 3:])
            loss_val = loss_intermediate + loss_target
        elif state_dim == 3:
            loss_val = self.loss(predicted_state, target_state)
        else:
            raise ValueError(f"Unsupported state dimension {state_dim} in SingleStepLoss")
        return loss_val

class MultiStepLoss(nn.Module):
    def __init__(self, loss_fn_3d, discount=0.99):
        super().__init__()
        self.loss = loss_fn_3d
        self.discount = discount

    def forward(self, model, state, actions, target_states):
        multi_step_loss = 0.0
        current_state = state
        num_steps = actions.size(1)
        state_dim = state.shape[-1]

        for step in range(num_steps):
            action = actions[:, step, :]
            target = target_states[:, step, :]
            predicted_state = model(current_state, action)

            if state_dim == 6:
                loss_intermediate = self.loss(predicted_state[..., :3], target[..., :3])
                loss_target = self.loss(predicted_state[..., 3:], target[..., 3:])
                step_loss = loss_intermediate + loss_target
            elif state_dim == 3:
                step_loss = self.loss(predicted_state, target)
            else:
                 raise ValueError(f"Unsupported state dimension {state_dim} in MultiStepLoss")

            multi_step_loss += (self.discount ** step) * step_loss
            current_state = predicted_state.detach()

        return multi_step_loss

class AbsoluteDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        hidden_size = 100
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, state_dim)
        self.relu = nn.ReLU()

    def forward(self, state, action):
        x = torch.cat((state, action), dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        next_state = self.fc3(x)
        return next_state

class ResidualDynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        hidden_size = 100
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, state_dim)
        self.relu = nn.ReLU()
        print(f"Initialized ResidualDynamicsModel with state_dim={state_dim}, action_dim={action_dim}")

    def forward(self, state, action):
        if state.shape[-1] != self.state_dim or action.shape[-1] != self.action_dim:
             raise ValueError(f"Input shape mismatch. State: {state.shape}, Action: {action.shape}. Expected state_dim={self.state_dim}, action_dim={self.action_dim}")

        x = torch.cat((state, action), dim=-1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        delta_state = self.fc3(x)
        next_state = state + delta_state
        return next_state

# --- Cost Functions (Keep only multi_body_cost_function for demo.py) ---
def multi_body_cost_function(state, action):
    device = state.device
    dtype = state.dtype
    try:
        # Ensure TARGET_POSE_MULTI is a tensor on the correct device
        goal = torch.as_tensor(TARGET_POSE_MULTI, dtype=dtype, device=device)
    except NameError:
        print("Warning: TARGET_POSE_MULTI not found globally, using default.")
        goal = torch.tensor([0.75, 0.0, 0.0], dtype=dtype, device=device)


    inter  = state[:, :3]
    targ   = state[:, 3:]

    # Target object cost (position and orientation)
    Q_t   = torch.diag(torch.tensor([10., 10., 0.1], device=device, dtype=dtype))
    delta_t = targ - goal
    # Wrap angle difference
    delta_t[:, 2] = torch.atan2(torch.sin(delta_t[:, 2]), torch.cos(delta_t[:, 2]))
    c_t   = torch.sum((delta_t @ Q_t) * delta_t, dim=-1)

    # Shaping cost: keep intermediate object near target (position only)
    c_i   = 0.1 * torch.sum((inter[:, :2] - targ[:, :2])**2, dim=-1)

    # Control penalty
    c_u   = 1e-3 * torch.sum(action**2, dim=-1)

    return c_t + c_i + c_u

class PushingController(object):
    def __init__(self, env, model, cost_function, num_samples=100, horizon=10):
        self.env = env
        self.model = model
        try:
            self.device = next(model.parameters()).device
        except StopIteration:
            print("Warning: Model has no parameters. Assuming CPU device for MPPI.")
            self.device = torch.device("cpu")

        self.model.eval()

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.shape[0]
        u_min = torch.from_numpy(env.action_space.low).to(self.device)
        u_max = torch.from_numpy(env.action_space.high).to(self.device)

        # MPPI Hyperparameters
        noise_sigma=torch.diag(torch.tensor([0.1, 0.2, 0.8], device=self.device))
        lambda_value = 0.01

        try:
            from mppi import MPPI
        except ImportError:
            raise ImportError("Could not import MPPI from mppi.py. Make sure the file is accessible.")

        self.mppi = MPPI(self._compute_dynamics,
                         cost_function,
                         nx=state_dim,
                         num_samples=num_samples,
                         horizon=horizon,
                         noise_sigma=noise_sigma,
                         lambda_=lambda_value,
                         u_min=u_min,
                         u_max=u_max,
                         device=self.device)

    def _compute_dynamics(self, state, action):
        with torch.no_grad():
            next_state = self.model(state, action)
        return next_state

    def control(self, state):
        action_np = None
        state_tensor = torch.from_numpy(state).float().to(self.mppi.d)
        action_tensor = self.mppi.command(state_tensor)
        action_np = action_tensor.detach().cpu().numpy()
        action_np = np.clip(action_np, self.env.action_space.low, self.env.action_space.high)
        return action_np

# --- Training Functions (Kept for reference, but not used in demo.py) ---
def train_step(model, train_loader, optimizer, loss_fn) -> float:
    model.train()
    train_loss = 0.
    device = next(model.parameters()).device
    for batch_idx, batch in enumerate(train_loader):
        state = batch['state'].to(device)
        action = batch['action'].to(device)
        next_state_gth = batch['next_state'].to(device)

        optimizer.zero_grad()
        # Adjust loss calculation based on whether it's SingleStepLoss or MultiStepLoss
        if isinstance(loss_fn, (SingleStepLoss, MultiStepLoss)):
             loss = loss_fn(model, state, action, next_state_gth)
        else: # Assume direct loss calculation if not one of the wrapper classes
             predicted_state = model(state, action)
             loss = loss_fn(predicted_state, next_state_gth)

        if torch.isnan(loss):
             print(f"Warning: NaN loss detected in train_step batch {batch_idx}. Skipping batch.")
             continue
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss / len(train_loader) if len(train_loader) > 0 else 0.0

def val_step(model, val_loader, loss_fn) -> float:
    model.eval()
    val_loss = 0.
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            state = batch['state'].to(device)
            action = batch['action'].to(device)
            next_state_gth = batch['next_state'].to(device)

            if isinstance(loss_fn, (SingleStepLoss, MultiStepLoss)):
                 loss = loss_fn(model, state, action, next_state_gth)
            else:
                 predicted_state = model(state, action)
                 loss = loss_fn(predicted_state, next_state_gth)

            if torch.isnan(loss):
                 print(f"Warning: NaN loss detected in val_step batch {batch_idx}. Skipping batch.")
                 continue
            val_loss += loss.item()
    return val_loss / len(val_loader) if len(val_loader) > 0 else 0.0

def train_model(model, train_dataloader, val_dataloader, loss_fn, num_epochs=100, lr=1e-3, device='cpu'):
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    pbar = tqdm(range(num_epochs))
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    for epoch_i in pbar:
        train_loss_i = train_step(model, train_dataloader, optimizer, loss_fn)
        val_loss_i = val_step(model, val_dataloader, loss_fn)
        pbar.set_description(f'Epoch {epoch_i+1}/{num_epochs} | Train Loss: {train_loss_i:.6f} | Val Loss: {val_loss_i:.6f}')
        train_losses.append(train_loss_i)
        val_losses.append(val_loss_i)

    return train_losses, val_losses
