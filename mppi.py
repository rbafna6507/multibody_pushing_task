import functools
import logging
import time

import numpy as np
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

logger = logging.getLogger(__name__)

def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))

def is_tensor_like(x):
    return torch.is_tensor(x) or type(x) is np.ndarray

def squeeze_n(v, n_squeeze):
    for _ in range(n_squeeze):
        v = v.squeeze(0)
    return v

def handle_batch_input(n):
    def _handle_batch_input(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            batch_dims = []
            original_shape = None
            for arg in args:
                if is_tensor_like(arg):
                    original_shape = arg.shape
                    if len(arg.shape) > n:
                        batch_dims = arg.shape[:-(n - 1)]
                        break
                    elif len(arg.shape) < n:
                        n_batch_dims_to_add = n - len(arg.shape)
                        batch_ones_to_add = [1] * n_batch_dims_to_add
                        new_args = []
                        for v in args:
                             if is_tensor_like(v) and len(v.shape) < n and len(v.shape)>0:
                                 try:
                                      new_args.append(v.view(*batch_ones_to_add, *v.shape))
                                 except Exception as e:
                                      print(f"Error reshaping arg in handle_batch_input (fewer dims): {e}, shape={v.shape}")
                                      new_args.append(v)
                             else:
                                 new_args.append(v)
                        args = tuple(new_args)

                        ret = func(*args, **kwargs)
                        if isinstance(ret, tuple):
                            return [squeeze_n(v, n_batch_dims_to_add) if is_tensor_like(v) else v for v in ret]
                        else:
                            return squeeze_n(ret, n_batch_dims_to_add) if is_tensor_like(ret) else ret

            if not batch_dims:
                return func(*args, **kwargs)

            num_batch_dims = len(batch_dims)
            new_args = []
            for v in args:
                 if is_tensor_like(v) and len(v.shape) > (n-1):
                     try:
                          flat_shape = (-1,) + v.shape[num_batch_dims:]
                          new_args.append(v.reshape(flat_shape))
                     except Exception as e:
                          print(f"Error reshaping arg in handle_batch_input (more dims): {e}, shape={v.shape}")
                          new_args.append(v)
                 else:
                     new_args.append(v)
            args = tuple(new_args)

            ret = func(*args, **kwargs)

            if isinstance(ret, tuple):
                reshaped_ret = []
                for v in ret:
                    if is_tensor_like(v) and len(v.shape) > 0:
                        try:
                            feature_shape = v.shape[1:]
                            final_shape = tuple(batch_dims) + feature_shape
                            reshaped_ret.append(v.reshape(final_shape))
                        except Exception as e:
                            print(f"Error reshaping result in handle_batch_input: {e}, shape={v.shape}, target={final_shape}")
                            reshaped_ret.append(v)
                    else:
                        reshaped_ret.append(v)
                return tuple(reshaped_ret)
            else:
                if is_tensor_like(ret) and len(ret.shape) > 0:
                     try:
                          feature_shape = ret.shape[1:]
                          final_shape = tuple(batch_dims) + feature_shape
                          return ret.reshape(final_shape)
                     except Exception as e:
                          print(f"Error reshaping single result in handle_batch_input: {e}, shape={ret.shape}, target={final_shape}")
                          return ret
                else:
                    return ret

        return wrapper
    return _handle_batch_input


class MPPI():
    def __init__(self, dynamics, running_cost, nx, noise_sigma, num_samples=100, horizon=15, device="cpu",
                 terminal_state_cost=None,
                 lambda_=1.,
                 noise_mu=None,
                 u_min=None,
                 u_max=None,
                 u_init=None,
                 U_init=None,
                 u_scale=1,
                 u_per_command=1,
                 step_dependent_dynamics=False,
                 rollout_samples=1,
                 rollout_var_cost=0,
                 rollout_var_discount=0.95,
                 sample_null_action=False,
                 noise_abs_cost=False):

        self.d = torch.device(device)
        if not torch.is_tensor(noise_sigma):
            noise_sigma = torch.tensor(noise_sigma, device=self.d)
        self.dtype = noise_sigma.dtype
        self.K = num_samples
        self.T = horizon

        self.nx = nx
        self.nu = 1 if len(noise_sigma.shape) == 0 else noise_sigma.shape[0]
        self.lambda_ = lambda_

        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype, device=self.d)
        elif not torch.is_tensor(noise_mu):
             noise_mu = torch.tensor(noise_mu, dtype=self.dtype, device=self.d)

        if u_init is None:
            self.u_init = torch.zeros_like(noise_mu)
        elif torch.is_tensor(u_init):
            self.u_init = u_init.to(dtype=self.dtype, device=self.d)
        else:
            self.u_init = torch.tensor(u_init, dtype=self.dtype, device=self.d)

        if self.nu == 1:
            noise_mu = noise_mu.view(-1)
            if len(noise_sigma.shape) < 2:
                noise_sigma = noise_sigma.view(-1, 1)

        self.u_min = u_min
        self.u_max = u_max
        self.u_scale = u_scale
        self.u_per_command = u_per_command
        if self.u_min is not None:
             if not torch.is_tensor(self.u_min): self.u_min = torch.tensor(self.u_min, dtype=self.dtype, device=self.d)
             else: self.u_min = self.u_min.to(device=self.d)
        if self.u_max is not None:
             if not torch.is_tensor(self.u_max): self.u_max = torch.tensor(self.u_max, dtype=self.dtype, device=self.d)
             else: self.u_max = self.u_max.to(device=self.d)
        if self.u_max is not None and self.u_min is None: self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None: self.u_max = -self.u_min

        self.noise_mu = noise_mu.to(self.d)
        self.noise_sigma = noise_sigma.to(self.d)
        try:
            self.noise_sigma_inv = torch.inverse(self.noise_sigma.float()).to(dtype=self.dtype)
        except Exception as e:
             print(f"Error inverting noise_sigma (device: {self.noise_sigma.device}, dtype: {self.noise_sigma.dtype}): {e}. Ensure it's non-singular.")
             try:
                 noise_sigma_stable = self.noise_sigma + torch.eye(self.nu, device=self.d, dtype=self.dtype) * 1e-6
                 self.noise_sigma_inv = torch.inverse(noise_sigma_stable.float()).to(dtype=self.dtype)
                 print("Used stabilized sigma for inversion.")
             except Exception as e_inner:
                 print(f"Stabilized inversion also failed: {e_inner}. Raising original error.")
                 raise e

        self.noise_dist_cpu = MultivariateNormal(self.noise_mu.cpu(), covariance_matrix=self.noise_sigma.cpu())

        if U_init is not None:
            if not torch.is_tensor(U_init): U_init = torch.tensor(U_init, dtype=self.dtype)
            self.U = U_init.to(device=self.d)
        else:
            _U_init_cpu = self.noise_dist_cpu.sample((self.T,))
            self.U = _U_init_cpu.to(dtype=self.dtype, device=self.d)

        self.step_dependency = step_dependent_dynamics
        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.sample_null_action = sample_null_action
        self.noise_abs_cost = noise_abs_cost
        self.state = None

        self.M = rollout_samples
        self.rollout_var_cost = rollout_var_cost
        self.rollout_var_discount = rollout_var_discount

        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None
        print(f"MPPI initialized on device: {self.d}")

    @handle_batch_input(n=2)
    def _dynamics(self, state, u, t):
        return self.F(state, u, t) if self.step_dependency else self.F(state, u)

    @handle_batch_input(n=2)
    def _running_cost(self, state, u):
        return self.running_cost(state, u)

    def command(self, state):
        with torch.no_grad():
            self.U = torch.roll(self.U, -1, dims=0)
            self.U[-1] = self.u_init
            return self._command(state)

    def _command(self, state):
        if not torch.is_tensor(state):
            state = torch.tensor(state, dtype=self.dtype)
        self.state = state.to(device=self.d)

        cost_total = self._compute_total_cost_batch()

        if not torch.isfinite(cost_total).all():
             print("Warning: Cost became NaN/Inf during MPPI command.")
             finite_mask = torch.isfinite(cost_total)
             if not finite_mask.any():
                 print("ERROR: All costs are NaN/Inf. Returning previous U[0] action.")
                 action = self.U[:self.u_per_command].clone()
                 if self.u_per_command == 1: action = action[0]
                 return action

             print(f"  Proceeding with {finite_mask.sum()} finite costs only.")
             cost_total_finite = cost_total[finite_mask]
             noise_finite = self.noise[finite_mask]

             beta = torch.min(cost_total_finite)
             try:
                  cost_total_non_zero_finite = _ensure_non_zero(cost_total_finite, beta, 1 / self.lambda_)
             except Exception as e:
                  print(f"Error in _ensure_non_zero with finite costs: {e}. Using uniform weights.")
                  cost_total_non_zero_finite = torch.ones_like(cost_total_finite)

             eta = torch.sum(cost_total_non_zero_finite)
             if eta < 1e-10 or not torch.isfinite(eta):
                  print("Warning: eta is near zero or non-finite after filtering. Using uniform weights.")
                  omega_finite = torch.ones_like(cost_total_finite, device=self.d) / cost_total_finite.numel()
             else:
                  omega_finite = (1. / eta) * cost_total_non_zero_finite

             delta_U = torch.zeros_like(self.U)
             for t in range(self.T):
                  weighted_noise = torch.sum(omega_finite.view(-1, 1) * noise_finite[:, t, :], dim=0)
                  delta_U[t] = weighted_noise
             self.U += delta_U

             self.omega = torch.zeros_like(cost_total)
             self.cost_total = cost_total

        else:
            beta = torch.min(cost_total)
            try:
                 self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)
            except Exception as e:
                  print(f"Error in _ensure_non_zero: {e}. Using uniform weights.")
                  self.cost_total_non_zero = torch.ones_like(cost_total)

            eta = torch.sum(self.cost_total_non_zero)
            if eta < 1e-10 or not torch.isfinite(eta):
                  print("Warning: eta is near zero or non-finite. Using uniform weights.")
                  self.omega = torch.ones_like(self.cost_total_non_zero, device=self.d) / self.cost_total_non_zero.numel()
            else:
                 self.omega = (1. / eta) * self.cost_total_non_zero

            delta_U = torch.zeros_like(self.U)
            for t in range(self.T):
                 weighted_noise = torch.sum(self.omega.view(-1, 1) * self.noise[:, t, :], dim=0)
                 delta_U[t] = weighted_noise
            self.U += delta_U
            self.cost_total = cost_total

        action = self.U[:self.u_per_command].clone()
        if self.u_per_command == 1:
            action = action[0]
        return action

    def reset(self):
        _U_cpu = self.noise_dist_cpu.sample((self.T,))
        self.U = _U_cpu.to(dtype=self.dtype, device=self.d)

    def _compute_rollout_costs(self, perturbed_actions):
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, device=self.d, dtype=self.dtype)
        cost_samples = torch.zeros((self.M, K), device=self.d, dtype=self.dtype)
        cost_var = torch.zeros(K, device=self.d, dtype=self.dtype)

        if len(self.state.shape) == 1:
             state_K = self.state.unsqueeze(0).expand(K, -1)
        elif self.state.shape[0] == K:
             state_K = self.state
        elif self.state.shape[0] == 1:
             state_K = self.state.repeat(K, 1)
        else:
             raise ValueError(f"Initial state shape {self.state.shape} incompatible with K={K}")
        state_K = state_K.to(self.d)

        state_MK = state_K.unsqueeze(0).repeat(self.M, 1, 1)

        states = torch.zeros((self.M, K, T, self.nx), device=self.d, dtype=self.dtype)
        actions = torch.zeros((self.M, K, T, self.nu), device=self.d, dtype=self.dtype)

        current_state_MK = state_MK

        for t in range(T):
            u_MK = self.u_scale * perturbed_actions[:, t].unsqueeze(0).repeat(self.M, 1, 1)
            current_state_flat = current_state_MK.view(-1, self.nx)
            u_flat = u_MK.view(-1, self.nu)
            next_state_flat = self._dynamics(current_state_flat, u_flat, t)
            current_state_MK = next_state_flat.view(self.M, K, self.nx)
            c_flat = self._running_cost(current_state_flat, u_flat)
            c = c_flat.view(self.M, K)
            cost_samples += c
            states[:, :, t, :] = current_state_MK
            actions[:, :, t, :] = u_MK

        if self.M > 1:
             cost_var = torch.var(cost_samples, dim=0, unbiased=True)

        c_terminal_per_K = torch.zeros(K, device=self.d, dtype=self.dtype)
        if self.terminal_state_cost:
             mean_states_KT = states.mean(dim=0)
             mean_actions_KT = actions.mean(dim=0)
             c_terminal_per_K = self.terminal_state_cost(mean_states_KT, mean_actions_KT)

        cost_total = cost_samples.mean(dim=0)
        cost_total += c_terminal_per_K
        cost_total += cost_var * self.rollout_var_cost

        return cost_total, states.mean(dim=0), actions.mean(dim=0)


    def _compute_total_cost_batch(self):
        _noise_cpu = self.noise_dist_cpu.sample((self.K, self.T))
        self.noise = _noise_cpu.to(dtype=self.dtype, device=self.d)

        self.U = self.U.to(self.d)
        self.perturbed_action = self.U + self.noise

        if self.sample_null_action:
            self.perturbed_action[self.K - 1] = 0

        self.perturbed_action = self._bound_action(self.perturbed_action)
        self.noise = self.perturbed_action - self.U

        if self.noise_abs_cost:
            noise_term = torch.abs(self.noise)
        else:
            noise_term = self.noise

        action_cost_term = torch.matmul(noise_term, self.noise_sigma_inv)
        perturbation_cost = torch.sum(self.U.unsqueeze(0) * action_cost_term, dim=(1, 2))

        self.cost_total, self.states, self.actions = self._compute_rollout_costs(self.perturbed_action)
        self.cost_total += perturbation_cost
        return self.cost_total

    def _bound_action(self, action):
        if self.u_max is not None and self.u_min is not None:
            u_max_b = self.u_max.view(1, 1, -1)
            u_min_b = self.u_min.view(1, 1, -1)
            action = torch.max(torch.min(action, u_max_b), u_min_b)
        elif self.u_max is not None:
             action = torch.min(action, self.u_max.view(1, 1, -1))
        elif self.u_min is not None:
             action = torch.max(action, self.u_min.view(1, 1, -1))
        return action

    def get_rollouts(self, state, num_rollouts=1):
        with torch.no_grad():
            if not torch.is_tensor(state): state = torch.tensor(state)
            state = state.to(dtype=self.dtype, device=self.d)

            state = state.view(-1, self.nx)
            batch_size = state.size(0)

            if batch_size == 1 and num_rollouts > 1:
                state = state.repeat(num_rollouts, 1)
                current_batch_size = num_rollouts
            else:
                 if num_rollouts > 1: print("Warning: get_rollouts using num_rollouts=1 per state for batched input.")
                 current_batch_size = batch_size

            T = self.U.shape[0]
            states = torch.zeros((current_batch_size, T + 1, self.nx), dtype=self.U.dtype, device=self.d)
            states[:, 0] = state

            U_expanded = self.U.unsqueeze(0).expand(current_batch_size, -1, -1)

            current_states = states[:, 0]
            for t in range(T):
                u_t = self.u_scale * U_expanded[:, t, :]
                current_states = self._dynamics(current_states.view(current_batch_size, -1), u_t.view(current_batch_size, -1), t)
                states[:, t + 1] = current_states.view(current_batch_size, -1)

            return states[:, 1:]

def run_mppi(mppi, env, retrain_dynamics, retrain_after_iter=50, iter=1000, render=True):
    dataset = torch.zeros((retrain_after_iter, mppi.nx + mppi.nu), dtype=mppi.dtype, device=mppi.d)
    total_reward = 0
    state = env.reset()
    for i in range(iter):
        state_tensor = torch.tensor(state, dtype=mppi.dtype, device=mppi.d)
        command_start = time.perf_counter()
        action = mppi.command(state_tensor)
        elapsed = time.perf_counter() - command_start

        action_np = action.detach().cpu().numpy()
        action_np = np.clip(action_np, env.action_space.low, env.action_space.high)

        try:
            s_next, r, done, _ = env.step(action_np)
            total_reward += r
            logger.debug("Iter: {} Action: {} Cost: {:.4f} Time: {:.5f}s".format(i, action_np, -r, elapsed))

            di = i % retrain_after_iter
            dataset[di, :mppi.nx] = state_tensor
            dataset[di, mppi.nx:] = action

            state = s_next

            if done:
                 logger.info(f"Episode finished at step {i}, resetting environment.")
                 state = env.reset()
                 mppi.reset()

            if (i + 1) % retrain_after_iter == 0 and i > 0 and callable(retrain_dynamics):
                logger.info(f"Retraining dynamics at iteration {i+1}")
                retrain_dynamics(dataset)

            if render:
                if hasattr(env, 'render_frame'): env.render_frame()
                elif hasattr(env, 'render'): env.render()

        except Exception as e:
            print(f"Error in MPPI loop at iteration {i}: {e}")
            break

    return total_reward, dataset
