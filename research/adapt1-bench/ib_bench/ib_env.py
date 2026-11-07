"""Siemens Industrial Benchmark (industrial_benchmark_python 2.0) with the small shims this NumPy needs, plus the
drifting-setpoint mode and a Gymnasium-API wrapper for the local comparators.

Observation (6): setpoint, velocity, gain, shift, fatigue, consumption. Reward: -(3*fatigue + consumption) ("classic").
Actions: the benchmark's own 27 discrete steerings {-1,0,1}^3 for (velocity, gain, shift), or continuous [-1,1]^3.
Episodes: EP_LEN steps from the benchmark's fixed start (v=g=h=50) with a fresh noise seed per reset.
"""
import numpy as np
np.float = float          # the 2017 package uses removed NumPy aliases
np.bool8 = np.bool_
import warnings
warnings.filterwarnings('ignore')
from industrial_benchmark_python.IBGym import IBGym
from industrial_benchmark_python.IDS import IDS

OBS_NAMES = ['setpoint', 'velocity', 'gain', 'shift', 'fatigue', 'consumption']
OBS_SCALE = np.array([100.0, 100.0, 100.0, 100.0, 100.0, 500.0])   # fixed scaling for Adapt-1 context features
R_MIN, R_MAX = -800.0, -100.0                                      # declared reward bounds (measured: zero policy ~-170, random ~-260 at p=50)
EP_LEN = 200


class IB(IBGym):
    """IBGym with an optional drifting setpoint (IDS stationary_p=False) and a fixed episode length."""
    def __init__(self, setpoint=50, drift=False, action_type='discrete', seed=0, ep_len=EP_LEN):
        self.drift = drift
        super().__init__(setpoint=setpoint, reward_type='classic', action_type=action_type, observation_type='classic',
                         reset_after_timesteps=ep_len, init_seed=seed)

    def reset(self):
        # mirrors IBGym.reset, with stationary_p passed through to IDS
        self.IB = IDS(self.setpoint, stationary_p=not self.drift, inital_seed=self.init_seed)
        self.init_seed = int(np.random.randint(0, 100000))
        self.observation = self.IB.visibleState()[:-2]
        self.info = self._markovian_state()
        self.reward = -self.IB.state['cost']
        self.delta_reward = 0
        self.smoothed_reward = self.reward
        self.env_steps = 0
        self.done = False
        return np.array(self.observation, dtype=np.float64)

    def step(self, action):
        obs, r, done, info = super().step(action)
        return np.array(obs, dtype=np.float64), float(r), bool(done), info


def scaled(obs):
    return {k: float(v) for k, v in zip(OBS_NAMES, np.asarray(obs, dtype=float) / OBS_SCALE)}


# ---- Gymnasium wrapper for stable-baselines3 -------------------------------------------------------------------------
import gymnasium as gym


class IBGymnasium(gym.Env):
    def __init__(self, setpoint=50, drift=False, discrete=False, seed=0, ep_len=EP_LEN):
        self.env = IB(setpoint, drift, 'discrete' if discrete else 'continuous', seed, ep_len)
        self.discrete = discrete
        self.action_space = gym.spaces.Discrete(27) if discrete else gym.spaces.Box(-1.0, 1.0, (3,), dtype=np.float32)
        self.observation_space = gym.spaces.Box(np.zeros(6, dtype=np.float32), np.array([100, 100, 100, 100, 1000, 1000], dtype=np.float32))

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        return self.env.reset().astype(np.float32), {}

    def step(self, action):
        a = int(action) if self.discrete else np.asarray(action, dtype=float)
        obs, r, done, info = self.env.step(a)
        return obs.astype(np.float32), r, False, done, {}
