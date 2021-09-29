import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random
from torch.utils.data import Dataset, DataLoader
import time
from skimage.util.shape import view_as_windows
from collections import deque
from torch import distributions as pyd
import kornia
import random

def shuffle(x, dim=0):
  idx = np.arange(x.shape[dim])
  np.random.shuffle(idx)
  idx = torch.as_tensor(idx, device=x.device).long()
  return torch.index_select(x, dim, idx)

class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2**bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2**(8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, action_shape, capacity, batch_size, device,image_size=84,transform=None):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
        
        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, action, reward, next_obs, done):
       
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):
        
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
        
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones


    def sample_cpc(self):

        start = time.time()
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
      
        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()

        obses = random_crop(obses, self.image_size)
        next_obses = random_crop(next_obses, self.image_size)
        pos = random_crop(pos, self.image_size)
    
        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()
        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos, next_obs=next_obses, actions=actions,
                          time_anchor=None, time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx],
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity 

class ReplayBufferFuture(ReplayBuffer):
    def __init__(self, K, obs_shape, action_shape, capacity, batch_size, device,image_size=84,transform=None):
      super().__init__(obs_shape, action_shape, capacity, batch_size,device,image_size,transform)
      K = max(K, 0)
      self.K = K
      obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8
      self.future_obs = np.empty((capacity, K + 1, *obs_shape), dtype=obs_dtype)
      self.future_actions = np.empty((capacity, K + 1, *action_shape), dtype=np.float32)
      self.future_rewards = np.empty((capacity, K + 1, 1), dtype=np.float32)
      image_pad = 4
      self.aug_trans = nn.Sequential(
          nn.ReplicationPad2d(image_pad),
          kornia.augmentation.RandomCrop((image_size, image_size)))

      self.deque_obs = deque(maxlen=K+1)
      self.deque_act = deque(maxlen=K+1)
      self.deque_rew = deque(maxlen=K+1)

      self.fidx = 0
      self.ffull = False

    def add(self, obs, action, reward, next_obs, done, ep_step, real_done):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)
        
        self.deque_obs.append(obs)
        self.deque_rew.append(np.array([reward]))
        self.deque_act.append(action)
        if len(self.deque_obs) == self.K + 1:
          np.copyto(self.future_obs[self.fidx],     np.asarray(self.deque_obs))
          np.copyto(self.future_actions[self.fidx], np.asarray(self.deque_act))
          np.copyto(self.future_rewards[self.fidx], np.asarray(self.deque_rew))
          self.fidx = (self.fidx + 1) % self.capacity
          self.ffull = self.ffull or self.fidx == 0
        if real_done:
          self.deque_obs.clear()
          self.deque_act.clear()
          self.deque_rew.clear()
        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0
    def post_process(self, x, is_obs, rd_crop):
        x = torch.as_tensor(x, device =self.device).float()
        if is_obs and rd_crop:
          x_shape = x.shape
          if len(x_shape) == 5: # (B, K+1, C, H, W)
            x = x.reshape((np.product(x_shape[0:2]), *x_shape[2:]))
          x = self.aug_trans(x)
          if len(x_shape) == 5: # (B, K+1, C, H, W)
            x = x.reshape(*x_shape[0:2], *x.shape[1:])
        return x 
    def sample(self, rd_crop=True):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )
        obs       = self.obses[idxs]
        next_obs  = self.next_obses[idxs]
        act       = self.actions[idxs]
        rew       = self.rewards[idxs]
        not_dones = self.not_dones[idxs]
        obs0, next_obs0 = \
              (self.post_process(v, True, rd_crop) for v in [obs, next_obs])
        obs1, next_obs1 = \
              (self.post_process(v, True, rd_crop) for v in [obs, next_obs])
        act, rew, not_dones = \
              (self.post_process(v, False, None) for v in [act, rew, not_dones])
        return obs0, act, rew, next_obs0, not_dones, obs1, next_obs1

    def sample_cpc(self, rd_crop=False):
        fidxs = np.random.randint(
            0, self.capacity if self.ffull else self.fidx, size=self.batch_size
        )
        fobs       = self.future_obs[fidxs]
        fact       = self.future_actions[fidxs]
        frew       = self.future_rewards[fidxs]

        obs        = self.post_process(fobs[:, 0, :, :, :], True, rd_crop)        

        fobs       = self.post_process(fobs, True, rd_crop)
        fact, frew = \
              (self.post_process(v, False, None) for v in [fact, frew])
        kwargs = dict(obs=obs, fobs=fobs, fact=fact, frew=frew)
        return kwargs
    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx],
            self.future_obs[self.last_save:self.idx],
            self.future_actions[self.last_save:self.idx],
            self.future_rewards[self.last_save:self.idx],
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.future_obs[start:end] = payload[5]
            self.future_actions[start:end] = payload[6]
            self.future_rewards[start:end] = payload[7]
            self.idx = end


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def random_crop(imgs, output_size, rd_crop=False):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # disable random_crop:
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # disable random_crop:
    if not rd_crop:
      w1 = np.array([int(crop_max / 2)] * n)
      h1 = np.array([int(crop_max / 2)] * n)

    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0,:,:, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs

def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h)//2
    left = (w - new_w)//2

    image = image[:, top:top + new_h, left:left + new_w]
    return image


class SquashedNormal(pyd.transformed_distribution.TransformedDistribution):
    def __init__(self, loc, scale):
        self.loc = loc
        self.scale = scale

        self.base_dist = pyd.Normal(loc, scale)
        transforms = [TanhTransform()]
        super().__init__(self.base_dist, transforms)

    @property
    def mean(self):
        mu = self.loc
        for tr in self.transforms:
            mu = tr(mu)
        return mu
