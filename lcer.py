import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import utils
from encoder import make_encoder

LOG_FREQ = 10000

def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters,
    ):
        super().__init__()
        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
        self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False,
    ):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        # constrain log_std inside [log_std_min, log_std_max]
        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""
    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""
    def __init__(
        self, obs_shape, action_shape, hidden_dim, encoder_type,
        encoder_feature_dim, num_layers, num_filters,
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):
        # detach_encoder allows to stop gradient propogation to encoder
        obs = self.encoder(obs, detach=detach_encoder)
        self.obs = obs
        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class LCER(nn.Module):
    """
    LCER
    """

    def __init__(self, obs_shape, z_dim, act_dim, batch_size, critic, critic_target, alg_type="lcer"):
        super(LCER, self).__init__()
        self.batch_size = batch_size

        self.encoder = critic.encoder

        self.encoder_target = critic_target.encoder 

        self.transition_model = nn.Sequential(*create_mlp(z_dim + act_dim, z_dim, [128, 128], nn.ReLU))
        self.backward_model = nn.Sequential(*create_mlp(z_dim, z_dim, [128, 128], nn.ReLU))
        self.predR_model = nn.Sequential(*create_mlp(z_dim + act_dim, 1, [128], nn.ReLU))

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))
        self.prediction = nn.Sequential(*create_mlp(z_dim, z_dim, [128], nn.ReLU))
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.alg_type = alg_type
        if 'lcer_beta' in self.alg_type:
          self.beta = float(self.alg_type.split('_')[-1])
        else:
          self.beta = 0
        self.score_net = nn.Sequential(*create_mlp(2 * z_dim, 1, [128], nn.ReLU))
        self.W2 = nn.Parameter(torch.rand(z_dim, z_dim))

        self.score_type = 'net'
        self.mi_loss_type = 'jsd'
        def score_fn(x, y):
          if self.score_type == 'pred':
            phi_x = self.prediction(x)
            phi_y = self.prediction(y)
            return (phi_x * phi_y).sum(-1)
          elif self.score_type == 'sep_pred':
            phi_x = self.prediction(x)
            phi_y = y
            return (phi_x * phi_y).sum(-1)
          elif self.score_type == 'net':
            return self.score_net(torch.cat([x, y], -1))
          elif self.score_type == 'bilinear':
            # y: (B,Z, 1), W2: (Z,Z), Wy--> (B,Z,1)
            Wy = torch.matmul(self.W2, y.unsqueeze(-1)) 
            # Wy: (B,Z,1), x:(B,1,Z), xWy:(B,1)
            xWy = torch.matmul(x.unsqueeze(1), Wy).squeeze(-1)
            return xWy
        def mi_est_fn(x, y):
          if self.mi_loss_type == 'jsd':
            loss = 2 * math.log(2) - F.softplus(-x) - F.softplus(y)
          elif self.mi_loss_type == 'nwj':
            loss = x.mean() - math.exp(-1) * torch.exp(y).mean()
          elif self.mi_loss_type == 'mine':
            loss = x.mean() - torch.log(torch.exp(y).mean())
          return loss.mean()
        self.score_fn = score_fn
        self.mi_est_fn = mi_est_fn
        assert self.beta >= 0  and self.beta <= 1
        self.apply(weight_init)

    def encode(self, x, detach=False, ema=False):
        """
        Encoder: z_t = e(x_t)
        :param x: x_t, x y coordinates
        :return: z_t, value in r2
        """
        if ema:
            with torch.no_grad():
                z_out = self.encoder_target(x)
        else:
            z_out = self.encoder(x)
        if detach:
            z_out = z_out.detach()
        return z_out

    def compute_logits(self, z_a, z_pos):
        """
        Uses logits trick for LCER:
        - compute (B,B) matrix z_a (W z_pos.T)
        - positives are all diagonal elements
        - negatives are all other elements
        - to compute loss use multiclass cross entropy with identity matrix for labels
        """
        W = self.W
        Wz = torch.matmul(W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]
        return logits

    def compute_single_loss(self, x, y, label):
        logits = self.compute_logits(x, y)
        loss = self.cross_entropy_loss(logits, label)
        return loss

    def compute_loss(self, kwargs, neg=False):
        obs  = kwargs['obs'] 
        fobs = kwargs['fobs'] 
        frew = kwargs['frew']
        fact = kwargs['fact']
        def select_idx(tensor, i):
          shape = tensor.shape
          if len(shape) == 3:
            return tensor[:, i, :]
          if len(shape) == 4:
            return tensor[:, i, :, :]
          if len(shape) == 5:
            return tensor[:, i, :, :, :]
          assert False
        B = fobs.shape[0]
        K = fobs.shape[1] - 1
        labels = torch.arange(B).long().to(fobs.device)
        loss = 0

        latent = self.encode(obs, ema=False)
        fuse_latent = latent

        future_latent = []
        pred_latent = []
        fuse_pred_latent = []
        fuse_fact = utils.shuffle(fact, 0)
        for k in range(K + 1):
          latent2 = self.encode(select_idx(fobs, k), ema=True)
          pred_latent.append(latent)
          fuse_pred_latent.append(fuse_latent)
          future_latent.append(latent2)
          if k < K:
              latent = self.transition_model(torch.cat([latent, select_idx(fact, k)], -1))
              fuse_latent = self.transition_model(torch.cat([fuse_latent, select_idx(fuse_fact, k)], -1))
        mi_loss = 0
        if self.beta < 1:
          for k in range(K+1):
            mi_loss = mi_loss + self.compute_single_loss(pred_latent[k], future_latent[k], labels)
        cond_loss = 0
        use_K0 = False
        if self.beta > 0:
          for k in range(K+1):
            if k == 0 and not use_K0:
              continue
            score = self.score_fn(pred_latent[k], future_latent[k])
            fuse_score = self.score_fn(fuse_pred_latent[k], future_latent[k])
            loss_k = -self.mi_est_fn(score, fuse_score)
            cond_loss = cond_loss + loss_k
        loss = mi_loss * (1-self.beta) + cond_loss * self.beta
        return loss, mi_loss, cond_loss

def create_mlp(
    input_dim, output_dim, net_arch, activation_fn, squash_output=False):
    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules


class LCERAgent(object):

    """LCER representation learning with SAC."""
    def __init__(
        self,
        K,
        obs_shape,
        action_shape,
        device,
        use_drq = False,
        alg_type='lcer',
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        num_layers=4,
        num_filters=32,
        cpc_update_freq=1,
        log_interval=100,
        detach_encoder=False,
        use_cpc=True,
        rd_crop = False,
        curl_latent_dim=128
    ):
        self.use_drq = use_drq
        self.alg_type = alg_type
        self.rd_crop=rd_crop
        self.K = K
        self.use_cpc = use_cpc
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.curl_latent_dim = curl_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type
        act_dim = np.product(action_shape)

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters,
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters,
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters,
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        # tie encoders between actor and critic, and LCER and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)
        
        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.encoder_type == 'pixel':
            self.LCER = LCER(obs_shape, encoder_feature_dim, act_dim,
                        self.curl_latent_dim, self.critic,self.critic_target,
                        alg_type=self.alg_type,
                        ).to(self.device)

            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            self.cpc_optimizer = torch.optim.Adam(
                self.LCER.parameters(), lr=encoder_lr
            )
        self.cross_entropy_loss = nn.CrossEntropyLoss()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type == 'pixel':
            self.LCER.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)
 
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic_drq(self, obs, obs_aug, action, reward, next_obs,
                      next_obs_aug, not_done, L, step):
        with torch.no_grad():
            _, next_action, log_prob, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + (not_done * self.discount * target_V)


            _, next_action_aug, log_prob_aug, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs_aug,
                                                      next_action_aug)
            target_V = torch.min(
                target_Q1, target_Q2) - self.alpha.detach() * log_prob_aug
            target_Q_aug = reward + (not_done * self.discount * target_V)

            target_Q = (target_Q + target_Q_aug) / 2

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
            current_Q2, target_Q)

        Q1_aug, Q2_aug = self.critic(obs_aug, action)

        critic_loss += F.mse_loss(Q1_aug, target_Q) + F.mse_loss(
            Q2_aug, target_Q)

        if step % self.log_interval == 0:
          L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(
            obs, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
            (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:                                    
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
    def update_cpc(self, cpc_kwargs, L, step, log=True):
        loss,mi_loss, cond_loss = self.LCER.compute_loss(cpc_kwargs)
        if step % self.log_interval == 0 and log:
            L.log('train/curl_loss', loss, step)
            L.log('train/mi_loss', mi_loss, step)
            L.log('train/cond_loss', cond_loss, step)
        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.cpc_optimizer.step()
    def update(self, replay_buffer, L, step):
        if self.encoder_type == 'pixel':
            obs, action, reward, next_obs, not_done, obs1, next_obs1 = replay_buffer.sample(self.rd_crop)
        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()
    
        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        if self.use_drq:
          self.update_critic_drq(obs, obs1, action, reward, next_obs, next_obs1, not_done, L, step)
        else:
          self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.cpc_update_freq == 0 and self.encoder_type == 'pixel' and self.use_cpc:
            for k in range(max(0, 1)):
              cpc_kwargs = replay_buffer.sample_cpc(self.rd_crop)
              self.update_cpc(cpc_kwargs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )

    def save_curl(self, model_dir, step):
        torch.save(
            self.LCER.state_dict(), '%s/curl_%s.pt' % (model_dir, step)
        )

    def load_curl(self, model_dir, step):
        self.LCER.load_state_dict(
            torch.load('%s/curl_%s.pt' % (model_dir, step))
        )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
 
