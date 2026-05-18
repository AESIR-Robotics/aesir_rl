"""
PPO trainer for the Aesir rescue robot.

Changes vs. the ANYmal C base script:
  * obs_len = 74 (qpos:29 + qvel:28 + sensordata:17), act_len = 12
  * Env is the door-traversal task in aesir_env.py
  * compute_action() returns clean tanh-squashed actions in [-1, 1].
    The ANYmal-specific hard clamps (action_clamped[0][0]*0.6-0.1 ...)
    have been removed so ActionAdapter.from_policy_output() receives a
    well-formed 12-vector.
  * Save / video threshold tuned for the door reward scale.
"""
import numpy as np
import wandb
import mediapy as media
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler

from env.aesir_env import Env

# Uncomment if a GPU is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT      = "AIDL-PPO-AESIR"
OBS_LEN      = 74      # qpos(29) + qvel(28) + sensordata(17)
ACT_LEN      = 12      # see ActionAdapter layout
SAVE_REWARD  = 50.0    # save model when running reward exceeds this

wandb.login()


# --------------------------------------------------------------------------- #
#                                  Agent                                       #
# --------------------------------------------------------------------------- #
class Agent(nn.Module):
    def __init__(self, obs_len, act_len):
        super().__init__()
        self.obs_len = obs_len
        self.act_len = act_len

        self.mlp = nn.Sequential(
            nn.Linear(obs_len, 128),
            nn.Tanh(),
            nn.Linear(128, 128),
            nn.Tanh(),
        )
        self.actor = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, act_len),
        )
        self.critic = nn.Sequential(
            nn.Linear(128, 128),
            nn.Tanh(),
            nn.Linear(128, 1),
        )

    def forward(self, state):
        out           = self.mlp(state)
        action_scores = self.actor(out)
        state_value   = self.critic(out)
        return action_scores, state_value

    def compute_action(self, state, action_std):
        """Sample an action and return it clean in [-1, 1].

        The ANYmal-specific scale-and-shift lines have been removed; the
        ActionAdapter expects raw values in [-1, 1] and will handle every
        per-DOF scaling itself.
        """
        state                = torch.from_numpy(state).float().unsqueeze(0)
        probs, state_value   = self(state)
        probs                = torch.tanh(probs)

        action_var = torch.full((self.act_len,), action_std * action_std)
        cov_mat    = torch.diag(action_var).unsqueeze(dim=0)
        m          = MultivariateNormal(probs, cov_mat)
        action     = m.sample()

        # Squash final sample to [-1, 1] — no ANYmal-specific clamps
        action_clamped = torch.tanh(action)

        return (
            action_clamped.detach().numpy(),
            m.log_prob(action_clamped).detach().numpy(),
            state_value.detach(),
        )


# --------------------------------------------------------------------------- #
#                              Replay memory                                  #
# --------------------------------------------------------------------------- #
transition = np.dtype([
    ('s',      np.float64, (OBS_LEN,)),
    ('a',      np.float64, (ACT_LEN,)),
    ('a_logp', np.float64, (ACT_LEN,)),
    ('r',      np.float64),
    ('s_',     np.float64, (OBS_LEN,)),
])


class ReplayMemory:
    def __init__(self, capacity):
        self.buffer_capacity = capacity
        self.buffer          = np.empty(capacity, dtype=transition)
        self.counter         = 0

    def store(self, tr):
        self.buffer[self.counter] = tr
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        return False


# --------------------------------------------------------------------------- #
#                              PPO update                                     #
# --------------------------------------------------------------------------- #
def train(policy, optimizer, memory, hparams, action_std):
    gamma      = hparams['gamma']
    ppo_epoch  = hparams['ppo_epoch']
    batch_size = hparams['batch_size']
    clip_param = hparams['clip_param']
    c1         = hparams['c1']
    c2         = hparams['c2']

    s   = torch.tensor(memory.buffer['s'],  dtype=torch.float)
    a   = torch.tensor(memory.buffer['a'],  dtype=torch.float)
    r   = torch.tensor(memory.buffer['r'],  dtype=torch.float).view(-1, 1)
    s_  = torch.tensor(memory.buffer['s_'], dtype=torch.float)

    old_a_logp = torch.tensor(memory.buffer['a_logp'], dtype=torch.float).view(-1, 1)
    action_var = torch.full((ACT_LEN,), action_std * action_std)
    cov_mat    = torch.diag(action_var).unsqueeze(dim=0)

    with torch.no_grad():
        target_v = r + gamma * policy(s_)[1]
        adv      = target_v - policy(s)[1]

    for _ in range(ppo_epoch):
        for index in BatchSampler(
            SubsetRandomSampler(range(memory.buffer_capacity)),
            batch_size,
            False,
        ):
            probs, _ = policy(s[index])
            dist     = MultivariateNormal(probs, cov_mat)
            entropy  = dist.entropy()
            a_logp   = dist.log_prob(a[index]).unsqueeze(dim=1)

            ratio = torch.exp(a_logp - old_a_logp[index])
            surr1 = ratio * adv[index]
            surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * adv[index]

            policy_loss = torch.min(surr1, surr2).mean()
            value_loss  = F.smooth_l1_loss(policy(s[index])[1], target_v[index])
            entropy     = entropy.mean()

            loss = -policy_loss + c1 * value_loss - c2 * entropy

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return -policy_loss.item(), value_loss.item(), entropy.item(), ratio.mean().item()


# --------------------------------------------------------------------------- #
#                              Evaluation                                     #
# --------------------------------------------------------------------------- #
def test(action_std, env, policy, episode, render=True):
    state                  = env.reset()
    ep_reward, done        = 0, False
    counter                = 0
    reward_list            = []
    cumulative_reward_list = []
    time_list              = []

    while not done:
        action, _, _              = policy.compute_action(state, action_std)
        state, reward, done       = env.step(action, render=render)
        reward_list.append(reward)
        time_list.append(counter * 0.002 * 5)   # TIMESTEP * MUJOCO_STEPS
        ep_reward += reward
        cumulative_reward_list.append(ep_reward)
        counter   += 1

    video_path = env.close(episode, ep_reward)
    try:
        wandb.log({
            "Video eval": wandb.Video(video_path, fps=4, format="mp4"),
        })
    except Exception as e:
        print(f"[warn] could not log video to W&B: {e}")

    plt.figure()
    plt.plot(time_list, reward_list,            label='instant')
    plt.plot(time_list, cumulative_reward_list, label='cumulative')
    plt.xlabel('Time (s)')
    plt.ylabel('Reward')
    plt.title(f'Episode {episode}')
    plt.legend()
    wandb.log({"Reward eval": plt})

    return ep_reward


# --------------------------------------------------------------------------- #
#                       Sweep + main training entry                           #
# --------------------------------------------------------------------------- #
sweep_configuration = {
    "name":   "ppo_aesir_sweep",
    "method": "bayes",
    "metric": {"name": "avg_reward", "goal": "maximize"},
    "parameters": {
        "lr":           {"distribution": "uniform",     "min": 1e-5, "max": 1e-4},
        "ppo_epoch":    {"distribution": "int_uniform", "min": 40,   "max": 60},
        "c2":           {"distribution": "uniform",     "min": 0.001, "max": 0.01},
        "replay_size":  {"distribution": "int_uniform", "min": 6000, "max": 10000},
        "std_init":     {"distribution": "uniform",     "min": 1.0,  "max": 1.1},
        "std_min":      {"distribution": "uniform",     "min": 0.5,  "max": 0.8},
    },
}


def train_or_sweep(is_sweep=False):
    hparams = {
        'gamma':        0.99,
        'log_interval': 50,
        'num_episodes': 15000,
        'lr':           1e-5,
        'clip_param':   0.1,
        'ppo_epoch':    48,
        'replay_size':  6400,
        'batch_size':   128,
        'c1':           1.0,
        'c2':           0.001,
        'std_init':     1.0,
        'std_min':      0.6,
    }

    wandb.init(project=PROJECT)
    if is_sweep:
        hparams.update(wandb.config)
        print('Params updated from sweep config')

    env = Env()
    print(f"Env created: obs_len={env.obs_len}, act_len={env.act_len}")

    # Reproducibility
    seed = 0
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)

    policy    = Agent(OBS_LEN, ACT_LEN)
    optimizer = torch.optim.Adam(policy.parameters(), lr=hparams['lr'])
    memory    = ReplayMemory(hparams['replay_size'])

    # To resume from a checkpoint, uncomment:
    # policy    = torch.load('./Policy_Reward_XXX.pt')
    # optimizer = torch.load('./Optimizer_Reward_XXX.pt')

    action_std_decay = -(hparams['std_min'] - hparams['std_init']) \
                       * hparams['log_interval'] / hparams['num_episodes']
    action_std       = hparams['std_init']

    target_reward = 300.0    # reaching door open + crossed bonus + shaping
    print(f"Target running reward to stop training: {target_reward}")

    running_reward = -100.0
    saving_reward  = SAVE_REWARD

    for i_episode in range(hparams['num_episodes']):
        state, ep_reward, done = env.reset(), 0.0, False

        while not done:
            action, a_logp, _   = policy.compute_action(state, action_std)
            next_state, reward, done = env.step(action, render=False)

            if memory.store((state, action, a_logp, reward, next_state)):
                policy_loss, value_loss, avg_entropy, ratio = train(
                    policy, optimizer, memory, hparams, action_std
                )
                wandb.log({
                    'policy_loss':  policy_loss,
                    'value_loss':   value_loss,
                    'avg_reward':   running_reward,
                    'avg_entropy':  avg_entropy,
                    'ratio':        ratio,
                })

            state      = next_state
            ep_reward += reward

        running_reward = round(0.05 * ep_reward + 0.95 * running_reward, 2)

        if i_episode % hparams['log_interval'] == 0:
            print(f'Episode {i_episode}\tLast reward: {ep_reward:.2f}\t'
                  f'Average reward: {running_reward:.2f}\tStd: {action_std:.4f}')
            action_std = round(action_std - action_std_decay, 5)

        if running_reward > saving_reward:
            saving_reward = running_reward
            name = f'./{wandb.run.name}_{i_episode}_Reward-{running_reward}'
            torch.save(policy,    f'{name}_policy.pt')
            torch.save(optimizer, f'{name}_optimizer.pt')
            wandb.save(f'{name}_policy.pt')
            wandb.save(f'{name}_optimizer.pt')
            print('Policy and Optimizer saved')
            ep_reward = test(action_std, env, policy, i_episode)

        if running_reward > target_reward:
            print("Solved!")
            name = f'./{wandb.run.name}_{i_episode}_Reward-{running_reward}_FINAL'
            torch.save(policy,    f'{name}_policy.pt')
            torch.save(optimizer, f'{name}_optimizer.pt')
            wandb.save(f'{name}_policy.pt')
            wandb.save(f'{name}_optimizer.pt')
            break

    print(f"Finished training. Running reward: {running_reward}")


if __name__ == "__main__":
    # Standard training run
    train_or_sweep(is_sweep=False)

    # Or run a wandb sweep:
    # sweep_id = wandb.sweep(sweep=sweep_configuration, project=PROJECT)
    # wandb.agent(sweep_id, function=lambda: train_or_sweep(is_sweep=True), count=50)
