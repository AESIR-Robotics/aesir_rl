#!/usr/bin/env python3
"""
train_fsm.py — PPO fine-tuning of the full mission (base -> tower -> button ->
goal) in pure MuJoCo, starting from the PRETRAINED base and arm policies.

  * One world per env: MissionEnv (base + arm on the same MjData).
  * One buffer PER AGENT holding ONLY the transitions where that agent acted,
    with the observation that produced the action and the raw sample that
    evaluate() expects (see policies.py).
  * GAE per agent and per env, cut at info["segment_end"]: the base segment
    to the tower, the arm segment and the base segment to the goal are
    separate episodes for their agent. Timeouts (mission or arm) are
    truncations, not terminations: gamma * V(final_obs) is added to the reward
    (as in train_boton.py).
  * Value normalization with one RunningMeanStd PER AGENT (denormalize before
    GAE, normalize the return as the target; as in train_fast/train_boton).
  * The update is base_training.ppo.ppo_update, once per agent, with a low
    learning rate (fine-tuning).

Reward: each sub-env's own reward (base_env / boton_env) plus a task bonus per
mission stage, paid to the agent that closes it:
  base  reaches the pre-dock and scripted docking succeeds   +bonus_dock
        reaches the pre-dock but docking fails               -pen_dock
        reaches the final goal (after pressing)              +bonus_goal
  arm   presses the button                                   +bonus_press
Docking depends on HOW the base arrives (heading, lateral offset), so its
outcome is a useful signal for the base.

Legacy base critic: the remote_2026100* checkpoints carry no ret_rms and their
critic head is on the server's raw reward scale. As in train_fast.py it is
re-initialized; during --critic-warmup iterations ONLY that head is trained
(linear on frozen features) so advantages are meaningful before the actor
moves.

Usage (from the repo root):
  MUJOCO_GL=egl .venv/bin/python3 rl_ws/fsm_training/train_fsm.py \\
      --base-checkpoint checkpoints_base/remote_20261010/fast_best.pt \\
      --arm-checkpoint checkpoints_boton/boton_best.pt \\
      --run-name fsm_flat --randomize-tower --n-envs 8 --steps 256 --iters 400

  # base only (arm frozen)
  ... --freeze-arm

  # resume
  ... --resume checkpoints_fsm/<run>/fsm_last.pt --run-name <run>

  # smoke test: checks logp(rollout) == logp(evaluate) before the update
  ... --iters 2 --steps 64 --n-envs 2
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_RL_WS = os.path.dirname(_HERE)
_ROOT = os.path.dirname(_RL_WS)
for _p in (_ROOT, _RL_WS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from rl_ws.base_training.ppo import RunningMeanStd, compute_gae, ppo_update  # noqa: E402
from fsm_training.mission_env import MissionEnv                              # noqa: E402
from fsm_training.policies import load_arm, load_base                        # noqa: E402

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

AGENTS = ("base", "arm")
STAGES = ("reached_tower", "docked", "pressed", "success")
FAIL_KEYS = ("SUCCESS", "base_to_tower", "dock", "arm", "base_to_goal", "plan_to_goal",
             "timeout", "nan")
# Reason prefix BotonArmEnv reports when the arm episode times out.
ARM_TIMEOUT_PREFIX = "tiempo agotado"


class VecMission:
    """N MissionEnv on threads (mj_step releases the GIL). Auto-reset: when an
    env finishes, the returned obs belongs to the new episode and the terminal
    step's info carries the final obs in info["terminal_obs"]."""

    def __init__(self, n_envs: int, seed: int, **env_kw):
        self.n = n_envs
        self.envs = [MissionEnv(seed=seed + 1000 * i, **env_kw) for i in range(n_envs)]
        self._pool = ThreadPoolExecutor(max_workers=n_envs)
        self.agent = [""] * n_envs
        self.obs = [None] * n_envs

    @staticmethod
    def _safe_reset(env):
        for k in range(5):
            try:
                return env.reset()
            except RuntimeError as exc:           # no route / no valid spawn
                print(f"[vec] reset failed ({exc}); retry {k + 1}/5", flush=True)
        raise RuntimeError("5 consecutive failed resets")

    def reset(self):
        res = list(self._pool.map(self._safe_reset, self.envs))
        for i, (a, o) in enumerate(res):
            self.agent[i], self.obs[i] = a, o
        return list(self.agent), list(self.obs)

    def _step_one(self, i, action):
        env = self.envs[i]
        agent, obs, r, done, info = env.step(action)
        if not done and not np.all(np.isfinite(obs)):
            # Unstable simulation (QACC NaN): end the episode as a failure.
            done = True
            info = env._finish(False, "nan", info)
            info["segment_end"] = True
        if done:
            info["terminal_obs"] = obs
            agent, obs = self._safe_reset(env)
        return agent, obs, float(r), bool(done), info

    def step(self, actions):
        res = list(self._pool.map(self._step_one, range(self.n), actions))
        out_r, out_d, out_i = [], [], []
        for i, (a, o, r, d, inf) in enumerate(res):
            self.agent[i], self.obs[i] = a, o
            out_r.append(r); out_d.append(d); out_i.append(inf)
        return list(self.agent), list(self.obs), out_r, out_d, out_i

    def close(self):
        self._pool.shutdown(wait=True)
        for e in self.envs:
            e.close()


def task_bonus(info: dict, args) -> float:
    """Mission-stage bonus for the agent that acted."""
    b = 0.0
    tr = info.get("transition")
    if info["agent"] == "base":
        if tr == "NAVIGATE_TO_TOWER->DOCK->PRESS_BUTTON":
            b += args.bonus_dock
        elif info.get("fail_stage") == "dock":
            b -= args.pen_dock
        elif info.get("success"):
            b += args.bonus_goal
    elif info.get("arm_success"):
        b += args.bonus_press
    return b


def is_truncation(info: dict) -> bool:
    """Segment ended by TIME, not by a true terminal state."""
    if info.get("fail_stage") == "timeout":
        return True
    if info["agent"] == "arm" and info.get("fail_stage") == "arm":
        return str(info.get("arm_reason", "")).startswith(ARM_TIMEOUT_PREFIX)
    return False


def critic_params(net):
    return [p for n, p in net.named_parameters() if n.startswith("critic.")]


def critic_warmup_update(net, opt_c, obs, ret_norm, epochs, batch, device):
    """Critic head only (opt_c holds no other parameters), the actor is
    unchanged. All networks return the value as the LAST output of forward()."""
    obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    ret_t = torch.as_tensor(ret_norm, dtype=torch.float32, device=device).unsqueeze(-1)
    n, tot, k = obs_t.shape[0], 0.0, 0
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            loss = F.smooth_l1_loss(net(obs_t[idx])[-1], ret_t[idx])
            opt_c.zero_grad(); loss.backward(); opt_c.step()
            tot += loss.item(); k += 1
    net.zero_grad(set_to_none=True)          # drop trunk gradients
    return tot / max(k, 1)


@torch.no_grad()
def logp_check(net, obs, acts, old_logp, device, n_max=4096):
    """max |logp_evaluate - logp_rollout|; with the same network the first
    PPO pass ratio must be exp(0) = 1."""
    k = min(len(obs), n_max)
    lp, _, _ = net.evaluate(torch.as_tensor(obs[:k], dtype=torch.float32, device=device),
                            torch.as_tensor(acts[:k], dtype=torch.float32, device=device))
    d = lp.squeeze(-1).cpu().numpy() - old_logp[:k]
    return float(np.abs(d).max()), float(np.exp(d).mean())


@torch.no_grad()
def approx_kl(net, obs, acts, old_logp, device, n_max=4096):
    """Approximate KL(old||new), Schulman's k3 estimator: E[(r-1) - log r]."""
    k = min(len(obs), n_max)
    lp, _, _ = net.evaluate(torch.as_tensor(obs[:k], dtype=torch.float32, device=device),
                            torch.as_tensor(acts[:k], dtype=torch.float32, device=device))
    logr = lp.squeeze(-1).cpu().numpy() - old_logp[:k]
    return float(np.mean(np.expm1(logr) - logr))


def build_batch(seqs, rms: RunningMeanStd, gamma, lam):
    """seqs: one dict per (env, contiguous segment) with obs/act/logp/val/rew/
    done lists and 'last_val' (normalized). Returns flat arrays."""
    O, A, LP, ADV, RET = [], [], [], [], []
    for s in seqs:
        if not s["rew"]:
            continue
        rew = np.asarray(s["rew"], dtype=np.float32)[:, None]
        val = rms.denormalize(np.asarray(s["val"], dtype=np.float32))[:, None]
        done = np.asarray(s["done"], dtype=np.float32)[:, None]
        last = rms.denormalize(np.array([s["last_val"]], dtype=np.float32))
        adv, ret = compute_gae(rew, val, done, last, gamma, lam)
        O.append(np.asarray(s["obs"], dtype=np.float32))
        A.append(np.asarray(s["act"], dtype=np.float32))
        LP.append(np.asarray(s["logp"], dtype=np.float32))
        ADV.append(adv[:, 0]); RET.append(ret[:, 0])
    if not O:
        return None
    return (np.concatenate(O), np.concatenate(A), np.concatenate(LP),
            np.concatenate(ADV), np.concatenate(RET))


def pick_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(args) -> None:
    device = pick_device(args.device)
    # No TF32: with TF32 (cuDNN default on Ampere+) the base convolution gives
    # a batch-size-dependent mean. Measured on an RTX 5050:
    # max|mu(batch=1) - mu(batch=N)| = 5.9e-3 with TF32 vs 3.8e-6 without; with
    # std=0.22 that pushed |logp_rollout - logp_evaluate| up to 8e-2 (noisy
    # first-pass PPO ratio instead of 1). Without TF32 it is back to ~1e-5.
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    run_dir = Path(args.ckpt_root) / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "args.json").write_text(json.dumps(vars(args), indent=2))
    print(f"device={device}  run_dir={run_dir}", flush=True)

    # ── policies ──
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        base, bsrc = load_base(args.resume, device)
        arm, asrc = load_arm(args.resume, device)
    else:
        base, bsrc = load_base(args.base_checkpoint, device, log_std=args.base_log_std)
        arm, asrc = load_arm(args.arm_checkpoint, device)
    agents = {"base": base, "arm": arm}
    if args.heatmap == "auto":
        args.heatmap = "legacy" if base.legacy else "current"
    rms = {a: (agents[a].ret_rms or RunningMeanStd()) for a in AGENTS}

    # Critic without ret_rms is on the raw reward scale: re-initialize it.
    fresh_critic = {"base": base.ret_rms is None and resume is None,
                    "arm": arm.ret_rms is None and resume is None}
    for a in AGENTS:
        if fresh_critic[a]:
            torch.nn.init.orthogonal_(agents[a].net.critic.weight, gain=1.0)
            torch.nn.init.zeros_(agents[a].net.critic.bias)
            print(f"[{a}] checkpoint without ret_rms: critic head re-initialized", flush=True)

    lr = {"base": args.lr_base, "arm": args.lr_arm}
    ent = {"base": args.ent_base, "arm": args.ent_arm}
    frozen = {"base": args.freeze_base, "arm": args.freeze_arm}
    opt = {a: torch.optim.Adam(agents[a].net.parameters(), lr=lr[a]) for a in AGENTS}
    opt_c = {a: torch.optim.Adam(critic_params(agents[a].net), lr=args.lr_critic_warmup)
             for a in AGENTS}
    start_iter, best_sr = 0, -1.0
    if resume is not None:
        for a in AGENTS:
            if "optimizer" in resume[a]:
                opt[a].load_state_dict(resume[a]["optimizer"])
                for g in opt[a].param_groups:
                    g["lr"] = lr[a]
        start_iter = int(resume.get("iter", 0))
        # The "best" score is inherited only when resuming the SAME run: a new
        # run (other run_dir, maybe another env setup) is not comparable.
        same_run = Path(args.resume).resolve().parent == run_dir.resolve()
        best_sr = float(resume.get("best_success", -1.0)) if same_run else -1.0
        print(f"[resume] {args.resume} iter={start_iter} best_success={best_sr:.3f}")
    for a in AGENTS:
        agents[a].net.train()

    env_kw = dict(track=args.track, tower_xy=tuple(args.tower),
                  approach_yaw=np.radians(args.approach_yaw_deg), target_button=args.button,
                  dock_mode="scripted", max_steps=args.max_steps, verbose=False,
                  randomize_tower=args.randomize_tower, tower_xy_range=args.tower_range,
                  heatmap_encoding=args.heatmap, backoff_dist=args.backoff,
                  tower_plan_half=args.tower_plan_half)
    venv = VecMission(args.n_envs, args.seed, **env_kw)
    N, T = args.n_envs, args.steps
    print(f"envs={N}  steps/env/iter={T}  base={base.arch} (stores {base.store_dim} dims)  "
          f"heatmap={args.heatmap}  tower={'random' if args.randomize_tower else tuple(args.tower)}  "
          f"frozen={[a for a in AGENTS if frozen[a]] or 'none'}", flush=True)

    use_wandb = args.wandb and _HAS_WANDB
    if use_wandb:
        wandb.init(project="AIDL-PPO-AESIR-FSM", name=args.run_name, config=vars(args))

    hist = deque(maxlen=args.window)          # per-episode milestone dicts
    fails = deque(maxlen=args.window)
    ep_ret = {a: np.zeros(N) for a in AGENTS}
    ret_hist = {a: deque(maxlen=args.window) for a in AGENTS}
    n_episodes = 0
    env_steps = 0                             # policy control steps (base + arm)
    t_start = time.time()
    csv_path = run_dir / "metrics.csv"
    agent_ids, obs = venv.reset()
    # Open segment per (agent, env), continued across iterations.
    open_seq = {a: [None] * N for a in AGENTS}
    log_f = open(run_dir / "metrics.jsonl", "a")

    def new_seq():
        return {"obs": [], "act": [], "logp": [], "val": [], "rew": [], "done": []}

    try:
        for it in range(start_iter, start_iter + args.iters):
            t0 = time.time()
            seqs = {a: [] for a in AGENTS}
            for a in AGENTS:
                for i in range(N):
                    open_seq[a][i] = new_seq()
                    seqs[a].append(open_seq[a][i])
            n_ep_it = 0

            for t in range(T):
                actions = [None] * N
                cache = {}
                for a in AGENTS:
                    idx = [i for i in range(N) if agent_ids[i] == a]
                    if not idx:
                        continue
                    o = np.stack([obs[i] for i in idx]).astype(np.float32)
                    env_a, store_a, lp, v = agents[a].act_batch(o)
                    for k, i in enumerate(idx):
                        actions[i] = env_a[k]
                        cache[i] = (a, o[k], store_a[k], lp[k], v[k])

                nids, nobs, rews, dones, infos = venv.step(actions)

                # Bootstrap values for truncations (final obs of the agent that acted).
                trunc = [i for i in range(N) if infos[i]["segment_end"] and is_truncation(infos[i])]
                v_trunc = {}
                for i in trunc:
                    a = infos[i]["agent"]
                    term = infos[i].get("terminal_obs", nobs[i]) if dones[i] else nobs[i]
                    v_trunc[i] = float(rms[a].denormalize(agents[a].value(term[None]))[0])

                for i in range(N):
                    a, o, sa, lp, v = cache[i]
                    inf = infos[i]
                    assert inf["agent"] == a
                    r = rews[i] + task_bonus(inf, args)
                    ep_ret[a][i] += r
                    seg_end = bool(inf["segment_end"])
                    if i in v_trunc:
                        r += args.gamma * v_trunc[i]
                    s = open_seq[a][i]
                    s["obs"].append(o); s["act"].append(sa); s["logp"].append(lp)
                    s["val"].append(v); s["rew"].append(r); s["done"].append(float(seg_end))
                    if dones[i]:
                        n_episodes += 1; n_ep_it += 1
                        hist.append({k: bool(inf.get(k)) for k in STAGES})
                        fails.append(inf.get("fail_stage") or "SUCCESS")
                        for b in AGENTS:
                            ret_hist[b].append(ep_ret[b][i]); ep_ret[b][i] = 0.0

                agent_ids, obs = nids, nobs

            # End-of-rollout bootstrap: the open segment of the ACTIVE agent.
            for a in AGENTS:
                for i in range(N):
                    s = open_seq[a][i]
                    s["last_val"] = 0.0
                    if s["rew"] and not s["done"][-1] and agent_ids[i] == a:
                        s["last_val"] = float(agents[a].value(np.asarray(obs[i], np.float32)[None])[0])

            # ── per-agent update ──
            mets = {}
            warm = (it - start_iter) < args.critic_warmup
            for a in AGENTS:
                batch = build_batch(seqs[a], rms[a], args.gamma, args.lam)
                if batch is None:
                    mets[a] = {"n": 0}
                    continue
                fo, fa, flp, adv, ret = batch
                rms[a].update(ret)
                fret = rms[a].normalize(ret).astype(np.float32)
                fadv = ((adv - adv.mean()) / (adv.std() + 1e-8)).astype(np.float32)
                dlp, ratio0 = logp_check(agents[a].net, fo, fa, flp, device)
                m = {"n": len(fo), "logp_err": dlp, "ratio0": ratio0}
                if frozen[a]:
                    pass
                elif warm:
                    m["v_warm"] = critic_warmup_update(agents[a].net, opt_c[a], fo, fret,
                                                       args.epochs, args.batch, device)
                else:
                    m.update(ppo_update(agents[a].net, opt[a], fo, fa, flp, fadv, fret,
                                        args.epochs, args.batch, args.clip, args.vf_coef,
                                        ent[a], device))
                    _, ratio1 = logp_check(agents[a].net, fo, fa, flp, device)
                    m["ratio_after"] = ratio1
                    m["kl"] = approx_kl(agents[a].net, fo, fa, flp, device)
                mets[a] = m

            env_steps += N * T
            # ── metrics ──
            sr = {k: (np.mean([h[k] for h in hist]) if hist else float("nan")) for k in STAGES}
            fail_mix = {}
            for f in fails:
                fail_mix[f] = fail_mix.get(f, 0) + 1
            dt = time.time() - t0
            print(f"[Iter {it:4d}] ep={n_episodes:5d} (+{n_ep_it})  window={len(hist)}  "
                  f"tower={sr['reached_tower']:.0%} dock={sr['docked']:.0%} "
                  f"press={sr['pressed']:.0%} SUCCESS={sr['success']:.0%}  "
                  f"({dt:.0f}s, {N * T / dt:.0f} steps/s){'  [critic warm-up]' if warm else ''}",
                  flush=True)
            for a in AGENTS:
                m = mets[a]
                if m.get("n"):
                    extra = (f"pi={m['pi']:+.4f} v={m['v']:.3f} ent={m['ent']:.2f} "
                             f"ratio_post={m['ratio_after']:.3f}" if "pi" in m else
                             f"v_warm={m['v_warm']:.3f}" if "v_warm" in m else "frozen")
                    rh = np.mean(ret_hist[a]) if ret_hist[a] else float("nan")
                    print(f"           {a:4s}: n={m['n']:5d} |logp_err|={m['logp_err']:.1e} "
                          f"ratio0={m['ratio0']:.4f} {extra} ret_ep={rh:8.1f} "
                          f"rms=({rms[a].mean:.1f},{rms[a].std:.1f})", flush=True)
            print(f"           failures: {fail_mix}", flush=True)

            # Per-iteration log: metrics.jsonl (everything, nested) and
            # metrics.csv (fixed columns, read by plot_fsm.py). Both flushed
            # every iteration so plots can be made while training.
            rec = {"iter": it + 1, "wall_s": time.time() - t_start, "iter_s": dt,
                   "env_steps": env_steps, "episodes": n_episodes, "window": len(hist),
                   **{f"sr_{k}": sr[k] for k in STAGES}}
            for f in FAIL_KEYS:
                rec[f"fail_{f}"] = fail_mix.get(f, 0)
            for a in AGENTS:
                m = mets[a]
                rec[f"{a}_ret_ep"] = float(np.mean(ret_hist[a])) if ret_hist[a] else float("nan")
                rec[f"{a}_n"] = m.get("n", 0)
                for k in ("pi", "v", "ent", "kl", "logp_err", "ratio0", "ratio_after", "v_warm"):
                    rec[f"{a}_{k}"] = m.get(k, float("nan"))
                rec[f"{a}_lr"] = 0.0 if frozen[a] else (args.lr_critic_warmup if warm else lr[a])
                rec[f"{a}_phase"] = "frozen" if frozen[a] else ("warmup" if warm else "ppo")
            log_f.write(json.dumps({**rec, "fails": fail_mix}, default=float) + "\n"); log_f.flush()
            new_csv = not csv_path.exists()
            with open(csv_path, "a") as cf:
                if new_csv:
                    cf.write(",".join(rec.keys()) + "\n")
                cf.write(",".join(str(v) for v in rec.values()) + "\n")
            if use_wandb:
                wandb.log({k: v for k, v in rec.items() if isinstance(v, (int, float))}, step=it + 1)

            def _ckpt():
                out = {"iter": it + 1, "best_success": best_sr, "metrics": rec,
                       "args": vars(args), "heatmap_encoding": args.heatmap}
                for a in AGENTS:
                    out[a] = {"policy": agents[a].net.state_dict(),
                              "optimizer": opt[a].state_dict(),
                              "ret_rms": rms[a].state_dict()}
                out["base"].update(arch=base.arch, store_dim=base.store_dim,
                                   source=str(args.base_checkpoint))
                out["arm"].update(obs_dim=28, act_dim=7, targets=asrc.get("targets"),
                                  source=str(args.arm_checkpoint))
                return out

            if (it + 1) % args.save_every == 0:
                torch.save(_ckpt(), run_dir / f"fsm_iter{it + 1:05d}.pt")
            if len(hist) >= args.min_episodes_best and sr["success"] > best_sr:
                best_sr = sr["success"]
                torch.save(_ckpt(), run_dir / "fsm_best.pt")
                print(f"           -> NEW BEST success={best_sr:.1%} ({len(hist)} ep): "
                      f"{run_dir / 'fsm_best.pt'}", flush=True)
            torch.save(_ckpt(), run_dir / "fsm_last.pt")
    finally:
        log_f.close()
        venv.close()
        if use_wandb:
            wandb.finish()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-checkpoint", default="checkpoints_base/remote_20261010/fast_best.pt")
    ap.add_argument("--arm-checkpoint", default="checkpoints_boton/boton_best.pt")
    ap.add_argument("--resume", default="", help="FSM checkpoint (fsm_*.pt) to resume from")
    ap.add_argument("--run-name", default=time.strftime("fsm_%Y%m%d_%H%M%S"))
    ap.add_argument("--ckpt-root", default="checkpoints_fsm")
    # env
    ap.add_argument("--track", default="flat")
    ap.add_argument("--tower", type=float, nargs=2, default=(2.0, 0.0))
    ap.add_argument("--approach-yaw-deg", type=float, default=0.0)
    ap.add_argument("--randomize-tower", action="store_true")
    ap.add_argument("--tower-range", type=float, default=5.0)
    ap.add_argument("--button", type=int, default=0)
    ap.add_argument("--max-steps", type=int, default=6000)
    ap.add_argument("--backoff", type=float, default=1.0,
                    help="back-off distance after pressing in m (env default 0.5). Measured with "
                         "fast_best, random tower, 12 ep: 0.5 m + box 0.25 -> 10/12 (2 stuck or "
                         "fallen against the tower); 1.0 m + box 0.40 -> 12/12")
    ap.add_argument("--tower-plan-half", type=float, default=0.40,
                    help="tower half-box for the planner (env default 0.25)")
    ap.add_argument("--heatmap", choices=("auto", "current", "legacy"), default="auto",
                    help="auto = legacy for a legacy (actor_mu) base, current otherwise. Measured "
                         "on flat (6 ep): iter 3900/4950 reach 0/6 with 'current' and 5/6 with "
                         "'legacy'")
    # rollout / PPO
    ap.add_argument("--n-envs", type=int, default=6)
    ap.add_argument("--steps", type=int, default=512, help="steps per env per iteration")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.1)
    ap.add_argument("--vf-coef", type=float, default=0.5)
    ap.add_argument("--lr-base", type=float, default=1e-5)
    ap.add_argument("--lr-arm", type=float, default=1e-5)
    ap.add_argument("--lr-critic-warmup", type=float, default=1e-3)
    ap.add_argument("--critic-warmup", type=int, default=5,
                    help="initial iterations that ONLY fit the critic head")
    ap.add_argument("--ent-base", type=float, default=0.0)
    ap.add_argument("--ent-arm", type=float, default=0.0)
    ap.add_argument("--base-log-std", type=float, default=None,
                    help="reset the base exploration log-std on load "
                         "(legacy checkpoints arrive with 1.0 = std 2.7)")
    ap.add_argument("--freeze-base", action="store_true")
    ap.add_argument("--freeze-arm", action="store_true")
    # task bonuses
    ap.add_argument("--bonus-dock", type=float, default=500.0)
    ap.add_argument("--pen-dock", type=float, default=250.0)
    ap.add_argument("--bonus-press", type=float, default=250.0)
    ap.add_argument("--bonus-goal", type=float, default=500.0)
    # logging
    ap.add_argument("--window", type=int, default=50, help="rolling-window size in episodes")
    ap.add_argument("--min-episodes-best", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
