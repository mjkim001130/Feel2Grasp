# train_iql.py 
# Usage:
#   python train_iql.py --data replay_buffer_iql_72d.npz --project Feel2Grasp-IQL --run_name iql_experiment_1

import argparse, time, os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from tqdm import tqdm


# -------------------------
# Utils
# -------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def soft_update(target: nn.Module, source: nn.Module, tau: float):
    with torch.no_grad():
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.mul_(1.0 - tau).add_(sp.data, alpha=tau)

def expectile_loss(diff: torch.Tensor, tau: float):
    # diff = q - v
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * diff.pow(2)).mean()


# -------------------------
# Replay Buffer (NPZ)
# -------------------------
class ReplayBufferNPZ:
    def __init__(self, npz_path: str, device: str = "cuda"):
        d = np.load(npz_path)

        self.device = device

        self.s  = torch.from_numpy(d["observations"]).float().to(device)
        self.a  = torch.from_numpy(d["actions"]).float().to(device)
        self.r  = torch.from_numpy(d["rewards"]).float().to(device)
        self.sp = torch.from_numpy(d["next_observations"]).float().to(device)
        self.d  = torch.from_numpy(d["terminals"]).float().to(device)

        self.n = self.s.shape[0]
        self.obs_dim = self.s.shape[1]
        self.act_dim = self.a.shape[1] if self.a.ndim == 2 else 1

    def sample(self, batch_size: int):
        idx = torch.randint(0, self.n, (batch_size,), device=self.s.device)
        s  = self.s[idx]
        a  = self.a[idx]
        r  = self.r[idx].unsqueeze(-1)
        sp = self.sp[idx]
        d  = self.d[idx].unsqueeze(-1)
        return s, a, r, sp, d


# -------------------------
# Networks
# -------------------------
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(256, 256), act=nn.ReLU):
        super().__init__()
        layers = []
        last = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(last, h), act()]
            last = h
        layers += [nn.Linear(last, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256)):
        super().__init__()
        self.mlp = MLP(obs_dim + act_dim, 1, hidden_dims)

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.mlp(x)

class VNetwork(nn.Module):
    def __init__(self, obs_dim, hidden_dims=(256, 256)):
        super().__init__()
        self.mlp = MLP(obs_dim, 1, hidden_dims)

    def forward(self, s):
        return self.mlp(s)

class GaussianPolicy(nn.Module):
    """
    IQL actor: AWR 
    """
    def __init__(self, obs_dim, act_dim, hidden_dims=(256, 256), log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.backbone = MLP(
            obs_dim,
            hidden_dims[-1],
            hidden_dims=hidden_dims[:-1] if len(hidden_dims) > 1 else ()
        )
        self.mu = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def forward(self, s):
        h = self.backbone(s)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(self.log_std_min, self.log_std_max)
        return mu, log_std

    def log_prob(self, s, a):
        mu, log_std = self(s)
        std = torch.exp(log_std)
        z = (a - mu) / std
        logp = -0.5 * (z.pow(2) + 2.0 * log_std + np.log(2.0 * np.pi))
        return logp.sum(dim=-1, keepdim=True)  # (B,1)


# -------------------------
# Train Loop
# -------------------------
def main():
    p = argparse.ArgumentParser()
    #p.add_argument("--data", type=str, required=True, help="replay_buffer_iql_72d.npz")
    p.add_argument("--data", type=str, default="replay_buffer_iql_72d.npz", help="replay_buffer_iql_72d.npz")
    p.add_argument("--project", type=str, default="Feel2Grasp-IQL")
    p.add_argument("--run_name", type=str, default="iql")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--steps", type=int, default=5_000_000)

    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau_expectile", type=float, default=0.7)
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--clip_exp", type=float, default=100.0)

    p.add_argument("--lr_q", type=float, default=3e-4)
    p.add_argument("--lr_v", type=float, default=3e-4)
    p.add_argument("--lr_pi", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)

    p.add_argument("--target_tau", type=float, default=0.005)
    p.add_argument("--log_interval", type=int, default=500)
    p.add_argument("--save_interval", type=int, default=5_000)
    p.add_argument("--save_dir", type=str, default="./IQL_checkpoints")

    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--save_deploy", action="store_true")

    args = p.parse_args()

    device = args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu"
    set_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)

    wandb.init(project=args.project, name=args.run_name, config=vars(args))

    rb = ReplayBufferNPZ(args.data, device=device)
    obs_dim, act_dim = rb.obs_dim, rb.act_dim
    wandb.config.update({"obs_dim": obs_dim, "act_dim": act_dim}, allow_val_change=True)

    hidden_dims = (args.hidden, args.hidden)

    q1 = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
    q2 = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
    q1_t = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
    q2_t = QNetwork(obs_dim, act_dim, hidden_dims).to(device)
    v = VNetwork(obs_dim, hidden_dims).to(device)
    pi = GaussianPolicy(obs_dim, act_dim, hidden_dims).to(device)

    q1_t.load_state_dict(q1.state_dict())
    q2_t.load_state_dict(q2.state_dict())
    for m in [q1_t, q2_t]:
        for p_ in m.parameters():
            p_.requires_grad_(False)

    opt_q  = torch.optim.AdamW(list(q1.parameters()) + list(q2.parameters()), lr=args.lr_q, weight_decay=args.weight_decay)
    opt_v  = torch.optim.AdamW(v.parameters(), lr=args.lr_v, weight_decay=args.weight_decay)
    opt_pi = torch.optim.AdamW(pi.parameters(), lr=args.lr_pi, weight_decay=args.weight_decay)

    start_time = time.time()
    t = tqdm(range(1, args.steps + 1), desc="Training IQL")

    for step in t:
        s, a, r, sp, d = rb.sample(args.batch_size)

        # 1) Q update: y = r + gamma*(1-d)*V(sp)
        with torch.no_grad():
            v_sp = v(sp)
            y = r + args.gamma * (1.0 - d) * v_sp

        q1_sa = q1(s, a)
        q2_sa = q2(s, a)
        loss_q = F.mse_loss(q1_sa, y) + F.mse_loss(q2_sa, y)

        opt_q.zero_grad(set_to_none=True)
        loss_q.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(list(q1.parameters()) + list(q2.parameters()), args.grad_clip)
        opt_q.step()

        # 2) V update: expectile regression to min target Q
        with torch.no_grad():
            q1t = q1_t(s, a)
            q2t = q2_t(s, a)
            q_min = torch.min(q1t, q2t)

        v_s = v(s)
        diff = (q_min - v_s)
        loss_v = expectile_loss(diff, tau=args.tau_expectile)

        opt_v.zero_grad(set_to_none=True)
        loss_v.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(v.parameters(), args.grad_clip)
        opt_v.step()

        # 3) Policy update: AWR with weights exp(adv/beta)
        with torch.no_grad():
            adv = (q_min - v_s)
            w_unclipped = torch.exp(adv / args.beta)
            w = w_unclipped.clamp(max=args.clip_exp)

        logp = pi.log_prob(s, a)
        loss_pi = -(w * logp).mean()

        opt_pi.zero_grad(set_to_none=True)
        loss_pi.backward()
        if args.grad_clip > 0:
            nn.utils.clip_grad_norm_(pi.parameters(), args.grad_clip)
        opt_pi.step()

        # 4) Target Q soft update
        soft_update(q1_t, q1, args.target_tau)
        soft_update(q2_t, q2, args.target_tau)

        # Logging
        if step % args.log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = step / max(elapsed, 1e-6)

            with torch.no_grad():
                mu, _ = pi(s)
                action_mse = (mu - a).pow(2).mean()
                logp_mean = logp.mean()
                w_clip_rate = (w_unclipped > args.clip_exp).float().mean()
                success_rate_batch = (r.squeeze(-1) > 0.9).float().mean()  

                stats = {
                    "loss/q": loss_q.item(),
                    "loss/v": loss_v.item(),
                    "loss/pi": loss_pi.item(),
                    "q/mean": q_min.mean().item(),
                    "v/mean": v_s.mean().item(),
                    "adv/mean": adv.mean().item(),
                    "w/mean": w.mean().item(),
                    "w/max": w.max().item(),
                    "w/clip_rate": w_clip_rate.item(),
                    "logp/mean": logp_mean.item(),
                    "action/mse": action_mse.item(),
                    "reward/mean": r.mean().item(),
                    "reward/min": r.min().item(),
                    "reward/max": r.max().item(),
                    "reward/success_rate_batch": success_rate_batch.item(),
                    "done/mean": d.mean().item(),
                    "perf/steps_per_sec": steps_per_sec,
                    "step": step,
                }

            wandb.log(stats, step=step)
            t.set_postfix({
                "Q": f"{stats['loss/q']:.4f}",
                "V": f"{stats['loss/v']:.4f}",
                "Pi": f"{stats['loss/pi']:.4f}",
                "ActMSE": f"{stats['action/mse']:.2e}",
                "clip%": f"{100*stats['w/clip_rate']:.1f}",
                "R/s": f"{steps_per_sec:.1f}",
            })

        # Save
        if step % args.save_interval == 0 or step == args.steps:
            full_ckpt = {
                "step": step,
                "q1": q1.state_dict(),
                "q2": q2.state_dict(),
                "q1_t": q1_t.state_dict(),
                "q2_t": q2_t.state_dict(),
                "v": v.state_dict(),
                "pi": pi.state_dict(),
                "opt_q": opt_q.state_dict(),
                "opt_v": opt_v.state_dict(),
                "opt_pi": opt_pi.state_dict(),
                "obs_dim": obs_dim,
                "act_dim": act_dim,
                "hidden": args.hidden,
                "gamma": args.gamma,
                "tau_expectile": args.tau_expectile,
                "beta": args.beta,
                "clip_exp": args.clip_exp,
            }

            path = os.path.join(args.save_dir, f"iql_step_{step}_full_{args.seed}_{args.tau_expectile}.pt")
            torch.save(full_ckpt, path)
            wandb.save(path)

            if args.save_deploy:
                deploy_ckpt = {
                    "step": step,
                    "pi": pi.state_dict(),
                    "obs_dim": obs_dim,
                    "act_dim": act_dim,
                    "hidden": args.hidden,
                }
                dpath = os.path.join(args.save_dir, f"iql_policy_step_{step}_{args.seed}_sec.pt")
                torch.save(deploy_ckpt, dpath)
                wandb.save(dpath)

            print(f"\nSaved: {path}")

    wandb.finish()


if __name__ == "__main__":
    main()
