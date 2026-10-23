"""Reinforcement-learning trading policies (discrete actions {-1, 0, +1}).

Environment (training rows only): state s_t = [standardised features_t, previous position];
action a_t = position held from t to t+1; reward r_t = a_t * y_t - cost * |a_t - a_{t-1}|,
where y_t is the forward 1-step return label (the realised return over the
holding interval, exactly what the position earns). At prediction time the
policy acts greedily, feeding back its own previous action (no labels used)."""
from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from fxlab.registry import model
from fxlab.models.base import Model

ACTIONS = np.array([-1.0, 0.0, 1.0])
RLK = dict(requires=("torch",), extra={"tasks": ["pol"]}, complexity="O(updates · batch · params)")


def _torch():
    import torch
    torch.set_num_threads(1)
    return torch


class _Policy(Model):
    task = "pol"
    is_sequence = True

    def __init__(self, task="pol", cost=0.0001, epochs=3, seed=0, max_train=20000, hidden=32):
        self.cost, self.epochs, self.seed, self.max_train, self.hidden = cost, epochs, seed, max_train, hidden

    def _prep_fit(self, X, y):
        X, y = np.asarray(X, float)[-self.max_train:], np.asarray(y, float)[-self.max_train:]
        ok = np.isfinite(y)
        self.sc = StandardScaler().fit(np.nan_to_num(X))
        Z = np.clip(self.sc.transform(np.nan_to_num(X)), -10, 10)[ok]
        y = y[ok]
        self.ys = np.std(y) + 1e-12
        return Z.astype(np.float32), (y / self.ys).astype(np.float32), self.cost / self.ys

    def _state(self, z, prev):
        return np.r_[z, prev].astype(np.float32)

    def act(self, s):
        raise NotImplementedError

    def predict(self, X):
        Z = np.clip(self.sc.transform(np.nan_to_num(np.asarray(X, float))), -10, 10).astype(np.float32)
        prev, out = 0.0, np.empty(len(Z))
        for t in range(len(Z)):
            prev = ACTIONS[self.act(self._state(Z[t], prev))]
            out[t] = prev
        return out

    def predict_proba(self, X):
        return (self.predict(X) + 1) / 2


class _DQN(_Policy):
    def fit(self, X, y):
        torch = _torch()
        nn = torch.nn
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        Z, y, c = self._prep_fit(X, y)
        p = Z.shape[1] + 1
        self.q = nn.Sequential(nn.Linear(p, self.hidden), nn.ReLU(), nn.Linear(self.hidden, 3))
        tgt = nn.Sequential(nn.Linear(p, self.hidden), nn.ReLU(), nn.Linear(self.hidden, 3))
        tgt.load_state_dict(self.q.state_dict())
        opt = torch.optim.Adam(self.q.parameters(), lr=1e-3)
        buf_s, buf_a, buf_r, buf_s2 = [], [], [], []
        eps, gamma, steps = 1.0, 0.9, 0
        for ep in range(self.epochs):
            prev = 0.0
            for t in range(len(Z) - 1):
                s = self._state(Z[t], prev)
                if rng.random() < eps:
                    a = int(rng.integers(3))
                else:
                    with torch.no_grad():
                        a = int(self.q(torch.tensor(s)).argmax())
                pos = ACTIONS[a]
                r = pos * y[t] - c * abs(pos - prev)
                s2 = self._state(Z[t + 1], pos)
                buf_s.append(s); buf_a.append(a); buf_r.append(r); buf_s2.append(s2)
                prev = pos
                steps += 1
                eps = max(0.05, eps * 0.9995)
                if len(buf_s) >= 512 and steps % 4 == 0:
                    i = rng.integers(0, len(buf_s), 128)
                    S = torch.tensor(np.array([buf_s[k] for k in i])); A = torch.tensor([buf_a[k] for k in i])
                    R = torch.tensor([buf_r[k] for k in i], dtype=torch.float32)
                    S2 = torch.tensor(np.array([buf_s2[k] for k in i]))
                    with torch.no_grad():
                        tq = R + gamma * tgt(S2).max(1).values
                    qv = self.q(S).gather(1, A.unsqueeze(1)).squeeze(1)
                    loss = nn.functional.smooth_l1_loss(qv, tq)
                    opt.zero_grad(); loss.backward(); opt.step()
                if steps % 1000 == 0:
                    tgt.load_state_dict(self.q.state_dict())
                if len(buf_s) > 50000:
                    del buf_s[:10000], buf_a[:10000], buf_r[:10000], buf_s2[:10000]
        return self

    def act(self, s):
        torch = _torch()
        with torch.no_grad():
            return int(self.q(torch.tensor(s)).argmax())


@model("M.RL.DQN", "reinforcement", "deep Q-network: MLP Q(s, a), experience replay, target network (sync 1000 steps), "
       "epsilon-greedy exploration decaying to 0.05, gamma 0.9, Huber TD loss", params={"epochs": 3},
       space={"epochs": [2, 5]}, **RLK)
def dqn(task="pol", epochs=3, **kw):
    return _DQN(task, epochs=epochs, **kw)


class _PPO(_Policy):
    def fit(self, X, y):
        torch = _torch()
        nn = torch.nn
        torch.manual_seed(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        Z, y, c = self._prep_fit(X, y)
        p = Z.shape[1] + 1
        h = self.hidden
        self.pi = nn.Sequential(nn.Linear(p, h), nn.Tanh(), nn.Linear(h, 3))
        v = nn.Sequential(nn.Linear(p, h), nn.Tanh(), nn.Linear(h, 1))
        opt = torch.optim.Adam(list(self.pi.parameters()) + list(v.parameters()), lr=3e-4)
        T, gamma, lam, clip = 256, 0.9, 0.95, 0.2
        for ep in range(self.epochs):
            prev = 0.0
            for start in range(0, len(Z) - T - 1, T):
                S, A, R, LP = [], [], [], []
                for t in range(start, start + T):
                    s = self._state(Z[t], prev)
                    with torch.no_grad():
                        logits = self.pi(torch.tensor(s))
                        dist = torch.distributions.Categorical(logits=logits)
                        a = int(torch.multinomial(torch.softmax(logits, 0), 1, generator=g))
                        LP.append(float(dist.log_prob(torch.tensor(a))))
                    pos = ACTIONS[a]
                    R.append(pos * y[t] - c * abs(pos - prev)); S.append(s); A.append(a)
                    prev = pos
                S_t = torch.tensor(np.array(S)); A_t = torch.tensor(A); R_t = torch.tensor(R, dtype=torch.float32)
                with torch.no_grad():
                    V = v(S_t).squeeze(1)
                    nxt = torch.cat([V[1:], V[-1:]])
                    delta = R_t + gamma * nxt - V
                    adv = torch.zeros_like(delta); acc = 0.0
                    for k in range(len(delta) - 1, -1, -1):
                        acc = float(delta[k]) + gamma * lam * acc
                        adv[k] = acc
                    ret = adv + V
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
                LP_t = torch.tensor(LP)
                for _ in range(4):
                    dist = torch.distributions.Categorical(logits=self.pi(S_t))
                    ratio = torch.exp(dist.log_prob(A_t) - LP_t)
                    pl = -torch.min(ratio * adv, ratio.clamp(1 - clip, 1 + clip) * adv).mean()
                    vl = ((v(S_t).squeeze(1) - ret) ** 2).mean()
                    loss = pl + 0.5 * vl - 0.01 * dist.entropy().mean()
                    opt.zero_grad(); loss.backward(); opt.step()
        return self

    def act(self, s):
        torch = _torch()
        with torch.no_grad():
            return int(self.pi(torch.tensor(s)).argmax())


@model("M.RL.PPO", "reinforcement", "proximal policy optimisation: categorical actor + value critic, GAE(0.9, 0.95), "
       "clip 0.2, 4 epochs per 256-step rollout, entropy bonus", params={"epochs": 3}, space={"epochs": [2, 5]}, **RLK)
def ppo(task="pol", epochs=3, **kw):
    return _PPO(task, epochs=epochs, **kw)


class _SAC(_Policy):
    """Discrete soft actor-critic (Christodoulou 2019): twin Q networks, soft policy over 3 actions,
    automatic entropy temperature, replay buffer."""

    def fit(self, X, y):
        torch = _torch()
        nn = torch.nn
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        g = torch.Generator().manual_seed(self.seed)
        Z, y, c = self._prep_fit(X, y)
        p, h = Z.shape[1] + 1, self.hidden
        mk = lambda: nn.Sequential(nn.Linear(p, h), nn.ReLU(), nn.Linear(h, 3))
        q1, q2, self.pol = mk(), mk(), mk()
        q1t, q2t = mk(), mk()
        q1t.load_state_dict(q1.state_dict()); q2t.load_state_dict(q2.state_dict())
        log_alpha = torch.zeros(1, requires_grad=True)
        oq = torch.optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=1e-3)
        op = torch.optim.Adam(self.pol.parameters(), lr=1e-3)
        oa = torch.optim.Adam([log_alpha], lr=1e-3)
        target_ent = 0.6 * np.log(3)
        buf = []
        gamma, tau = 0.9, 0.01
        for ep in range(self.epochs):
            prev = 0.0
            for t in range(len(Z) - 1):
                s = self._state(Z[t], prev)
                with torch.no_grad():
                    pr = torch.softmax(self.pol(torch.tensor(s)), 0)
                a = int(torch.multinomial(pr, 1, generator=g))
                pos = ACTIONS[a]
                buf.append((s, a, pos * y[t] - c * abs(pos - prev), self._state(Z[t + 1], pos)))
                prev = pos
                if len(buf) > 50000:
                    del buf[:10000]
                if len(buf) >= 512 and t % 4 == 0:
                    i = rng.integers(0, len(buf), 128)
                    S = torch.tensor(np.array([buf[k][0] for k in i])); A = torch.tensor([buf[k][1] for k in i])
                    R = torch.tensor([buf[k][2] for k in i], dtype=torch.float32)
                    S2 = torch.tensor(np.array([buf[k][3] for k in i]))
                    alpha = log_alpha.exp().detach()
                    with torch.no_grad():
                        lp2 = torch.log_softmax(self.pol(S2), 1); p2 = lp2.exp()
                        v2 = (p2 * (torch.min(q1t(S2), q2t(S2)) - alpha * lp2)).sum(1)
                        tq = R + gamma * v2
                    l_q = ((q1(S).gather(1, A[:, None]).squeeze(1) - tq) ** 2).mean() + \
                          ((q2(S).gather(1, A[:, None]).squeeze(1) - tq) ** 2).mean()
                    oq.zero_grad(); l_q.backward(); oq.step()
                    lp = torch.log_softmax(self.pol(S), 1); pp = lp.exp()
                    with torch.no_grad():
                        qm = torch.min(q1(S), q2(S))
                    l_p = (pp * (alpha * lp - qm)).sum(1).mean()
                    op.zero_grad(); l_p.backward(); op.step()
                    ent = -(pp.detach() * lp.detach()).sum(1).mean()
                    l_a = (log_alpha * (ent - target_ent)).mean()
                    oa.zero_grad(); l_a.backward(); oa.step()
                    with torch.no_grad():
                        for a_, b_ in ((q1, q1t), (q2, q2t)):
                            for pa, pb in zip(a_.parameters(), b_.parameters()):
                                pb.mul_(1 - tau).add_(tau * pa)
        return self

    def act(self, s):
        torch = _torch()
        with torch.no_grad():
            return int(self.pol(torch.tensor(s)).argmax())


@model("M.RL.SAC", "reinforcement", "discrete soft actor-critic: twin soft Q networks with Polyak targets, stochastic "
       "softmax policy, automatic entropy temperature", params={"epochs": 2}, space={"epochs": [1, 3]}, **RLK)
def sac(task="pol", epochs=2, **kw):
    return _SAC(task, epochs=epochs, **kw)


class _LinUCB(_Policy):
    """Contextual bandit: one ridge regression per arm on context features predicting reward; UCB exploration
    (alpha) during the training pass; greedy at prediction. Position-change costs are included in the reward."""

    def __init__(self, task="pol", alpha=0.5, **kw):
        super().__init__(task, **kw)
        self.alpha = alpha

    def fit(self, X, y):
        Z, y, c = self._prep_fit(X, y)
        d = Z.shape[1] + 2
        self.A = [np.eye(d) for _ in range(3)]
        self.b = [np.zeros(d) for _ in range(3)]
        prev = 0.0
        for t in range(len(Z)):
            x = np.r_[Z[t], prev, 1.0]
            ucb = []
            for k in range(3):
                Ai = np.linalg.inv(self.A[k])
                th = Ai @ self.b[k]
                ucb.append(th @ x + self.alpha * np.sqrt(x @ Ai @ x))
            a = int(np.argmax(ucb))
            pos = ACTIONS[a]
            r = pos * y[t] - c * abs(pos - prev)
            self.A[a] += np.outer(x, x)
            self.b[a] += r * x
            prev = pos
        self.theta = [np.linalg.solve(self.A[k], self.b[k]) for k in range(3)]
        return self

    def act(self, s):
        x = np.r_[s, 1.0]
        return int(np.argmax([th @ x for th in self.theta]))


@model("M.RL.LINUCB", "reinforcement", "LinUCB contextual bandit (per-arm ridge, UCB exploration alpha) over positions "
       "{-1,0,+1} with cost-aware reward", params={"alpha": 0.5}, space={"alpha": [0.1, 0.5, 1.0]},
       extra={"tasks": ["pol"]}, complexity="O(n d^2)")
def linucb(task="pol", alpha=0.5, **kw):
    return _LinUCB(task, alpha=alpha, **kw)
