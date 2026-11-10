"""EGMR 打分函数（与模型无关，可在 CPU 上单元测试）。

EGMR：Exemplar-Grounded Multimodal Retrieval。
  sim(q, j) = cos(h_q, h_j) + β·cos(v_q, v_j)·1[q 与 j 都有图]
  e_{q,c}  = 训练样本中带标签 c 的前 m 个最相似样本的平均相似度（类别均衡；冲突感知时冲突组样本对其组内全部标签计数）
  S_{q,c}  = f_{q,c} + λ·(e_{q,c} − mean_{c'∈Top-K} e_{q,c'})        仅对 Top-K 候选；非候选保持 f
全局 kNN 对照（kNN-LM 式）：p_knn = Σ_{前 k 个邻居} softmax(sim/τ) 投到其标签；S = log((1−λ)·softmax(f) + λ·p_knn)。
"""
from __future__ import annotations

import torch
from torch.nn import functional as F


def similarity(hq, hm, vq=None, vm=None, has_q=None, has_m=None, beta=0.0):
    """hq [Q,H]、hm [M,H] 已 L2 归一化；返回 [Q,M]。"""
    s = hq.float() @ hm.float().T
    if beta and vq is not None:
        both = has_q[:, None].float() * has_m[None, :].float()
        s = s + beta * (vq.float() @ vm.float().T) * both
    return s


def exemplar_scores(sim, label_sets, cands, m):
    """sim [Q,M]；label_sets [M,C] 布尔；cands [Q,K] → e [Q,K]（无该类样本时取该行最小相似度）。"""
    q, k = cands.shape
    e = torch.empty(q, k, device=sim.device)
    floor = sim.min(-1).values
    for c in torch.unique(cands).tolist():
        cols = label_sets[:, c].nonzero().flatten()
        pos = (cands == c).nonzero()
        rows = pos[:, 0]
        if len(cols) == 0:
            e[pos[:, 0], pos[:, 1]] = floor[rows]
            continue
        top = sim[rows][:, cols].topk(min(m, len(cols)), dim=-1).values.mean(-1)
        e[pos[:, 0], pos[:, 1]] = top
    return e


def egmr_logits(f, cands, e, lam):
    if lam == 0:
        return f.clone()
    out = f.clone().float()
    out.scatter_add_(1, cands, lam * (e - e.mean(-1, keepdim=True)))
    return out


def knn_logits(f, sim, labels, num_classes, k, tau, lam):
    if lam == 0:
        return f.clone()
    top, idx = sim.topk(min(k, sim.shape[1]), dim=-1)
    w = torch.softmax(top / tau, -1)
    p = torch.zeros(f.shape[0], num_classes, device=f.device).scatter_add_(1, labels[idx], w)
    return torch.log(((1 - lam) * torch.softmax(f.float(), -1) + lam * p).clamp_min(1e-12))
