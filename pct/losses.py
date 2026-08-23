from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass
class Flow:
    z: torch.Tensor
    mask: torch.Tensor


def layer_indices(num_hidden_states: int, mode: str) -> list[int]:
    if mode == "last":
        return [num_hidden_states - 1]
    if mode == "all":
        return list(range(1, num_hidden_states))
    if mode == "even":
        return list(range(1, num_hidden_states, 2))
    if mode == "odd":
        return list(range(2, num_hidden_states, 2))
    raise ValueError(f"Unknown layer mode: {mode}")


def hidden_states_to_flow(
    hidden_states: tuple[torch.Tensor, ...],
    prompt_len: int,
    response_mask: torch.Tensor,
    layers: str = "last",
    eps: float = 1e-8,
) -> Flow:
    selected = [hidden_states[i].float() for i in layer_indices(len(hidden_states), layers)]
    h = torch.stack([x[:, prompt_len:, :] for x in selected], dim=1)
    delta = h[:, :, 1:, :] - h[:, :, :-1, :]
    z = F.normalize(delta, dim=-1, eps=eps)
    mask = response_mask[:, 1:] & response_mask[:, :-1]
    return Flow(z=z, mask=mask)


def phf_direction_loss(student: Flow, teacher: Flow, eps: float = 1e-8) -> torch.Tensor:
    mask = (student.mask & teacher.mask).to(student.z.dtype).unsqueeze(1)
    per = 1.0 - (student.z * teacher.z.detach()).sum(dim=-1)
    mask = mask.expand_as(per)
    return (per * mask).sum() / mask.sum().clamp_min(eps)


def phf_geometry_loss(student: Flow, teacher: Flow, eps: float = 1e-8) -> torch.Tensor:
    if student.z.shape[2] < 2:
        return student.z.new_tensor(0.0)
    s_gram = student.z @ student.z.transpose(-1, -2)
    t_gram = teacher.z.detach() @ teacher.z.detach().transpose(-1, -2)
    mask = (student.mask & teacher.mask).to(student.z.dtype)
    pair_mask = mask[:, None, :, None] * mask[:, None, None, :]
    pair_mask = pair_mask.expand_as(s_gram)
    return (((s_gram - t_gram) ** 2) * pair_mask).sum() / pair_mask.sum().clamp_min(eps)


def mean_flow(teachers: list[Flow], eps: float = 1e-8) -> Flow:
    z = torch.stack([t.z.detach() for t in teachers], dim=0).mean(dim=0)
    z = F.normalize(z, dim=-1, eps=eps)
    mask = teachers[0].mask.clone()
    for teacher in teachers[1:]:
        mask = mask & teacher.mask
    return Flow(z=z, mask=mask)


def _masked_flatten(flow: Flow, eps: float = 1e-8) -> torch.Tensor:
    mask = flow.mask.to(flow.z.dtype).unsqueeze(1).unsqueeze(-1)
    return F.normalize((flow.z.detach() * mask).reshape(-1), dim=0, eps=eps)


def medoid_flow(teachers: list[Flow]) -> tuple[Flow, int]:
    flat = torch.stack([_masked_flatten(teacher) for teacher in teachers], dim=0)
    sim = flat @ flat.T
    dist = 1.0 - sim
    idx = int(dist.sum(dim=1).argmin().item())
    return teachers[idx], idx


def grassmann_flow_loss(
    student: Flow,
    teachers: list[Flow],
    rank: int = 2,
) -> torch.Tensor:
    teacher_z = torch.stack([teacher.z.detach() for teacher in teachers], dim=0)
    valid = student.mask.clone()
    for teacher in teachers:
        valid = valid & teacher.mask

    losses = []
    max_rank = min(rank, teacher_z.shape[0], teacher_z.shape[-1])
    if max_rank < 1:
        return student.z.new_tensor(0.0)

    for b in range(student.z.shape[0]):
        positions = torch.nonzero(valid[b], as_tuple=False).flatten()
        for l in range(student.z.shape[1]):
            for t in positions:
                atoms = teacher_z[:, b, l, t, :].T
                basis, _, _ = torch.linalg.svd(atoms, full_matrices=False)
                basis = basis[:, :max_rank]
                z = student.z[b, l, t, :]
                projection = basis @ (basis.T @ z)
                losses.append((z - projection).square().sum())

    if not losses:
        return student.z.new_tensor(0.0)
    return torch.stack(losses).mean()


def set_phf_loss(
    student: Flow,
    teachers: list[Flow],
    tau: float,
    geometry_weight: float = 0.0,
) -> torch.Tensor:
    losses = torch.stack(
        [
            phf_direction_loss(student, teacher) + geometry_weight * phf_geometry_loss(student, teacher)
            for teacher in teachers
        ]
    )
    return -tau * torch.logsumexp(-losses / tau, dim=0) + tau * torch.log(
        torch.tensor(float(len(teachers)), device=losses.device)
    )


def _flow_atoms(flow: Flow, sample: int, max_atoms: int) -> torch.Tensor:
    z = flow.z[sample].permute(1, 0, 2).reshape(-1, flow.z.shape[-1])
    mask = flow.mask[sample].unsqueeze(1).expand(flow.mask.shape[1], flow.z.shape[1]).reshape(-1)
    z = z[mask]
    if z.shape[0] > max_atoms:
        idx = torch.linspace(0, z.shape[0] - 1, max_atoms, device=z.device).round().long()
        z = z[idx]
    return F.normalize(z, dim=-1)


def sinkhorn_plan(
    cost: torch.Tensor,
    epsilon: float = 0.05,
    n_iters: int = 40,
    unbalanced: bool = False,
    rho: float = 0.5,
) -> torch.Tensor:
    n, m = cost.shape
    log_a = cost.new_full((n,), -torch.log(cost.new_tensor(float(n))))
    log_b = cost.new_full((m,), -torch.log(cost.new_tensor(float(m))))
    log_k = -cost / epsilon
    log_u = torch.zeros_like(log_a)
    log_v = torch.zeros_like(log_b)
    tau = rho / (rho + epsilon) if unbalanced else 1.0
    for _ in range(n_iters):
        log_u = tau * (log_a - torch.logsumexp(log_k + log_v.unsqueeze(0), dim=1))
        log_v = tau * (log_b - torch.logsumexp(log_k + log_u.unsqueeze(1), dim=0))
    return torch.exp(log_k + log_u.unsqueeze(1) + log_v.unsqueeze(0))


def ot_flow_loss(
    student: Flow,
    teacher: Flow,
    max_atoms: int = 64,
    epsilon: float = 0.05,
    sinkhorn_iters: int = 40,
    unbalanced: bool = False,
    rho: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    losses = []
    masses = []
    for b in range(student.z.shape[0]):
        zs = _flow_atoms(student, b, max_atoms=max_atoms)
        zt = _flow_atoms(teacher, b, max_atoms=max_atoms).detach()
        if zs.numel() == 0 or zt.numel() == 0:
            continue
        cost = 1.0 - zs @ zt.T
        plan = sinkhorn_plan(
            cost,
            epsilon=epsilon,
            n_iters=sinkhorn_iters,
            unbalanced=unbalanced,
            rho=rho,
        )
        losses.append((plan * cost).sum())
        masses.append(plan.sum().detach())
    if not losses:
        zero = student.z.new_tensor(0.0)
        return zero, zero
    return torch.stack(losses).mean(), torch.stack(masses).mean()


def generalized_kl(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Generalized KL divergence for non-negative, not necessarily normalized measures."""
    safe_x = x.clamp_min(eps)
    safe_y = y.clamp_min(eps)
    return (x * (safe_x.log() - safe_y.log()) - x + y).sum()


def uot_flow_loss(
    student: Flow,
    teacher: Flow,
    max_atoms: int = 64,
    epsilon: float = 0.05,
    sinkhorn_iters: int = 40,
    rho: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full entropic UOT objective plus interpretable transport diagnostics.

    Returns ``(objective, raw_transport_cost, normalized_transport_cost, mass)``.
    The objective includes the coupling entropy and both relaxed-marginal KL
    penalties, so reducing transported mass cannot lower the training loss for
    free. The normalized cost and mass are diagnostics and are not optimized
    independently.
    """
    objectives = []
    raw_costs = []
    masses = []
    for batch_idx in range(student.z.shape[0]):
        zs = _flow_atoms(student, batch_idx, max_atoms=max_atoms)
        zt = _flow_atoms(teacher, batch_idx, max_atoms=max_atoms).detach()
        if zs.numel() == 0 or zt.numel() == 0:
            continue

        cost = 1.0 - zs @ zt.T
        plan = sinkhorn_plan(
            cost,
            epsilon=epsilon,
            n_iters=sinkhorn_iters,
            unbalanced=True,
            rho=rho,
        )
        a = zs.new_full((zs.shape[0],), 1.0 / zs.shape[0])
        b = zs.new_full((zt.shape[0],), 1.0 / zt.shape[0])
        reference = a[:, None] * b[None, :]
        raw_cost = (plan * cost).sum()
        mass = plan.sum()
        objective = (
            raw_cost
            + epsilon * generalized_kl(plan, reference)
            + rho * generalized_kl(plan.sum(dim=1), a)
            + rho * generalized_kl(plan.sum(dim=0), b)
        )

        objectives.append(objective)
        raw_costs.append(raw_cost.detach())
        masses.append(mass.detach())

    if not objectives:
        zero = student.z.new_tensor(0.0)
        return zero, zero, zero, zero
    mean_raw_cost = torch.stack(raw_costs).mean()
    mean_mass = torch.stack(masses).mean()
    normalized_cost = mean_raw_cost / mean_mass.clamp_min(1e-12)
    return torch.stack(objectives).mean(), mean_raw_cost, normalized_cost, mean_mass


def pairwise_structure(z: torch.Tensor) -> torch.Tensor:
    return (1.0 - z @ z.T).clamp_min(0.0)


def fgw_flow_loss(
    student: Flow,
    teacher: Flow,
    max_atoms: int = 64,
    epsilon: float = 0.05,
    sinkhorn_iters: int = 40,
    fgw_outer: int = 4,
    feature_weight: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    losses = []
    masses = []
    for b in range(student.z.shape[0]):
        zs = _flow_atoms(student, b, max_atoms=max_atoms)
        zt = _flow_atoms(teacher, b, max_atoms=max_atoms).detach()
        if zs.numel() == 0 or zt.numel() == 0:
            continue

        feature_cost = 1.0 - zs @ zt.T
        dx = pairwise_structure(zs)
        dy = pairwise_structure(zt)
        a = zs.new_full((zs.shape[0],), 1.0 / zs.shape[0])
        b_mass = zs.new_full((zt.shape[0],), 1.0 / zt.shape[0])
        plan = sinkhorn_plan(feature_cost, epsilon=epsilon, n_iters=sinkhorn_iters)

        for _ in range(fgw_outer):
            gw_cost = (dx.square() @ a).unsqueeze(1)
            gw_cost = gw_cost + (dy.square() @ b_mass).unsqueeze(0)
            gw_cost = gw_cost - 2.0 * dx @ plan @ dy.T
            fused_cost = (1.0 - feature_weight) * feature_cost + feature_weight * gw_cost
            plan = sinkhorn_plan(fused_cost, epsilon=epsilon, n_iters=sinkhorn_iters)

        feature_loss = (plan * feature_cost).sum()
        structure_delta = dx[:, None, :, None] - dy[None, :, None, :]
        structure_loss = (structure_delta.square() * plan[:, :, None, None] * plan[None, None, :, :]).sum()
        losses.append((1.0 - feature_weight) * feature_loss + feature_weight * structure_loss)
        masses.append(plan.sum().detach())

    if not losses:
        zero = student.z.new_tensor(0.0)
        return zero, zero
    return torch.stack(losses).mean(), torch.stack(masses).mean()


def pct_loss(
    student: Flow,
    teachers: list[Flow],
    method: str,
    tau: float = 0.05,
    geometry_weight: float = 0.0,
    max_atoms: int = 64,
    sinkhorn_epsilon: float = 0.05,
    sinkhorn_iters: int = 40,
    uot_rho: float = 0.5,
    fgw_outer: int = 4,
    fgw_feature_weight: float = 0.5,
    grassmann_rank: int = 2,
) -> tuple[torch.Tensor, dict[str, float]]:
    metrics: dict[str, float] = {}
    if method == "phf_single":
        loss = phf_direction_loss(student, teachers[0]) + geometry_weight * phf_geometry_loss(student, teachers[0])
    elif method == "phf_random":
        teacher = random.choice(teachers)
        loss = phf_direction_loss(student, teacher) + geometry_weight * phf_geometry_loss(student, teacher)
    elif method == "phf_mean":
        teacher = mean_flow(teachers)
        loss = phf_direction_loss(student, teacher) + geometry_weight * phf_geometry_loss(student, teacher)
    elif method == "phf_medoid":
        teacher, idx = medoid_flow(teachers)
        metrics["pct_medoid_index"] = float(idx)
        loss = phf_direction_loss(student, teacher) + geometry_weight * phf_geometry_loss(student, teacher)
    elif method == "phf_grassmann":
        loss = grassmann_flow_loss(student, teachers, rank=grassmann_rank)
    elif method == "phf_set":
        loss = set_phf_loss(student, teachers, tau=tau, geometry_weight=geometry_weight)
    elif method in {"set_ot", "set_uot"}:
        vals = []
        raw_costs = []
        normalized_costs = []
        masses = []
        for teacher in teachers:
            if method == "set_uot":
                val, raw_cost, normalized_cost, mass = uot_flow_loss(
                    student,
                    teacher,
                    max_atoms=max_atoms,
                    epsilon=sinkhorn_epsilon,
                    sinkhorn_iters=sinkhorn_iters,
                    rho=uot_rho,
                )
            else:
                val, mass = ot_flow_loss(
                    student,
                    teacher,
                    max_atoms=max_atoms,
                    epsilon=sinkhorn_epsilon,
                    sinkhorn_iters=sinkhorn_iters,
                )
                raw_cost = val.detach()
                normalized_cost = raw_cost / mass.clamp_min(1e-12)
            vals.append(val)
            raw_costs.append(raw_cost)
            normalized_costs.append(normalized_cost)
            masses.append(mass)
        losses = torch.stack(vals)
        loss = -tau * torch.logsumexp(-losses / tau, dim=0) + tau * torch.log(
            torch.tensor(float(len(teachers)), device=losses.device)
        )
        metrics["pct_transport_cost_raw"] = float(torch.stack(raw_costs).mean().detach().cpu())
        metrics["pct_transport_cost_normalized"] = float(
            torch.stack(normalized_costs).mean().detach().cpu()
        )
        metrics["pct_transport_mass"] = float(torch.stack(masses).mean().detach().cpu())
    elif method == "set_fgw":
        vals = []
        masses = []
        for teacher in teachers:
            val, mass = fgw_flow_loss(
                student,
                teacher,
                max_atoms=max_atoms,
                epsilon=sinkhorn_epsilon,
                sinkhorn_iters=sinkhorn_iters,
                fgw_outer=fgw_outer,
                feature_weight=fgw_feature_weight,
            )
            vals.append(val)
            masses.append(mass)
        losses = torch.stack(vals)
        loss = -tau * torch.logsumexp(-losses / tau, dim=0) + tau * torch.log(
            torch.tensor(float(len(teachers)), device=losses.device)
        )
        metrics["pct_transport_mass"] = float(torch.stack(masses).mean().detach().cpu())
    else:
        raise ValueError(f"Unknown PCT method: {method}")
    metrics["pct_loss"] = float(loss.detach().cpu())
    return loss, metrics
