import torch

from pct.losses import (
    Flow,
    fgw_flow_loss,
    grassmann_flow_loss,
    hidden_states_to_flow,
    medoid_flow,
    ot_flow_loss,
    pairwise_structure,
    pct_loss,
    phf_direction_loss,
    sinkhorn_plan,
)


def make_flow(seed: int = 0) -> Flow:
    generator = torch.Generator().manual_seed(seed)
    z = torch.randn(2, 1, 5, 8, generator=generator)
    z = torch.nn.functional.normalize(z, dim=-1)
    mask = torch.ones(2, 5, dtype=torch.bool)
    return Flow(z=z, mask=mask)


def test_hidden_states_to_flow_shapes_and_mask():
    hidden = tuple(torch.randn(2, 7, 4) for _ in range(4))
    response_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, False, True, True],
        ]
    )
    flow = hidden_states_to_flow(hidden, prompt_len=3, response_mask=response_mask, layers="last")
    assert flow.z.shape == (2, 1, 3, 4)
    assert flow.mask.tolist() == [[True, True, False], [False, False, True]]


def test_phf_direction_loss_is_near_zero_for_identical_flows():
    flow = make_flow()
    assert phf_direction_loss(flow, flow).item() < 1e-6


def test_balanced_sinkhorn_mass_is_one():
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    plan = sinkhorn_plan(cost, epsilon=0.1, n_iters=50, unbalanced=False)
    assert torch.isclose(plan.sum(), torch.tensor(1.0), atol=1e-4)


def test_unbalanced_ot_reports_relaxed_mass():
    student = make_flow(0)
    teacher = make_flow(1)
    _, balanced_mass = ot_flow_loss(student, teacher, unbalanced=False, sinkhorn_iters=20)
    _, unbalanced_mass = ot_flow_loss(student, teacher, unbalanced=True, sinkhorn_iters=20, rho=0.5)
    assert torch.isclose(balanced_mass, torch.tensor(1.0), atol=1e-4)
    assert unbalanced_mass.item() > 0
    assert unbalanced_mass.item() != balanced_mass.item()


def test_pairwise_structure_and_fgw_loss_are_well_formed():
    flow = make_flow(0)
    atoms = flow.z[0, 0]
    structure = pairwise_structure(atoms)
    assert structure.shape == (5, 5)
    assert torch.allclose(structure.diag(), torch.zeros(5), atol=1e-6)

    loss, mass = fgw_flow_loss(flow, flow, sinkhorn_iters=5, fgw_outer=2, max_atoms=8)
    assert loss.ndim == 0
    assert loss.item() >= 0
    assert torch.isclose(mass, torch.tensor(1.0), atol=1e-4)


def test_grassmann_loss_projects_onto_teacher_subspace():
    student = make_flow(0)
    same_a = Flow(z=student.z.clone(), mask=student.mask.clone())
    same_b = Flow(z=student.z.clone(), mask=student.mask.clone())

    loss = grassmann_flow_loss(student, [same_a, same_b], rank=1)
    assert loss.ndim == 0
    assert loss.item() < 1e-6


def test_pct_loss_methods_return_scalar_and_metrics():
    student = make_flow(0)
    teachers = [make_flow(1), make_flow(2), make_flow(3), make_flow(4)]
    for method in (
        "phf_single",
        "phf_random",
        "phf_mean",
        "phf_medoid",
        "phf_grassmann",
        "phf_set",
        "set_ot",
        "set_fgw",
        "set_uot",
    ):
        loss, metrics = pct_loss(student, teachers, method=method, sinkhorn_iters=5, max_atoms=8)
        assert loss.ndim == 0
        assert "pct_loss" in metrics


def test_medoid_flow_selects_central_teacher():
    base = make_flow(0)
    close = Flow(z=base.z.clone(), mask=base.mask.clone())
    far = make_flow(10)
    teacher, idx = medoid_flow([far, base, close])
    assert idx in {1, 2}
    assert teacher.z.shape == base.z.shape
