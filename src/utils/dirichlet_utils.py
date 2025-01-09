import torch


def combined_dirichlet(evidence_a, evidence_b, k_class=3):
    """
    evidences: [batchsize, channel, H, W]
    """

    alpha_a = evidence_a + 1
    alpha_b = evidence_b + 1
    dirichlet_strength_a = alpha_a.sum(dim=1, keepdim=True)
    dirichlet_strength_b = alpha_b.sum(dim=1, keepdim=True)
    belief_a = evidence_a / dirichlet_strength_a
    belief_b = evidence_b / dirichlet_strength_b
    uncertainty_a = k_class / dirichlet_strength_a
    uncertainty_b = k_class / dirichlet_strength_b

    sum_conflicts = (
        belief_a.unsqueeze(2)
        * belief_b.unsqueeze(1)
        * (1 - torch.eye(k_class, device="cuda").view(1, k_class, k_class, 1, 1))
    ).sum([1, 2])

    scale_factor = 1 / (1 - sum_conflicts).unsqueeze(1)

    combined_beliefs = (
        scale_factor * belief_a * belief_b
        + belief_a * uncertainty_b
        + belief_b * uncertainty_a
    )
    combined_uncertainties = scale_factor * uncertainty_a * uncertainty_b
    return combined_beliefs, combined_uncertainties


def convert_belief_mass_to_prediction(belief, uncertainty, k_class=3):
    return belief + uncertainty / k_class
