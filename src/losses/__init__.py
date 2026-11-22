"""Classification losses (2026-11-22 refactor).

    functional  canonical formulas on (logits, dense candidate mask Y)
    masks       padded <-> dense label conversion
    stateless   nn.Module wrappers with the historical class names / signatures
    confidence  soft-label losses with a per-sample confidence buffer (+ conf_buffer accessor)
    composite   alpha / rho mixes of the above
    swap        uniform-interface losses for the swap / solo / prune / bandit experiments
"""
