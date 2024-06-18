import torch
from functools import partial


def _single_tensor_adam(
    gradient, stepsize, beta_1, beta_2, moment_1, moment_2, tstep, eps
):
    if gradient is None:
        return None

    tstep.add_(1)
    moment_1.mul_(beta_1).add_((1 - beta_1) * gradient)
    moment_2.mul_(beta_2).add_((1 - beta_2) * gradient**2)

    moment_1_bias_corrected = moment_1 / (1 - beta_1**tstep)
    moment_2_bias_corrected = moment_2 / (1 - beta_2**tstep)

    update = stepsize * (
        moment_1_bias_corrected / (moment_2_bias_corrected.sqrt() + eps)
    )

    return update


def adam_step(gradients, moment_1, moment_2, tstep, beta_1, beta_2, stepsize, eps):
    if not isinstance(gradients, torch.Tensor):
        updates = []
        for g, m1, m2, t in zip(gradients, moment_1, moment_2, tstep):
            update = _single_tensor_adam(g, stepsize, beta_1, beta_2, m1, m2, t, eps)
            updates.append(update)
        return updates
    else:
        return _single_tensor_adam(
            gradients, stepsize, beta_1, beta_2, moment_1, moment_2, tstep, eps
        )


def initialize_adam(
    params, stepsize=0.001, beta_1=0.9, beta_2=0.999, eps=1e-8, differentiable=False
):
    if not isinstance(params, torch.Tensor):
        moment_1 = []
        moment_2 = []
        tstep = []
        for p in params:
            moment_1.append(torch.zeros_like(p).to(p.device))
            moment_2.append(torch.zeros_like(p).to(p.device))
            tstep.append(torch.tensor(0.0).to(p.device))
    else:
        moment_1 = torch.zeros_like(params).to(params.device)
        moment_2 = torch.zeros_like(params).to(params.device)
        tstep = torch.tensor(0.0).to(params.device)

    func = partial(
        adam_step,
        moment_1=moment_1,
        moment_2=moment_2,
        tstep=tstep,
        beta_1=beta_1,
        beta_2=beta_2,
        stepsize=stepsize,
        eps=eps,
    )
    if not differentiable:
        func = torch.no_grad()(func)
    return func


def update_parameters(params, updates):
    for p, u in zip(params, updates):
        if u is None:
            continue
        p.data -= u
