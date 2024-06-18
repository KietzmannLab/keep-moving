import numpy as np
import quadprog
import torch
from torch.utils.data import DataLoader
import torch.nn.functional

from avalanche.models import avalanche_forward
from avalanche.training.plugins.strategy_plugin import SupervisedPlugin
from avalanche.training.plugins.evaluation import default_evaluator
from avalanche.training.templates import SupervisedTemplate


class LogitGEMPlugin(SupervisedPlugin):
    """
    Gradient Episodic Memory Plugin.
    GEM projects the gradient on the current minibatch by using an external
    episodic memory of patterns from previous experiences. The gradient on
    the current minibatch is projected so that the dot product with all the
    reference gradients of previous tasks remains positive.
    This plugin does not use task identities.
    """

    def __init__(self, patterns_per_experience: int, memory_strength: float):
        """
        :param patterns_per_experience: number of patterns per experience in the
            memory.
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        """

        super().__init__()

        self.patterns_per_experience = int(patterns_per_experience)
        self.memory_strength = memory_strength

        self.memory_x, self.memory_y, self.memory_tid = {}, {}, {}

        self.G = None

    def before_training_iteration(self, strategy, **kwargs):
        """
        Compute gradient constraints on previous memory samples from all
        experiences.
        """

        if strategy.clock.train_exp_counter > 0:
            G = []
            strategy.model.train()
            for t in range(strategy.clock.train_exp_counter):
                strategy.model.train()
                strategy.optimizer.zero_grad()
                xref = self.memory_x[t].to(strategy.device)
                yref = self.memory_y[t].to(strategy.device)
                out = avalanche_forward(strategy.model, xref, self.memory_tid[t])
                loss = torch.nn.functional.mse_loss(out, yref)
                loss.backward()

                G.append(
                    torch.cat(
                        [
                            p.grad.flatten()
                            if p.grad is not None
                            else torch.zeros(p.numel(), device=strategy.device)
                            for p in strategy.model.parameters()
                        ],
                        dim=0,
                    )
                )

            self.G = torch.stack(G)  # (experiences, parameters)

    @torch.no_grad()
    def after_backward(self, strategy, **kwargs):
        """
        Project gradient based on reference gradients
        """

        if strategy.clock.train_exp_counter > 0:
            g = torch.cat(
                [
                    p.grad.flatten()
                    if p.grad is not None
                    else torch.zeros(p.numel(), device=strategy.device)
                    for p in strategy.model.parameters()
                ],
                dim=0,
            )

            to_project = (torch.mv(self.G, g) < 0).any()
        else:
            to_project = False

        if to_project:
            v_star = self.solve_quadprog(g).to(strategy.device)

            num_pars = 0  # reshape v_star into the parameter matrices
            for p in strategy.model.parameters():
                curr_pars = p.numel()
                if p.grad is not None:
                    p.grad.copy_(v_star[num_pars : num_pars + curr_pars].view(p.size()))
                num_pars += curr_pars

            assert num_pars == v_star.numel(), "Error in projecting gradient"

    def after_training_exp(self, strategy, **kwargs):
        """
        Save a copy of the model after each experience
        """

        self.update_memory(
            strategy.experience.dataset,
            strategy.clock.train_exp_counter,
            strategy.train_mb_size,
            strategy,
        )

    @torch.no_grad()
    def update_memory(self, dataset, t, batch_size, strategy):
        """
        Update replay memory with patterns from current experience.
        """
        collate_fn = dataset.collate_fn if hasattr(dataset, "collate_fn") else None
        dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        tot = 0
        for mbatch in dataloader:
            x, y, tid = mbatch[0], mbatch[1], mbatch[-1]
            y = strategy.model(x.to(strategy.device), tid).to(
                "cpu"
            )  # store logits instead of labels
            if tot + x.size(0) <= self.patterns_per_experience:
                if t not in self.memory_x:
                    self.memory_x[t] = x.clone()
                    self.memory_y[t] = y.clone()
                    self.memory_tid[t] = tid.clone()
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x), dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], y), dim=0)
                    self.memory_tid[t] = torch.cat((self.memory_tid[t], tid), dim=0)

            else:
                diff = self.patterns_per_experience - tot
                if t not in self.memory_x:
                    self.memory_x[t] = x[:diff].clone()
                    self.memory_y[t] = y[:diff].clone()
                    self.memory_tid[t] = tid[:diff].clone()
                else:
                    self.memory_x[t] = torch.cat((self.memory_x[t], x[:diff]), dim=0)
                    self.memory_y[t] = torch.cat((self.memory_y[t], y[:diff]), dim=0)
                    self.memory_tid[t] = torch.cat(
                        (self.memory_tid[t], tid[:diff]), dim=0
                    )
                break
            tot += x.size(0)

    def solve_quadprog(self, g):
        """
        Solve quadratic programming with current gradient g and
        gradients matrix on previous tasks G.
        Taken from original code:
        https://github.com/facebookresearch/GradientEpisodicMemory/blob/master/model/gem.py
        """

        memories_np = self.G.cpu().double().numpy()
        gradient_np = g.cpu().contiguous().view(-1).double().numpy()
        t = memories_np.shape[0]
        P = np.dot(memories_np, memories_np.transpose())
        P = 0.5 * (P + P.transpose()) + np.eye(t) * 1e-3
        q = np.dot(memories_np, gradient_np) * -1
        G = np.eye(t)
        h = np.zeros(t) + self.memory_strength
        v = quadprog.solve_qp(P, q, G, h)[0]
        v_star = np.dot(v, memories_np) + gradient_np

        return torch.from_numpy(v_star).float()


class LogitGEM(SupervisedTemplate):
    """Gradient Episodic Memory (GEM) strategy.

    See GEM plugin for details.
    This strategy does not use task identities.
    """

    def __init__(
        self,
        model,
        optimizer,
        criterion,
        patterns_per_exp: int,
        memory_strength: float = 0.5,
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins=None,
        evaluator=default_evaluator(),
        eval_every=-1,
        **base_kwargs
    ):
        """Init.

        :param model: The model.
        :param optimizer: The optimizer to use.
        :param criterion: The loss criterion to use.
        :param patterns_per_exp: number of patterns per experience in the memory
        :param memory_strength: offset to add to the projection direction
            in order to favour backward transfer (gamma in original paper).
        :param train_mb_size: The train minibatch size. Defaults to 1.
        :param train_epochs: The number of training epochs. Defaults to 1.
        :param eval_mb_size: The eval minibatch size. Defaults to 1.
        :param device: The device to use. Defaults to None (cpu).
        :param plugins: Plugins to be added. Defaults to None.
        :param evaluator: (optional) instance of EvaluationPlugin for logging
            and metric computations.
        :param eval_every: the frequency of the calls to `eval` inside the
            training loop. -1 disables the evaluation. 0 means `eval` is called
            only at the end of the learning experience. Values >0 mean that
            `eval` is called every `eval_every` epochs and at the end of the
            learning experience.
        :param base_kwargs: any additional
            :class:`~avalanche.training.BaseTemplate` constructor arguments.
        """

        gem = LogitGEMPlugin(patterns_per_exp, memory_strength)
        if plugins is None:
            plugins = [gem]
        else:
            plugins.append(gem)

        super().__init__(
            model,
            optimizer,
            criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs
        )
