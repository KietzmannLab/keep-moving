from load_data import load_data, map_classes_for_task
from model import Model
from subspace_estimation import (
    compute_subspace,
    project_gradients,
    project_gradients_on_vector,
)
from custom_adam import initialize_adam, update_parameters

import random
import torch
from tqdm import tqdm
import numpy as np
import wandb
from torchmetrics.functional.classification.accuracy import accuracy
from functools import partial
import os
import yaml

import hydra
from omegaconf import DictConfig, OmegaConf

# automatically determine DEVICE. Order of preference: cuda, mps, cpu
# set globally here so it is consistent across the whole script

if torch.cuda.is_available():
    DEVICE = "cuda"
# else if mps is available
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"


def train_step(
    global_batch,
    model,
    optimizer,
    criterion,
    x,
    y,
    t,
    classes,
    metric_dict={},
    t0_eval_data=None,
    t0_train_data=None,
    regularizer=None,
    subspaces=None,
    use_projected_gradients=False,
    use_t0_projection=False,
    knobs=None,
    commit=False,
    update_backbone=True,
):
    model.train()

    # define all variables that are possibly unset as None here. This lets us check if they are set correctly when needed later
    t0_grads_eval = None
    t0_grads_train = None
    grads_projected = None
    grads_regularized = None
    grads_t0_projected = None

    # -------------- t0 eval metrics --------------

    if not t0_eval_data is None:
        model.zero_grad()
        x_t0, y_t0 = t0_eval_data
        x_t0 = x_t0.to(DEVICE)
        y_t0 = y_t0.to(DEVICE)
        y_t0_hat = model(x_t0, 0)
        # compute accuracy
        # get t0 grads
        loss_t0 = criterion(y_t0_hat, y_t0)
        loss_l1_t0 = model.l1_penalty()
        loss_t0 += loss_l1_t0
        t0_grads_eval = torch.autograd.grad(
            loss_t0, model.blocks.parameters(), allow_unused=True
        )

        t0_sample_acc = accuracy(
            y_t0_hat, y_t0, task="multiclass", num_classes=len(classes[0])
        ).item()
        loss_t0.item()
    else:
        t0_sample_acc = torch.nan
        loss_t0 = torch.nan
    wandb.log(
        {
            "performance/t0 eval sample accuracy": t0_sample_acc,
            "performance/t0 eval sample loss": loss_t0,
        },
        commit=False,
    )  # wait for remainder of metrics before commit

    # -------------- t0 train metrics --------------

    if not t0_train_data is None:
        model.zero_grad()
        x_t0, y_t0 = t0_train_data
        x_t0 = x_t0.to(DEVICE)
        y_t0 = y_t0.to(DEVICE)
        y_t0_hat = model(x_t0, 0)
        # compute accuracy
        # get t0 grads
        loss_t0 = criterion(y_t0_hat, y_t0)
        loss_l1_t0 = model.l1_penalty()
        loss_t0 += loss_l1_t0
        t0_grads_train = torch.autograd.grad(
            loss_t0, model.blocks.parameters(), allow_unused=True
        )

        t0_sample_acc = accuracy(
            y_t0_hat, y_t0, task="multiclass", num_classes=len(classes[0])
        ).item()
        loss_t0.item()
    else:
        t0_sample_acc = torch.nan
        loss_t0 = torch.nan
    wandb.log(
        {
            "performance/t0 train sample accuracy": t0_sample_acc,
            "performance/t0 train sample loss": loss_t0,
        },
        commit=False,
    )  # wait for remainder of metrics before commit

    # -------------- train step --------------

    # make very sure our metric gradients do not pollute the model gradients
    model.zero_grad()

    y_hat = model(x, t)
    loss = criterion(y_hat, y)
    loss_l1 = model.l1_penalty()
    # log l1
    wandb.log({"performance/l1 loss": loss_l1}, commit=False)
    loss += loss_l1
    # compute gradients w/o continual learning regularisation
    loss.backward()

    task_gradients = [torch.clone(p.grad).detach() for p in model.blocks.parameters()]

    # -------------- regularisation or projection --------------

    if (subspaces is not None) or (regularizer is not None) or use_t0_projection:
        # get projected gradients
        if subspaces is not None:
            # project gradients
            with torch.no_grad():
                grads_projected = project_gradients(subspaces, task_gradients, knobs)
        if regularizer is not None:
            model.blocks.zero_grad()  # make sure model backbone is clean
            reg_loss = regularizer(model, y)
            reg_grad = torch.autograd.grad(reg_loss, model.blocks.parameters())
            grads_regularized = [
                gt + gr.clone().detach()
                for gt, gr in zip(
                    task_gradients, reg_grad
                )  # make sure all gradients are detached to avoid polluting them accidentally
            ]
        if use_t0_projection:
            assert t0_grads_train is not None, "t0 train data must be provided"
            grads_t0_projected = project_gradients_on_vector(
                task_gradients, t0_grads_train, knobs, t0_bias=None
            )

        # here we decide which regularisation to use. Regularizer takes precedence over subspaces (but subspaces will still be used for metrics if both are available)
        if regularizer is not None:
            ps_with_reg = grads_regularized
        elif use_projected_gradients:
            ps_with_reg = grads_projected
        elif use_t0_projection:
            ps_with_reg = grads_t0_projected
        else:
            ps_with_reg = task_gradients

        # -------------- gradient metrics compare with unregularised --------------

        # for each layer, compute cosine similarity and magnitude difference between grad with and without regularisation
        # this assesses the similarity between the "true" task gradient and the update the model is actually making
        with torch.no_grad():
            norms_without_reg, norms_with_reg, cosine_diffs = [], [], []
            for i, (p_without_reg, p_with_reg) in enumerate(
                zip(task_gradients, ps_with_reg)
            ):
                # cosine similarity
                cos_sim = torch.nn.functional.cosine_similarity(
                    p_with_reg.flatten(), p_without_reg.flatten(), dim=0
                ).item()
                # magnitudes
                norm_reg = torch.norm(p_with_reg).item()
                norm_without_reg = torch.norm(p_without_reg).item()
                # append
                norms_without_reg.append(norm_without_reg)
                norms_with_reg.append(norm_reg)
                cosine_diffs.append(cos_sim)

            # log to wandb
            named_params = [n for n, _ in model.blocks.named_parameters()]
            for i, (n, c, c1) in enumerate(
                zip(norms_without_reg, norms_with_reg, cosine_diffs)
            ):
                wandb.log(
                    {
                        f"gradient norms/{model.layer_names[named_params[i]]} norm without reg": n,
                        f"gradient norms/{model.layer_names[named_params[i]]} norm with reg": c,
                        f"gradient cosines/{model.layer_names[named_params[i]]} cosim": c1,
                    },
                    commit=False,
                )

        # -------------- set gradients for update step --------------

        # decide which gradients to actually use for training
        # if subspaces have been computed and should be used for training, use projected gradients
        if use_projected_gradients and subspaces is not None:
            if regularizer is not None:
                raise ValueError(
                    "subspaces and regularizer cannot both be used for training"
                )
            update_grads = grads_projected
        elif regularizer is not None:
            update_grads = grads_regularized
        elif use_t0_projection:
            update_grads = grads_t0_projected
        else:
            update_grads = task_gradients

        # set the actual gradients to use for training. Will be updated with optimizer
        for p, g in zip(model.blocks.parameters(), update_grads):
            p.grad = g

        # -------------- compute nullspace and range norms --------------
        # if subspaces are available, then compute how much of the gradient lives in the range vs subspace at each layer
        if subspaces is not None:
            with torch.no_grad():
                nullspace_gradients = project_gradients(subspaces, update_grads)
                range_gradients = [
                    g - ng for g, ng in zip(update_grads, nullspace_gradients)
                ]
                norm_range_gradients = [torch.norm(g).item() for g in range_gradients]
                norm_nullspace_gradients = [
                    torch.norm(g).item() for g in nullspace_gradients
                ]

                named_params = [n for n, _ in model.blocks.named_parameters()]
                for i, (grad_r, grad_n) in enumerate(
                    zip(norm_range_gradients, norm_nullspace_gradients)
                ):
                    wandb.log(
                        {
                            f"gradient norms/{model.layer_names[named_params[i]]} norm range": grad_r,
                            f"gradient norms/{model.layer_names[named_params[i]]} norm nullspace": grad_n,
                        },
                        commit=False,
                    )

    # -------------- t0 gradient metrics --------------

    if t0_eval_data is not None:
        # compute cosine similarity between t0 grads and current grads
        t0_cosine_sims_eval = []
        t0_cosine_sims_train = []
        for i, ((pname, p), p_t0_eval, p_t0_train) in enumerate(
            zip(model.blocks.named_parameters(), t0_grads_eval, t0_grads_train)
        ):
            # cosine similarity
            cos_sim_train = torch.nn.functional.cosine_similarity(
                p_t0_train.flatten(), p.grad.flatten(), dim=0
            ).item()
            cos_sim_eval = torch.nn.functional.cosine_similarity(
                p_t0_eval.flatten(), p.grad.flatten(), dim=0
            ).item()
            t0_cosine_sims_train.append(cos_sim_train)
            t0_cosine_sims_eval.append(cos_sim_eval)

        named_params = [n for n, _ in model.blocks.named_parameters()]
        for i, (c0_train, c0_eval) in enumerate(
            zip(t0_cosine_sims_train, t0_cosine_sims_eval)
        ):
            wandb.log(
                {
                    f"gradient cosines t0/{model.layer_names[named_params[i]]} t0 cosim train": c0_train,
                    f"gradient cosines t0/{model.layer_names[named_params[i]]} t0 cosim eval": c0_eval,
                },
                commit=False,
            )
    # log batch
    wandb.log({"batch": global_batch}, commit=commit)

    # update model
    backbone_optim, readouts_optim = optimizer
    if update_backbone:
        gs = [p.grad for p in model.blocks.parameters()]
        updates = backbone_optim(gs)
        # if we use gradient_projection or t0 projection, make sure that the gradient with momentum is still legal
        if use_projected_gradients and subspaces is not None:
            updates = project_gradients(subspaces, updates, knobs)
        elif use_t0_projection:
            updates = project_gradients_on_vector(updates, t0_grads_train, knobs)

        update_parameters(model.blocks.parameters(), updates)
    gs = [p.grad for p in model.readouts.parameters()]
    updates = readouts_optim(gs)
    update_parameters(model.readouts.parameters(), updates)

    metric_results = {k: metric(y_hat, y) for k, metric in metric_dict.items()}
    return loss.item(), metric_results


@torch.no_grad()
def test_step(model, criterion, x, y, t, metric_dict={}):
    model.eval()
    y_hat = model(x, t)
    loss = criterion(y_hat, y)
    metric_results = {k: metric(y_hat, y) for k, metric in metric_dict.items()}
    return loss.item(), metric_results


def accumulate_epoch_metrics(metric_dict):
    for k, v in metric_dict.items():
        if len(v) > 0:
            metric_dict[k] = np.mean(v)
        else:
            metric_dict[k] = torch.nan

    return metric_dict


def train(
    global_batch,
    task,
    classes,
    model,
    optimizer,
    criterion,
    train_loader,
    test_loaders,
    epochs=10,
    regularizer=None,
    train_metrics=dict(),
    val_metrics=dict(),
    grad_subspaces=None,
    use_projected_gradients=False,
    use_t0_projection=False,
    knobs=None,
    t0_eval_data=None,
    t0_train_data=None,
    recompute_subspace=None,
    recompute_every_n_epochs=None,
    warmup_epochs=0,
    recompute_every_n_steps=None,
):
    print(f"training task {task}")
    print(f"classes: {classes}")
    print(f"for {epochs} epochs")

    all_metric_results = dict()
    for k, v in train_metrics.items():
        all_metric_results[k] = []
    for k, v in val_metrics.items():
        for k2, v2 in v.items():
            all_metric_results[f"performance/test_loss_task_{k}"] = []
            all_metric_results[k2] = []

    # always track train loss
    all_metric_results["performance/train_loss"] = []

    for epoch in range(epochs):
        if task > 0 and epoch < warmup_epochs:
            update_backbone = False
        else:
            update_backbone = True
        train_metric_results = {k: [] for k in train_metrics.keys()}

        # always log loss
        train_metric_results["performance/train_loss"] = []

        nbatches = len(train_loader)
        for batch_id, (x, y) in enumerate(tqdm(train_loader)):
            task_batch = len(train_loader) * epoch + batch_id + 1
            global_batch += 1
            if not recompute_subspace is None:
                if not recompute_every_n_steps is None:
                    if not epoch < warmup_epochs:
                        if (task_batch % recompute_every_n_steps) == 0:
                            print("recomputing subspace")
                            grad_subspaces = recompute_subspace()

            y = map_classes_for_task(y, classes[task])
            # move to device
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            train_loss, metres = train_step(
                global_batch,
                model,
                optimizer,
                criterion,
                x,
                y,
                task,
                classes,
                train_metrics,
                t0_eval_data=t0_eval_data,
                t0_train_data=t0_train_data,
                regularizer=regularizer,
                subspaces=grad_subspaces,
                use_projected_gradients=use_projected_gradients,
                use_t0_projection=use_t0_projection,
                knobs=knobs,
                commit=(not batch_id == nbatches - 1),
                update_backbone=update_backbone,
            )  # do not commit on last batch, so we can log epoch metrics
            # add grad stats

            train_metric_results["performance/train_loss"].append(train_loss)
            for k, v in metres.items():
                train_metric_results[k].append(v.item())

        train_epoch_metric_results = accumulate_epoch_metrics(train_metric_results)
        for k, v in train_epoch_metric_results.items():
            print(f"{k}: {v}")

        test_metrics = dict()
        for test_task, test_loader in enumerate(test_loaders):
            task_metrics = val_metrics[test_task]
            val_metric_results = {k: [] for k in task_metrics.keys()}
            # always track loss
            val_metric_results[f"performance/test_loss_task_{test_task}"] = []
            for x, y in test_loader:
                y = map_classes_for_task(y, classes[test_task])
                # move to device
                x = x.to(DEVICE)
                y = y.to(DEVICE)
                loss, metres = test_step(
                    model, criterion, x, y, test_task, task_metrics
                )
                val_metric_results[f"performance/test_loss_task_{test_task}"].append(
                    loss
                )
                for k, v in metres.items():
                    val_metric_results[k].append(v.item())

            val_epoch_metric_results = accumulate_epoch_metrics(val_metric_results)
            # add to test_metrics
            for k, v in val_epoch_metric_results.items():
                test_metrics[k] = v

        for k, v in test_metrics.items():
            print(f"{k}: {v}")

        # add epoch and task
        wandb.log(
            {
                "epoch": epoch,
                "task": task,
            },
            commit=False,
        )

        # join all metrics and log to wandb
        all_metrics = {**train_epoch_metric_results, **test_metrics}
        wandb.log(all_metrics)
        for k, v in all_metrics.items():
            all_metric_results[k].append(v)

        if not recompute_subspace is None:
            if not recompute_every_n_epochs is None:
                if (epoch + 1) % recompute_every_n_epochs == 0:
                    print("recomputing subspace")
                    grad_subspaces = recompute_subspace()
                else:
                    print("not recomputing subspace")

    return all_metric_results, global_batch


@hydra.main(config_path="config", config_name="default", version_base="1.2")
def main(cfg):
    # fixed for all experiments, dont touch

    # print device used
    print(f"Using device: {DEVICE}")

    # override defaults with cfg
    project = cfg.project
    batchsize = cfg.batchsize
    nworkers = cfg.nworkers
    nsplits = cfg.nsplits
    shuffle_classes = cfg.shuffle_classes
    seed = cfg.seed
    epochs = cfg.epochs
    warmup_epochs = cfg.warmup_epochs
    with_bias = cfg.with_bias
    l1 = cfg.l1
    channels_per_block = cfg.channels_per_block
    dense_size = cfg.dense_size
    lr = cfg.lr
    ewc_lambda = cfg.ewc_lambda
    subspace_epochs = cfg.subspace_epochs
    subspace_batchsize = cfg.subspace_batchsize
    var_explained = cfg.var_explained
    nsamples_t0_eval = cfg.nsamples_t0_eval
    nsamples_t0_train = cfg.nsamples_t0_train
    recompute_every_n_epochs = cfg.recompute_every_n_epochs
    recompute_every_n_steps = cfg.recompute_every_n_steps
    use_projected_gradients = cfg.use_projected_gradients
    use_t0_projection = cfg.use_t0_projection
    knob_null = cfg.knob_null
    knob_range = cfg.knob_range

    cfg_dict = OmegaConf.to_container(cfg)

    if not knob_null is None and not knob_range is None:
        knobs = (knob_null, knob_range)
    else:
        knobs = None

    if subspace_epochs < 1:
        subspace_epochs = None

    name = ""
    is_baseline = True
    if not ewc_lambda is None:
        is_baseline = False
        name += f"ewc_lambda_{ewc_lambda}_"
    if use_projected_gradients:
        is_baseline = False
        name += f"subspace_epochs_{subspace_epochs}_sub_bs_{subspace_batchsize}_var_exp_{var_explained}"
    if use_t0_projection:
        is_baseline = False
        name += f"t0_projection_"
        name += f"t0_train_samples_{nsamples_t0_train}"
    if knobs is not None:
        frac_null, frac_range = knobs
        name += f"_knobs_{frac_null}_{frac_range}"
    if recompute_every_n_epochs is not None:
        name += f"_recompute_every_{recompute_every_n_epochs}_epochs"
    if recompute_every_n_steps is not None:
        name += f"_recompute_every_{recompute_every_n_steps}_steps"
    if is_baseline:
        name = "baseline_" + name
    if warmup_epochs > 0:
        name += f"warmup_{warmup_epochs}_"

    name += f"params_{batchsize}_{nworkers}_{nsplits}_{shuffle_classes}_{seed}_{epochs}_{with_bias}_{l1}"

    # create local log dir for run
    # check whether logdir exists
    base_logdir = os.path.join("logs", project, name)
    logdir = base_logdir
    counter = 1
    while os.path.exists(logdir):
        logdir = base_logdir + f"_{counter}"
        counter += 1
    os.makedirs(logdir)

    # fix all the seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    data, classes = load_data(
        batchsize=batchsize,
        nworkers=nworkers,
        nsplits=nsplits,
        shuffle_classes=shuffle_classes,
        seed=seed,
    )
    cfg_dict["classes"] = classes

    run = wandb.init(
        project=project,
        config=cfg_dict,
        name=name,
        settings=wandb.Settings(code_dir="."),
        dir=logdir,
    )

    print(f"loaded {len(data)} tasks.")

    model = Model(
        device=DEVICE,
        with_bias=with_bias,
        l1=l1,
        channels_per_block=channels_per_block,
        dense_size=dense_size,
    )
    # save checkpoint to logdir
    torch.save(model.state_dict(), os.path.join(logdir, "model_init.pt"))
    wandb.save(os.path.join(logdir, "model_init.pt"))
    # add wandb run id to cfg
    cfg_dict["wandb_id"] = run.id
    cfg_dict["wandb_url"] = run.get_url()
    cfg_dict["wandb_name"] = run.name
    cfg_dict["wandb_project"] = run.project
    cfg_dict["local_logdir"] = logdir
    cfg_dict["hostname"] = os.uname()[1]
    # save as yaml to logdir
    with open(os.path.join(logdir, "cfg.yaml"), "w") as f:
        yaml.dump(cfg_dict, f)
    model.to(DEVICE)

    criterion = torch.nn.CrossEntropyLoss()

    test_loaders = [test_loader for train_loader, test_loader in data]
    train_loaders = [train_loader for train_loader, test_loader in data]

    # random samples from task 0 for evaluation
    eval_loader = torch.utils.data.DataLoader(
        test_loaders[0].dataset,
        batch_size=nsamples_t0_eval,
        shuffle=True,
    )
    eval_x, eval_y = next(iter(eval_loader))
    eval_y = map_classes_for_task(eval_y, classes[0])
    eval_data = (eval_x, eval_y)
    del eval_loader

    # random samples from task 0 for training
    train_loader = torch.utils.data.DataLoader(
        train_loaders[0].dataset,
        batch_size=nsamples_t0_train,
        shuffle=True,
    )
    train_x, train_y = next(iter(train_loader))
    train_y = map_classes_for_task(train_y, classes[0])
    train_data = (train_x, train_y)
    del train_loader

    results = []
    regularizer = None
    grad_subspaces = None
    recompute_subspace = None
    global_batch = 0

    for task, train_loader in enumerate(train_loaders):
        optimizers = [
            initialize_adam(model.blocks.parameters(), stepsize=lr),
            initialize_adam(model.readouts.parameters(), stepsize=lr),
        ]
        train_metrics = {
            "performance/train_accuracy": partial(
                accuracy, task="multiclass", num_classes=len(classes[task])
            ),
        }

        # test metrics per task
        val_metrics = {
            test_task: {
                f"performance/test_accuracy_task_{test_task}": partial(
                    accuracy, task="multiclass", num_classes=len(classes[test_task])
                ),
            }
            for test_task, _ in enumerate(test_loaders)
        }

        result_dict, global_batch = train(
            global_batch=global_batch,
            epochs=epochs,
            task=task,
            classes=classes,
            model=model,
            optimizer=optimizers,
            criterion=criterion,
            train_loader=train_loader,
            test_loaders=test_loaders,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            regularizer=regularizer,
            grad_subspaces=grad_subspaces,
            use_projected_gradients=use_projected_gradients,
            use_t0_projection=use_t0_projection
            and task > 0,  # only use t0 projection for tasks > 0
            knobs=knobs,
            t0_eval_data=eval_data,
            t0_train_data=train_data,
            recompute_subspace=recompute_subspace,
            recompute_every_n_epochs=recompute_every_n_epochs,
            recompute_every_n_steps=recompute_every_n_steps,
            warmup_epochs=warmup_epochs,
        )

        results.append(result_dict)

        if not ewc_lambda is None:
            model.compute_importances(train_loader, task, classes[task])

            def regularizer(model, y):
                return model.ewc_penalty() * ewc_lambda

        if subspace_epochs is not None:
            subspace_loader = torch.utils.data.DataLoader(
                train_loader.dataset, batch_size=subspace_batchsize, shuffle=True
            )
            grad_subspaces = compute_subspace(
                model,
                criterion,
                subspace_loader,
                task,
                classes[task],
                epochs=subspace_epochs,
                var_exp=var_explained,
            )

            recompute_subspace = partial(
                compute_subspace,
                model,
                criterion,
                subspace_loader,
                task,
                classes[task],
                epochs=subspace_epochs,
                var_exp=var_explained,
            )
        else:
            grad_subspaces = None

        # save checkpoint
        torch.save(model.state_dict(), os.path.join(logdir, f"model_task_{task}.pt"))
        wandb.save(os.path.join(logdir, f"model_task_{task}.pt"))

        # --------------------

    # merge dicts
    result_dict = dict()
    for k in results[0].keys():
        result_dict[k] = []
        for result in results:
            result_dict[k].extend(result[k])

    # finish the run
    run.finish()

    print("done.")


if __name__ == "__main__":
    main()
