import avalanche as avl
from strategies import ActivityReplay
from strategies import NullspaceRegularisationLearning
from strategies import NullspaceReplay
from strategies import LogitReplay
from strategies import LogitGEM
from strategies import ClampNullspace
from strategies import ClampNullspaceReplay
from strategies import EWCNullspaceMovement


def instantiate_strategy(
    config, model, optimizer, device, criterion, evaluation_plugin, **kwargs
):
    algo_cfg = config.algorithm
    scenario_cfg = config.scenario

    print(algo_cfg)

    if algo_cfg.name == "naive":
        cl_strategy = avl.training.Naive(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            train_epochs=scenario_cfg.episodes,
            train_mb_size=algo_cfg.train_mb_size,
            eval_mb_size=algo_cfg.test_mb_size,
            device=device,
            evaluator=evaluation_plugin,
            **kwargs,
        )

    elif algo_cfg.name == "si":
        cl_strategy = avl.training.SynapticIntelligence(
            model,
            optimizer,
            criterion,
            si_lambda=algo_cfg.si_lambda,
            eps=algo_cfg.si_eps,
            train_mb_size=algo_cfg.train_mb_size,
            train_epochs=scenario_cfg.episodes,
            eval_mb_size=algo_cfg.test_mb_size,
            device=device,
            evaluator=evaluation_plugin,
            **kwargs,
        )

    elif algo_cfg.name == "lwf":
        cl_strategy = avl.training.LwF(
            model,
            optimizer,
            criterion,
            alpha=algo_cfg.lwf_alpha,
            temperature=algo_cfg.lwf_temperature,
            train_mb_size=algo_cfg.train_mb_size,
            train_epochs=scenario_cfg.episodes,
            eval_mb_size=algo_cfg.test_mb_size,
            device=device,
            evaluator=evaluation_plugin,
            **kwargs,
        )

    elif algo_cfg.name == "ewc":
        cl_strategy = avl.training.EWC(
            model,
            optimizer,
            criterion,
            ewc_lambda=algo_cfg.ewc_lambda,
            train_mb_size=algo_cfg.train_mb_size,
            train_epochs=scenario_cfg.episodes,
            eval_mb_size=algo_cfg.test_mb_size,
            device=device,
            evaluator=evaluation_plugin,
            **kwargs,
        )
    elif algo_cfg.name == "gem":
        cl_strategy = avl.training.GEM(
            model,
            optimizer,
            criterion,
            patterns_per_exp=algo_cfg.gem_patterns_per_experience,
            memory_strength=algo_cfg.gem_memory_strength,
            train_mb_size=algo_cfg.train_mb_size,
            train_epochs=scenario_cfg.episodes,
            eval_mb_size=algo_cfg.test_mb_size,
            device=device,
            evaluator=evaluation_plugin,
            **kwargs,
        )

    elif algo_cfg.name == "agem":
        cl_strategy = avl.training.AGEM(
            model,
            optimizer,
            criterion,
            patterns_per_exp=algo_cfg.gem_patterns_per_experience,
            sample_size=algo_cfg.gem_sample_size,
            train_mb_size=algo_cfg.train_mb_size,
            train_epochs=scenario_cfg.episodes,
            eval_mb_size=algo_cfg.test_mb_size,
            device=device,
            evaluator=evaluation_plugin,
            **kwargs,
        )

    elif algo_cfg.name == "replay":
        cl_strategy = avl.training.Replay(
            model,
            optimizer,
            criterion,
            mem_size=algo_cfg.replay_mem_size,
            train_mb_size=algo_cfg.train_mb_size,
            train_epochs=scenario_cfg.episodes,
            eval_mb_size=algo_cfg.test_mb_size,
            device=device,
            evaluator=evaluation_plugin,
            **kwargs,
        )

    elif algo_cfg.name == "activity_replay":
        cl_strategy = ActivityReplay(
            model,
            optimizer,
            criterion,
            mem_size=algo_cfg.replay_mem_size,
            replay_batch_size=algo_cfg.replay_batch_size,
            exclude_list=algo_cfg.exclude_list,
            replay_activity_lambda=algo_cfg.replay_activity_lambda,
            train_mb_size=algo_cfg.train_mb_size,
            train_epochs=scenario_cfg.episodes,
            eval_mb_size=algo_cfg.test_mb_size,
            device=device,
            evaluator=evaluation_plugin,
            **kwargs,
        )

    elif algo_cfg.name == "nullspace_regularisation":
        cl_strategy = NullspaceRegularisationLearning(
            model,
            optimizer,
            criterion,
            nullspace_lambda=algo_cfg.nullspace_lambda,
            layer=algo_cfg.nullspace_layer,
            rcond=algo_cfg.nullspace_rcond,
            scaled=algo_cfg.scaled,
            control=algo_cfg.control,
            no_bias=algo_cfg.no_bias,
            reg_criterion=algo_cfg.reg_criterion,
            train_mb_size=algo_cfg.train_mb_size,
            train_epochs=scenario_cfg.episodes,
            eval_mb_size=algo_cfg.test_mb_size,
            device=device,
            evaluator=evaluation_plugin,
            **kwargs,
        )

    elif algo_cfg.name == "nullspace_regularisation_replay":
        cl_strategy = NullspaceReplay(
            model,
            optimizer,
            criterion,
            mem_size=algo_cfg.replay_mem_size,
            replay_batch_size=algo_cfg.replay_batch_size,
            nullspace_lambda=algo_cfg.nullspace_lambda,
            layer=algo_cfg.nullspace_layer,
            scaled=algo_cfg.scaled,
            control=algo_cfg.control,
            nullspace_rcond=algo_cfg.nullspace_rcond,
            train_mb_size=algo_cfg.train_mb_size,
            train_epochs=scenario_cfg.episodes,
            eval_mb_size=algo_cfg.test_mb_size,
            device=device,
            evaluator=evaluation_plugin,
            **kwargs,
        )

    elif algo_cfg.name == "logit_replay":
        cl_strategy = LogitReplay(
            model,
            optimizer,
            criterion,
            mem_size=algo_cfg.replay_mem_size,
            auxiliary_loss=algo_cfg.auxiliary_loss,
            replay_batch_size=algo_cfg.replay_batch_size,
            replay_lambda=algo_cfg.replay_lambda,
            train_mb_size=algo_cfg.train_mb_size,
            train_epochs=scenario_cfg.episodes,
            eval_mb_size=algo_cfg.test_mb_size,
            device=device,
            evaluator=evaluation_plugin,
            **kwargs,
        )

    elif algo_cfg.name == "logit_gem":
        cl_strategy = LogitGEM(
            model,
            optimizer,
            criterion,
            patterns_per_exp=algo_cfg.gem_patterns_per_experience,
            memory_strength=algo_cfg.gem_memory_strength,
            train_mb_size=algo_cfg.train_mb_size,
            train_epochs=scenario_cfg.episodes,
            eval_mb_size=algo_cfg.test_mb_size,
            device=device,
            evaluator=evaluation_plugin,
        )

    elif algo_cfg.name == "clamp_nullspace":
        cl_strategy = ClampNullspace(
            model,
            optimizer,
            criterion,
            mem_size=algo_cfg.replay_mem_size,
            replay_batch_size=algo_cfg.replay_batch_size,
            nullspace_lambda=algo_cfg.nullspace_lambda,
            layer=algo_cfg.nullspace_layer,
            scaled=algo_cfg.scaled,
            control=algo_cfg.control,
            nullspace_rcond=algo_cfg.nullspace_rcond,
            train_mb_size=algo_cfg.train_mb_size,
            train_epochs=scenario_cfg.episodes,
            eval_mb_size=algo_cfg.test_mb_size,
            device=device,
            evaluator=evaluation_plugin,
            **kwargs,
        )

    elif algo_cfg.name == "clamp_nullspace_replay":
        cl_strategy = ClampNullspaceReplay(
            model,
            optimizer,
            criterion,
            mem_size=algo_cfg.replay_mem_size,
            replay_batch_size=algo_cfg.replay_batch_size,
            nullspace_lambda=algo_cfg.nullspace_lambda,
            layer=algo_cfg.nullspace_layer,
            scaled=algo_cfg.scaled,
            control=algo_cfg.control,
            nullspace_rcond=algo_cfg.nullspace_rcond,
            train_mb_size=algo_cfg.train_mb_size,
            train_epochs=scenario_cfg.episodes,
            eval_mb_size=algo_cfg.test_mb_size,
            device=device,
            evaluator=evaluation_plugin,
            **kwargs,
        )

    elif algo_cfg.name == "ewc_nullspace":
        cl_strategy = EWCNullspaceMovement(
            model,
            optimizer,
            criterion,
            ewc_lambda=algo_cfg.ewc_lambda,
            layer=algo_cfg.nullspace_layer,
            nullspace_lambda=algo_cfg.nullspace_lambda,
            rcond=algo_cfg.nullspace_rcond,
            num_workers=algo_cfg.num_workers,
            mem_size=algo_cfg.replay_mem_size,
            batch_size_mem=algo_cfg.replay_batch_size,
            train_mb_size=algo_cfg.train_mb_size,
            train_epochs=scenario_cfg.episodes,
            eval_mb_size=algo_cfg.test_mb_size,
            device=device,
            evaluator=evaluation_plugin,
            **kwargs,
        )

    else:
        raise NotImplementedError(f"Strategy {algo_cfg.name} not implemented.")

    return cl_strategy
