def format_class_order_from_config(config):
    if config.name == "cifar110":
        class_order = config.fixed_class_order
        class_order = class_order[1:]  # remove pretraining classes
        fixed_class_order = []
        for task_classes in class_order:
            task_classes = [c - 10 for c in task_classes]
            fixed_class_order += task_classes
    elif (
        config.name == "split_mnist"
        or config.name == "rotated_mnist"
        or config.name == "split_mini_ecoset"
        or config.name == "ecoset"
        or config.name == 'tinyimagenet'
    ):
        class_order = config.fixed_class_order
        fixed_class_order = []
        for task_classes in class_order:
            fixed_class_order += task_classes
    else:
        raise NotImplementedError

    return fixed_class_order


def format_class_order_from_benchmark(fixed_class_order, name):
    fixed_class_order = [
        list(classes) for classes in fixed_class_order
    ]  # convert to list from set
    if name == "cifar110":
        class_order = fixed_class_order
        class_order = class_order[1:]  # remove pretraining classes
        fixed_class_order = []
        for task_classes in class_order:
            task_classes = [c - 10 for c in task_classes]
            fixed_class_order += task_classes
    elif (
        name == "split_mnist"
        or name == "rotated_mnist"
        or name == "split_mini_ecoset"
        or name == "ecoset"
        or name == 'tinyimagenet'
    ):
        class_order = fixed_class_order
        fixed_class_order = []
        for task_classes in class_order:
            fixed_class_order += task_classes
    else:
        raise NotImplementedError

    return fixed_class_order
