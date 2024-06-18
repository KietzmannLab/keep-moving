import pandas as pd
import os


def load_history(is_wandb, projectdir, logpath):
    if not is_wandb:
        history = pd.read_csv(logpath)
    else:
        import wandb

        api = wandb.Api()
        run = api.run(logpath)
        name = run.name
        logpath = f"logs/{projectdir}/{name}"
        # make sure path exists
        if not os.path.exists(logpath):
            os.makedirs(logpath, exist_ok=False)
            history = pd.DataFrame(run.scan_history())
            history.to_csv(f"{logpath}/wandb_history.csv")
        else:
            logpath = f"logs/{projectdir}/{name}"
            history = pd.read_csv(f"{logpath}/wandb_history.csv")
    return history, logpath, name
