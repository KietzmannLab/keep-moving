# calls the script autograd_ff_fix_grad_single.py for each pair of lambdas
import numpy as np
import subprocess

# lambdas for ewc between 0 and 100000, log scale 1000 samples
lambdas_ewc = np.logspace(0, 5, 1000) - 1  # include 0
print(lambdas_ewc)

for lambda_ewc in lambdas_ewc:
    # make call to script
    subprocess.call(
        [
            "sbatch",
            "submit_ewc.sh",
            "--ewc_lambda",
            str(lambda_ewc),
            "--logdir",
            "linear_ewc",
        ]
    )
