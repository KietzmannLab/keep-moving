# calls the script autograd_ff_fix_grad_single.py for each pair of lambdas
import numpy as np
import subprocess

lambdas_filter = np.linspace(0, 1.0, 33)
lambdas_anti_filter = np.linspace(0, 1.0, 33)

for lambda_filter in lambdas_filter:
    for lambda_anti_filter in lambdas_anti_filter:
        # make call to script
        subprocess.call(
            [
                "sbatch",
                "submit_gradient_decomposition.sh",
                "--lambda_filter",
                str(lambda_filter),
                "--lambda_anti_filter",
                str(lambda_anti_filter),
                "--logdir",
                "gradient_decomposition",
            ]
        )
