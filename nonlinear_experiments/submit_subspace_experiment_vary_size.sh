sbatch submit_train.sh project=subspace_estimation_vary_size use_projected_gradients=true recompute_every_n_steps=200 knob_null=1. knob_range=0. channels_per_block=[16,16,32,32,64] dense_size=128
sbatch submit_train.sh project=subspace_estimation_vary_size use_projected_gradients=true recompute_every_n_steps=200 knob_null=1. knob_range=0. channels_per_block=[16,16,32,32,64] dense_size=64
sbatch submit_train.sh project=subspace_estimation_vary_size use_projected_gradients=true recompute_every_n_steps=200 knob_null=1. knob_range=0. channels_per_block=[16,16,32,32,64] dense_size=256
sbatch submit_train.sh project=subspace_estimation_vary_size use_projected_gradients=true recompute_every_n_steps=200 knob_null=1. knob_range=0. channels_per_block=[4,4,16,16,32] dense_size=32

