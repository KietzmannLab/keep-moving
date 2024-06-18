# Readout Misalignment

This repository contains the code for 
"Diagnosing Catastrophe: Large Parts of Accuracy Loss in Continual Learning can be Accounted for by Readout Misalignment".

## Installation

The required python packages are listed in ```requirements.txt```

## Train Models

To train a new model, use the ```train_models.py``` file. 
All available configurations are available in ```configuration/train_conf``` and can be overwritten on the command line.

Experiments are configured with [hydra](hydra.cc)

To replicate the models reported in the paper, use the following commands:

### Main Result (Fig1. A)

```python train_models.py experiment=main_experiment```

### Capacity Result (Fig1. C & D)

```python train_models.py experiment=capacity_experiment_{32, 64, 128, 1024, 2048}```

## Analyze Models

### Diagnostic Readouts

The first analysis step is to get the diagnostic readouts for the model. These can be obtained by running:

```python finetune_models.py project=$project run_name=$name```

```$project``` should be the the project the run is stored in, 
by default the runs for reproducing the main figure are in:

```logs/experiment_main_figure``` for the main result, and in
```logs/experiment_capacity``` for the capacity experiment.

```$name``` will be the name of the runs, which will be the folder name created in ```logs/$project``` when the model was trained. 
(This will be a timestamped version of the name the run was given during training).

### Procrustes Analysis

The second analysis step is to obtain the Procrustes aligned results. These can be obtained with:

```python procrustes_analysis.py project=$project run_name=$name embedding_layer=$embedding_layer```

```embedding_layer``` will be the pre-readout layer for which we will align the embeddings. 
In the case of the Cifar networks used here, this will be ```dense_dropout```.

### Plotting

Scripts to reproduce all plots are included as jupyter notebooks.

## Issues

In case of any questions or if you encounter any unforseen issues, please either open an Issue on this repository or contact me at danthes (at) uos (dot) de.
