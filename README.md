# gnn-multistability

Source code for model multiplicity of Graph Neural Networks in downstream tasks beyond node classification

sample usage for single run

```bash
python train.py n_repeat=2 n_epochs=200 dataset=aspirin model.hidden_dim=8 model.num_layers=2 model=gin
```

sample usage for multi-run

```bash
python train.py n_repeat=2 n_epochs=200 dataset=aspirin,collab,aids model.hidden_dim=8,16,32 model.num_layers=2,3,4 model=gin,gat,gcn -m
```

## Project structure

### config

this folder contains the configuration of the project

the arguments can be configured by modifying the `yaml` files defined here
(not recommended) or by passing them as input parameters
(similar to the sample above). 
Please refer to the documentation of [hydra](https://hydra.cc/docs/intro/) for more information about how to use it.

### dataset

this folder stores the dataset

### multirun

this folder stores the results of multi-runs (when running the code with multiple series of parameters and with `-m`
flag)

### notebooks

this folder contains notebooks that analyze the results

### outputs

this folder contains the results of a singular execution

### src

this folder contains the main code

`main.py` is the entry point of this program and executing it will first cause the program to train the models then it
will evaluate their results with prediction multiplicity metrics

Alternatively `train.py` can be used to only train the models, and then the results can be evaluated with `evaluate.py`
for single runs or `evaluate_multy.py` for multi-runs. Please save the activation functions when using this method.

Both `environment.yml` and `menv.yaml` contain the conda environment used in this project. Each of them is created from
a different system but both should be usable to replicate the results. The former is created from system running windows
10 while the latter is created from ubuntu.

#### common

helper functions

#### data_loaders

tools to load the datasets

The following datasets are fully supported:

| Task                 | Dataset        | Source      |
|----------------------|----------------|-------------|
| Graph Classification | aids           | TUDataset   |
| Graph Classification | enzymes        | TUDataset   |
| Graph Classification | ptc_fm         | TUDataset   |
| Graph Classification | proteins       | TUDataset   |
| link prediction      | ddi            | OGB Dataset |
| link prediction      | collab         | OGB Dataset |
| link prediction      | ppa            | OGB Dataset |
| Graph Regression     | aspirin        | TUDataset   |
| Graph Regression     | salicylic_acid | TUDataset   |
| Graph Regression     | toluene        | TUDataset   |
| Graph Regression     | uracil         | TUDataset   |
| Graph Regression     | naphthalene    | TUDataset   |
| Graph Regression     | alchemy        | TUDataset   |
| Graph Regression     | zinc           | TUDataset   |
| Graph Regression     | qm9            | TUDataset   |

Code and config contain some other datasets, but they may or may not work correctly
(some more infor are provided in `src/data_loaders/get_dataset.py` in `DATASETS` dictionary next to their names)

#### evaluation

folder containing the code related to analysing the prediction multiplicity of the models

#### models

folder containing the gnn models and code used to train them

The following GNN models are implemented: `gat`, `gatedgcn`, `gcn`, `gin`, `graphsage`, and `resgatedgcn`.
Please note that `gatedgcn` only works
when the input shape is smaller than their hidden dimension and `gin` model does not work with `ddi` dataset.

#### plots

helper functions used for plotting things (used in the evaluation step)

MSELoss:
[2023-10-11 11:46:14,159][models.train_models][INFO] - Model has 625 parameters (625 trainable).
[2023-10-11 11:46:14,159][models.train_models][INFO] - time=124.87 epoch=0: train_loss=0.006, valid_loss=0.000
[2023-10-11 11:47:57,008][models.train_models][INFO] - time=227.72 epoch=1: train_loss=0.000, valid_loss=0.000
[2023-10-11 11:49:40,984][models.train_models][INFO] - time=331.69 epoch=2: train_loss=0.000, valid_loss=0.000
[2023-10-11 11:51:25,111][models.train_models][INFO] - time=435.82 epoch=3: train_loss=0.000, valid_loss=0.000
[2023-10-11 11:53:04,175][models.train_models][INFO] - time=534.88 epoch=4: train_loss=0.000, valid_loss=0.000
[2023-10-11 11:53:04,178][models.train_models][INFO] - Reverting to model with best val loss
[2023-10-11 11:54:08,264][models.train_models][INFO] - train_loss=0.000, valid_loss=0.000, test_loss=0.000

L1Loss
[2023-10-11 12:00:22,759][models.train_models][INFO] - Model has 625 parameters (625 trainable).
[2023-10-11 12:00:22,759][models.train_models][INFO] - time=115.24 epoch=0: train_loss=0.052, valid_loss=0.002
[2023-10-11 12:01:58,015][models.train_models][INFO] - time=210.50 epoch=1: train_loss=0.002, valid_loss=0.002
[2023-10-11 12:03:34,218][models.train_models][INFO] - time=306.70 epoch=2: train_loss=0.002, valid_loss=0.003
[2023-10-11 12:05:09,405][models.train_models][INFO] - time=401.89 epoch=3: train_loss=0.001, valid_loss=0.000
[2023-10-11 12:06:44,922][models.train_models][INFO] - time=497.40 epoch=4: train_loss=0.002, valid_loss=0.000
[2023-10-11 12:06:44,926][models.train_models][INFO] - Reverting to model with best val loss
[2023-10-11 12:07:41,579][models.train_models][INFO] - train_loss=0.000, valid_loss=0.000, test_loss=0.000
