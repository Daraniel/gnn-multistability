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
