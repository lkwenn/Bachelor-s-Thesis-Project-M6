# Optimisation Algorithms for Federated Learning

This repository contains the code for the publication: [Optimisation Algorithms for Federated Learning](url) (link will be available once published)
 
## Table of Contents
- [Installation](#installation)
- [Generation of datasets](#generation-of-datasets)
- [Training](#training)
- [Configuration](#configuration)
- [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/lkwenn/kex.git
   cd kex
   ```

2. Install the provided dependencies [`requirements.txt`](requirements.txt) using `pip`:

   ```bash
   pip install -r requirements.txt
   ```
## Generation of datasets

To prepare and generate the dataset required for training, run:

```bash
python -m data_prepare.generate
```

The dataset generating script [`generate.py`](data_prepare/generate.py) and the main training script [`main.py`](main.py) will read configurations from the [`config.yml`](config.yml) file. 

## Training

Once the dataset is prepared, you can run a specified optimiser by running:
```bash
python main.py nameOfOptimiser
```
Available optimisers to run are: fedavg, fedadam, fedyogi, adafedadam, fedavgm. To run all optimisers and produce graphs run:
```bash
python main.py
```
The training script will load configurations from the [`config.yml`](config.yml) file. To change any experiment settings - such as hyperparameters, data options, or model settings - update [`config.yml`](config.yml) accordingly. Certain hyperparameters like $\beta_1, \beta_2$ has to be changed manually. See full list of parameters that can be changed under [Configuration](#configuration).

## Configuration

Most configurations are done in the [`config.yml`](config.yml) file. The settings you can adjust are:
- Dataset preparation settings
  - Type of dataset (support for MNIST, CIFAR10, Fashion MNIST)
  - Non-IID level
  - Seed
- System settings
  - Number of clients
  - Batch size
  - Local epochs
  - Number of global rounds
  - Percentage of clients that participate every global round
- Results
  - Rounds to evaluate

Hyperparameters for the optimisers are configured manually in the [`trainer.py`](component/trainer.py) file. 

## Acknowledgements
The authors of the paper would like to thank the following open-source project for their framework:
- [AdaFedAdam](https://github.com/li-ju666/adafedadam/tree/main)

## License
This project is licensed under the MIT License.