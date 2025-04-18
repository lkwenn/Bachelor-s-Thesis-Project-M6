# Optimisation Algorithms for Federated Learning

This repository contains the code for the publication: [Optimisation Algorithms for Federated Learning](url)
 
## Table of Contents
- [Installation](#installation)
- [Dataset Generation](#dataset-generation)
- [Training](#training)
- [Configuration](#configuration)
- [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/lkwenn/kex.git
   cd kex
   ```

2. Install the dependencies using `pip` and the provided [`requirements.txt`](requirements.txt):

   ```bash
   pip install -r requirements.txt
   ```
## Dataset Generation

To prepare and generate the dataset required for training, run:

```bash
python -m data_prepare.generate
```

Both the dataset generating script and training script will read configurations (e.g., data paths, parameters, etc.) from the [`config.yml`](config.yml) file.

## Training

Once the dataset is prepared, you can run a specified optimiser by running:
```bash
python main.py nameOfOptimiser
```
Available optimisers to run are: fedavg, fedadam, fedyogi, adafedadam, fedavgm. To run all optimisers and produce graphs run:
```bash
python main.py
```
The training script will load configurations from the [`config.yml`](config.yml) file. To modify any experiment settings (hyperparameters, data parameters, model configurations, etc.), please update [`config.yml`](config.yml) accordingly.

## Configuration

All configurations are done in the [`config.yml`](config.yml) file. The settings you can adjust are:
- Dataset preparation settings (type of dataset, non-IID level, seed)
- System settings (number of clients, batch size, local epochs, etc.)

## Acknowledgements
The authors of the paper would like to thank the following open-source project for their framework:
- [AdaFedAdam](https://github.com/li-ju666/adafedadam/tree/main)

## License
This project is licensed under ...