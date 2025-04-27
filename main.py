import copy
import yaml
import ujson
import argparse
from component.client import Client
from utils import preprocess
import torch
from component.trainer import FedAvg, FedAdam, FedYogi, AdaFedAdam, FedAvgM
import matplotlib.pyplot as plt
import numpy as np


def main(config):
    # Check if cuda cores are available and run on cuda if available else run on cpu
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))
    device = "cuda:0" if torch.cuda.is_available() else 'cpu'

    data_path = f"../kex/data/{config['dataset']}/partitioned.json"
    # Prepare the data
    with open(data_path, 'r') as inf:
        data = ujson.load(inf)
    data = data['user_data']
    data = {idx: data[idx] for idx in list(data.keys())[:config['num_clients']]}

    # Build dataset
    preprocessor = getattr(preprocess, config['dataset'])
    clients = [
        Client(
            k, config['dataset'], preprocessor(v),
            config['batch_size'], device) for k, v in data.items()]

    # Build the trainer
    trainer_dict = {
        "fedavg": FedAvg,
        "fedadam": FedAdam,
        "fedyogi": FedYogi,
        "adafedadam": AdaFedAdam,
        "fedavgm": FedAvgM
    }
    trainer = trainer_dict[config['trainer']](config['dataset'])

    # Initialize evaluation list
    lst_acc = []
    lst_loss = []
    eval_rounds = []
    lst_std_acc = []
    lst_std_loss = []
    lst_worst_acc = []

    first_round = True
    # Global training loop
    for round in range(config['num_rounds']):

        # Evaluate the model
        if round % config['eval_every'] == 0:
            eval_result = trainer.eval(clients)
            # print(f"Evaluation result: {eval_result}")
            print(
                f"Round {round+1}/{config['num_rounds']} - "
                f"Avg acc (%): {eval_result['avg_acc']:.4f}, "
                f"Avg error: {eval_result['avg_error']:.4f}, "
                f"Std acc (%): {eval_result['std_acc']:.4f}, "
                f"Std error: {eval_result['std_error']:.4f}, "
                f"Worst 30% avg acc (%): {eval_result['worst_acc']:.4f}")
            lst_acc.append(eval_result['avg_acc'])
            lst_loss.append(eval_result['avg_error'])
            eval_rounds.append(round + 1)
            lst_std_acc.append(eval_result['std_acc'])
            lst_std_loss.append(eval_result['std_error'])
            lst_worst_acc.append(eval_result['worst_acc'])

        # Train the model
        if config['trainer'] == "fedavgm":  #FedAvg-M
            if first_round:
                global_weights, d = trainer.train(clients, config['participation'], config['local_epochs'], first_round)
                first_round = False
            else:
                global_weights, d = trainer.train(clients, config['participation'], config['local_epochs'], first_round, global_weights, d)
        else:  # FedAvg, FedAdam, FedYogi, AdaFedAdam
            trainer.train(clients, config['participation'], config['local_epochs'])

    # Perform final update for FedAvg-M (doesn't affect results)
    if config['trainer'] == "fedavgm":
        for client in clients:
            client.set_weights(global_weights)
            loss, grad = client.get_grad()
            local_weights = client.get_weights()
            # Note: gamma value is default here
            updated_weights = {k: local_weights[k] - 1e-3 * grad[k] for k in local_weights}
            client.model.load_state_dict(updated_weights)
        # Final eval
        eval_result_fin = trainer.eval(clients)
        print(
            f"After final update - "
            f"Avg acc (%): {eval_result_fin['avg_acc']:.4f}, "
            f"Avg error: {eval_result_fin['avg_error']:.4f}, "
            f"Std acc (%): {eval_result_fin['std_acc']:.4f}, "
            f"Std error: {eval_result_fin['std_error']:.4f}")

    return lst_acc, lst_loss, eval_rounds, lst_std_acc, lst_std_loss, lst_worst_acc

def results(TRAINERS):
    """
    Runs all federated optimizers and plot their accuracy, loss and convergence
    """
    eval_acc = {}
    eval_loss = {}
    rounds = {}
    eval_std_acc = {}
    eval_std_loss = {}
    eval_worst_acc = {}
    for tr in TRAINERS:
        config_copy = copy.deepcopy(config)
        config_copy['trainer'] = tr
        (eval_acc[tr], eval_loss[tr], rounds[tr], eval_std_acc[tr],
         eval_std_loss[tr], eval_worst_acc[tr]) = main(config_copy)

    # Plot accuracy
    colors = {"fedavg": 'firebrick', "fedadam": 'chartreuse', "fedyogi": 'deepskyblue', "adafedadam": 'magenta',
              "fedavgm": 'mediumblue'}
    fed_names = {"fedavg": 'FedAvg', "fedadam": 'FedAdam', "fedyogi": 'FedYogi', "adafedadam": 'AdaFedAdam',
                 "fedavgm": 'FedAvg-M'}
    plt.rcParams["font.family"] = "Times New Roman"
    for tr in TRAINERS:
        plt.plot(rounds[tr], eval_acc[tr], color=colors[tr], label=fed_names[tr])
    plt.xlabel('Number of rounds')
    plt.ylabel('Accuracy / %')
    if config['dataset'] == "fashionmnist":
        plt.title(f"Fashion-MNIST | Non-IID level: {config['non_iid_level']}")
    else:
        plt.title(f"{config['dataset'].upper()} | Non-IID level: {config['non_iid_level']}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Plot loss
    for tr in TRAINERS:
        plt.plot(rounds[tr], eval_loss[tr], color=colors[tr], label=fed_names[tr])
    plt.xlabel('Number of rounds')
    plt.ylabel('Loss')
    if config['dataset'] == "fashionmnist":
        plt.title(f"Fashion-MNIST | Non-IID level: {config['non_iid_level']}")
    else:
        plt.title(f"{config['dataset'].upper()} | Non-IID level: {config['non_iid_level']}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Plot worst 30%
    if eval_worst_acc != eval_acc:
        for tr in TRAINERS:
            plt.plot(rounds[tr], eval_worst_acc[tr], color=colors[tr], label=fed_names[tr])
        plt.xlabel('Number of rounds')
        plt.ylabel('Accuracy / %')
        if config['dataset'] == "fashionmnist":
            plt.title(f"Fashion-MNIST | Worst 30% | Non-IID level: {config['non_iid_level']}")
        else:
            plt.title(f"{config['dataset'].upper()} | Worst 30% | Non-IID level: {config['non_iid_level']}")
        plt.legend()
        plt.grid()
        plt.tight_layout()
        plt.show()

    # Plot standard deviation of accuracy
    for tr in TRAINERS:
        plt.plot(rounds[tr], eval_std_acc[tr], color=colors[tr], label=fed_names[tr])
    plt.xlabel('Number of rounds')
    plt.ylabel('Accuracy / %')
    if config['dataset'] == "fashionmnist":
        plt.title(f"Fashion-MNIST | Standard Deviation | Non-IID level: {config['non_iid_level']}")
    else:
        plt.title(f"{config['dataset'].upper()} | Standard Deviation | Non-IID level: {config['non_iid_level']}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Plot standard deviation of loss
    for tr in TRAINERS:
        plt.plot(rounds[tr], eval_std_loss[tr], color=colors[tr], label=fed_names[tr])
    plt.xlabel('Number of rounds')
    plt.ylabel('Loss')
    if config['dataset'] == "fashionmnist":
        plt.title(f"Fashion-MNIST | Standard Deviation | Non-IID level: {config['non_iid_level']}")
    else:
        plt.title(f"{config['dataset'].upper()} | Standard Deviation | Non-IID level: {config['non_iid_level']}")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Print final STD for each model
    print("Averaged last five evaluation values of each optimizer:")
    for tr in TRAINERS:
        # Accuracy
        last_acc = eval_acc[tr][-5:]
        mean_acc = np.mean(last_acc)
        std_acc = np.std(last_acc, ddof=1)
        error_acc = std_acc / np.sqrt(5)

        # Standard deviation
        last_std = eval_std_acc[tr][-5:]
        mean_std = np.mean(last_std)
        std_std = np.std(last_std, ddof=1)
        error_std = std_std / np.sqrt(5)

        # Worst 30%
        last_worst = eval_worst_acc[tr][-5:]
        mean_worst = np.mean(last_worst)
        std_worst = np.std(last_worst, ddof=1)
        error_worst = std_worst / np.sqrt(5)

        # Print final results
        print(
            f"Optim: {fed_names[tr]} - "
            f"Acc. (%): {mean_acc} +/- {error_acc} | "
            f"Acc. STD (%): {mean_std} +/- {error_std} | "
            f"Acc worst (%): {mean_worst} +/- {error_worst}"
        )

if __name__ == "__main__":
    # Get arguments from config file
    with open("config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Get trainers from terminal
    TRAINERS = ["fedavg", "fedadam", "fedyogi", "adafedadam", "fedavgm"]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trainer", type=str, default=None,
        nargs="?",
        choices=TRAINERS,
        help="The trainer to use")

    args = parser.parse_args()
    if args.trainer is None:
        results(TRAINERS)
    elif args.trainer in TRAINERS:
        config['trainer'] = args.trainer
        _, *_ = main(config)
    else:
        raise RuntimeError('No optimizer with that name exist in module files')
