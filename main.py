import copy
import yaml
import ujson
import argparse
from component.client import Client
from utils import preprocess
import torch
from component.trainer import FedAvg, FedAdam, FedYogi, AdaFedAdam, FedAvgM
import matplotlib.pyplot as plt


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

    first_round = True
    # Global training loop
    for round in range(config['num_rounds']):

        # Evaluate the model
        if round % config['eval_every'] == 0:
            eval_result = trainer.eval(clients)
            # print(f"Evaluation result: {eval_result}")
            print(
                f"Round {round+1}/{config['num_rounds']} - "
                f"Avg acc: {eval_result['avg_acc']:.4f}, "
                f"Avg error: {eval_result['avg_error']:.4f}, "
                f"Std acc: {eval_result['std_acc']:.4f}, "
                f"Std error: {eval_result['std_error']:.4f}")
            lst_acc.append(eval_result['avg_acc'])
            lst_loss.append(eval_result['avg_error'])
            eval_rounds.append(round + 1)
            lst_std_acc.append(eval_result['std_acc'])
            lst_std_loss.append(eval_result['std_error'])

        # Train the model
        if config['trainer'] == "fedavgm":  #FedAvg-M
            if first_round:
                global_weights, d = trainer.train(clients, config['participation'], config['local_epochs'], first_round)
                first_round = False
            else:
                global_weights, d = trainer.train(clients, config['participation'], config['local_epochs'], first_round, global_weights, d)
        else:  # FedAvg, FedAdam, FedYogi, AdaFedAdam
            trainer.train(clients, config['participation'], config['local_epochs'])

    return lst_acc, lst_loss, eval_rounds, lst_std_acc, lst_std_loss

def results(TRAINERS):
    """
    Runs all federated optimizers and plot their accuracy, loss and convergence
    """
    eval_acc = {}
    eval_loss = {}
    rounds = {}
    eval_std_acc = {}
    eval_std_loss = {}
    for tr in TRAINERS:
        config_copy = copy.deepcopy(config)
        config_copy['trainer'] = tr
        eval_acc[tr], eval_loss[tr], rounds[tr], eval_std_acc[tr], eval_std_loss[tr] = main(config_copy)

    # Plot accuracy
    colors = {"fedavg": 'firebrick', "fedadam": 'chartreuse', "fedyogi": 'deepskyblue', "adafedadam": 'magenta',
              "fedavgm": 'mediumblue'}
    for tr in TRAINERS:
        """
        for i in range(len(rounds[tr])):
            plt.plot(rounds[tr][i], eval_acc[tr][i], colors[tr], label=tr)
            plt.fill_between(rounds[tr][i], eval_acc[tr][i] - eval_std_acc[tr][i],
                     eval_acc[tr][i] + eval_std_acc[tr][i], alpha=0.3)
            # plt.fill_between(rounds[tr], )
        """
        plt.plot(rounds[tr], eval_acc[tr], colors[tr], label=tr)
        eval_neg = []
        eval_pos = []
        for i in range(len(rounds[tr])):
            eval_neg.append(eval_acc[tr][i] - eval_std_acc[tr][i])
            eval_pos.append(eval_acc[tr][i] + eval_std_acc[tr][i])
        plt.fill_between(rounds[tr], eval_neg, eval_pos, alpha=0.3, color=colors[tr])
    plt.xlabel('Number of rounds')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

    # Plot loss
    for tr in TRAINERS:
        plt.plot(rounds[tr], eval_loss[tr], colors[tr], label=tr)
        eval_neg = []
        eval_pos = []
        for i in range(len(rounds[tr])):
            eval_neg.append(eval_loss[tr][i] - eval_std_loss[tr][i])
            eval_pos.append(eval_loss[tr][i] + eval_std_loss[tr][i])
        plt.fill_between(rounds[tr], eval_neg, eval_pos, alpha=0.3, color=colors[tr])
    plt.xlabel('Number of rounds')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    plt.show()

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
