import copy
import yaml
import ujson
import argparse
from kex.kex.component.client import Client
from kex.kex.utils import preprocess
import torch
from kex.kex.component.trainer import FedAvg, FedAdam, FedYogi, AdaFedAdam, FedMGDAM
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
        "fedmgdam": FedMGDAM
    }
    trainer = trainer_dict[config['trainer']](config['dataset'])

    # Initialize evaluation list
    lst_acc = []
    lst_loss = []
    eval_rounds = []

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
                f"Avg error: {eval_result['avg_error']:.4f}, ")
                # f"Std acc: {eval_result['std_acc']:.4f}, "
                # f"Std error: {eval_result['std_error']:.4f}")
            lst_acc.append(eval_result['avg_acc'])
            lst_loss.append(eval_result['avg_error'])
            eval_rounds.append(round + 1)

        # Train the model
        if config['trainer'] == "fedmgdam":  #FedMGDAM
            if first_round:
                global_weights, d = trainer.train(clients, config['participation'], config['local_epochs'], first_round)
                first_round = False
            else:
                global_weights, d = trainer.train(clients, config['participation'], config['local_epochs'], first_round, global_weights, d)
        else:  # FedAvg, FedAdam, FedYogi, AdaFedAdam
            trainer.train(clients, config['participation'], config['local_epochs'])

    return lst_acc, lst_loss, eval_rounds

def results():
    """
    Runs all federated optimizers and plot their accuracy, loss and convergence
    """
    eval_acc = {}
    eval_loss = {}
    rounds = {}
    for tr in TRAINERS:
        config_copy = copy.deepcopy(config)
        config_copy['trainer'] = tr
        eval_acc[tr], eval_loss[tr], rounds[tr] = main(config_copy)

    # Plot accuracy
    markers = {"fedavg": 'o', "fedadam": 'v', "fedyogi": '^', "adafedadam": 's', "fedmgdam": 'd'}
    for tr in TRAINERS:
        plt.plot(rounds[tr], eval_acc[tr], markers[tr], label=tr)
    plt.xlabel('Number of rounds')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Plot loss
    for tr in TRAINERS:
        plt.plot(rounds[tr], eval_loss[tr], markers[tr], label=tr)
    plt.xlabel('Number of rounds')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Get arguments from config file
    with open("config.yml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    # Get trainers from terminal
    TRAINERS = ["fedavg", "fedadam", "fedyogi", "adafedadam", "fedmgdam"]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "trainer", type=str, default="fedavg",
        nargs="?",
        choices=TRAINERS,
        help="The trainer to use")

    args = parser.parse_args()
    if args.trainer is None:
        results()
    elif args.trainer in TRAINERS:
        config['trainer'] = args.trainer
        _, _, _ = main(config)
    else:
        raise RuntimeError('No optimizer with that name exist in module files')
