import argparse
from runner import FedRunner


def get_args():
    parser = argparse.ArgumentParser(description='Federated Learning Argument Parser')

    # Adding arguments to the parser
    parser.add_argument('--model', type=str, default='mnist2nn', help='Model type (e.g., SimpleCNN, mnist2nn)')
    parser.add_argument('--dataset', type=str, default='mnist', help='Dataset type (e.g., mnist, cifar10)')
    parser.add_argument('--num_clients', type=int, default=5, help='Total number of clients in federated learning')
    parser.add_argument('--num_comm_rounds', type=int, default=2, help='Number of communication rounds')
    parser.add_argument('--clients_each_round', type=float, default=0.1, help='Fraction of clients to sample in each round')
    parser.add_argument('--distribution', type=str, default='uniform', help='Data distribution type (uniform or non-iid)')
    parser.add_argument('--num_shards', type=int, default=10, help='Total number of shards for Non-IID distribution')
    parser.add_argument('--shards_per_client', type=int, default=2, help='Number of shards assigned to each client')
    parser.add_argument('--train_epochs', type=int, default=1, help='Number of training epochs for each client')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for client training')
    parser.add_argument('--optimizer', type=str, default='adam', help='Optimizer type (adam, sgd, etc.)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--loss_fn', type=str, default='cross_entropy', help='Loss function type (cross_entropy, mse)')

    # Parse the arguments from the command line
    args = parser.parse_args()

    return args

def main(args):
    """
    Important things that control the computation:
    1. clients_each_round (C in the paper)
    2. train_epochs (E in the paper)
    3. batch_size (B in the paper)
    4. num_comm_rounds
    5. num_clients
    6. distribution
    7. model
    """
    print("==== Experiment Configuration ====")
    print(f"Client Fraction: {args.clients_each_round}")
    print(f"Local Training Epochs: {args.train_epochs}")
    print(f"Batch Size: {args.batch_size}")
    print(f"Number of Communication Rounds: {args.num_comm_rounds}")
    print(f"Number of Clients: {args.num_clients}")
    print(f"Data Distribution: {args.distribution}")
    print("=================================")

    runner = FedRunner(args)
    runner.run()

if __name__ == '__main__':
    args = get_args()
    main(args)