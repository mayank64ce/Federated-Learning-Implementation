from client import Client
from server import Server
from utils import get_dataset, distribute_train_test_data_uniformly, distribute_train_test_data_non_iid
import random
import copy

class FedRunner:
    def __init__(self, args):
         # Initialize the server
            # Initialize the global_model in server
        self.server = Server(args) 
        self.args = args

        # make datasets

        full_train_dataset, full_test_dataset = get_dataset(args)

        if args.distribution == 'uniform':
            client_train_datasets, client_test_datasets = distribute_train_test_data_uniformly(full_train_dataset, full_test_dataset, args)
        elif args.distribution == 'non-iid':
            client_train_datasets, client_test_datasets = distribute_train_test_data_non_iid(full_train_dataset, full_test_dataset, args)
        else:
            raise ValueError("Invalid data distribution. Please choose between 'uniform' and 'non-iid'.")
       
        # Note to self: to initialize each client we need train_data, test_data and args
        # Initialize the clients
        self.client_list = []
        for i in range(args.num_clients):
            client = Client(client_train_datasets[i], client_test_datasets[i], args, id=i)
            self.client_list.append(client)


    def run(self):
        for round_num in range(1, self.args.num_comm_rounds + 1):
            print(f"Starting Round {round_num}/{self.args.num_comm_rounds}")

            model_weights = []
            scaling_factors = []

            # Step 1: Sample a client_each_round of clients for this round
            num_sampled_clients = max(1, int(self.args.clients_each_round * len(self.client_list)))
            sampled_indices = random.sample(range(len(self.client_list)), num_sampled_clients)

            print(f"Selected clients for round {round_num}: {sampled_indices}")

            client_list = [self.client_list[i] for i in sampled_indices]
            total_train_samples = sum([client.get_train_data_length() for client in client_list])

            # Step 2: Server gets the global model
            global_model = self.server.get_global_model()

            # Step 3: Each client sets the global model
            for client in client_list:
                client.set_local_model(copy.deepcopy(global_model))
                client.train()
                local_model = client.get_local_model()
                model_weights.append(copy.deepcopy(local_model))
                scaling_factors.append(client.get_train_data_length() / total_train_samples)
            
            # Step 4: Server aggregates the models
            aggregated_model = self.server.aggregate(model_weights, scaling_factors)

            print("Successfully aggregated the models!")

            # Step 5: Server updates the global model
            self.server.set_global_model(aggregated_model)

            # Step 6: Evaluate the global model on all clients
            
            print("Evaluating the global model on all clients...")

            self.evaluate_on_all_clents()

    def evaluate_on_all_clents(self):
        for idx, client in enumerate(self.client_list):
            print(f"Client {idx} is evaluating the model...")
            client.set_local_model(self.server.get_global_model())
            client.evaluate()

