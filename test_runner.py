import unittest
from runner import FedRunner
from server import Server
from client import Client
from utils import Args
import random

class TestFedRunner(unittest.TestCase):
    def setUp(self):
        # Set up the Args with small test parameters for quick testing
        self.args = Args(
            dataset='mnist',
            num_clients=5,
            num_comm_rounds=2,
            clients_each_round=0.4,
            distribution='uniform',
            model='mnist2nn',
            train_epochs=1,
            batch_size=10,
            optimizer='sgd',
            lr=0.01,
            loss_fn='cross_entropy'
        )

        # Initialize FedRunner with the test args
        self.fed_runner = FedRunner(self.args)

    def test_initialization(self):
        # Test that the server and clients are initialized correctly
        self.assertIsInstance(self.fed_runner.server, Server)
        self.assertEqual(len(self.fed_runner.client_list), self.args.num_clients)

    def test_client_sampling(self):
        # Run one round to check that clients are sampled correctly
        num_sampled_clients = max(1, int(self.args.clients_each_round * len(self.fed_runner.client_list)))
        sampled_indices = random.sample(range(len(self.fed_runner.client_list)), num_sampled_clients)

        # Ensure that the number of sampled clients matches the expected count
        self.assertEqual(len(sampled_indices), num_sampled_clients)
        self.assertTrue(all(0 <= idx < self.args.num_clients for idx in sampled_indices))

    def test_run_one_round(self):
        # Run the first communication round
        self.fed_runner.run()

        # Check if the global model was successfully updated and aggregated
        global_model = self.fed_runner.server.get_global_model()
        self.assertIsNotNone(global_model)
        print("Global model updated successfully after one round.")

    def test_evaluation(self):
        # Test the evaluation of the global model on all clients
        self.fed_runner.evaluate_on_all_clents()

        # Check that each client has the global model set and evaluated
        for client in self.fed_runner.client_list:
            local_model = client.get_local_model()
            self.assertIsNotNone(local_model)
            print(f"Client {client.id} evaluated successfully.")

if __name__ == '__main__':
    unittest.main()