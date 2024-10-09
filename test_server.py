import unittest
import torch
from server import Server
from models import model_factory
from utils import Args

class TestServer(unittest.TestCase):
    def setUp(self):
        # Set up dummy arguments for testing
        self.args = Args(
            model='mnist2nn'
        )
        # Initialize a Server object
        self.server = Server(self.args)

    def test_server_initialization(self):
        # Check if the server model is initialized correctly
        self.assertIsInstance(self.server.model, torch.nn.Module)

    def test_get_global_model(self):
        # Get the global model's state dict and verify it's a dictionary
        global_model_state = self.server.get_global_model()
        self.assertIsInstance(global_model_state, dict)
        self.assertTrue(all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in global_model_state.items()))

    def test_set_global_model(self):
        # Create a new state dict and set it as the global model state
        dummy_state_dict = model_factory[self.args.model]().state_dict()
        self.server.set_global_model(dummy_state_dict)

        # Verify that the server's model state was updated correctly
        server_model_state = self.server.get_global_model()
        for key in dummy_state_dict:
            self.assertTrue(torch.equal(server_model_state[key], dummy_state_dict[key]))

    def test_aggregate(self):
        # Create two dummy state dictionaries to simulate local model weights
        model_weights_1 = model_factory[self.args.model]().state_dict()
        model_weights_2 = model_factory[self.args.model]().state_dict()

        # Modify the second state dictionary slightly to ensure aggregation changes
        for key in model_weights_2:
            model_weights_2[key] += 1.0

        # List of model weights and scaling factors
        model_weights = [model_weights_1, model_weights_2]
        scaling_factors = [0.6, 0.4]  # Weights sum up to 1.0 for proper averaging

        # Aggregate the model weights using the server's method
        aggregated_weights = self.server.aggregate(model_weights, scaling_factors)

        # Verify that the aggregation works as expected
        for key in model_weights_1:
            expected_value = model_weights_1[key] * scaling_factors[0] + model_weights_2[key] * scaling_factors[1]
            self.assertTrue(torch.allclose(aggregated_weights[key], expected_value))

if __name__ == '__main__':
    unittest.main()