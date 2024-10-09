import unittest
import torch
from client import Client
from models import model_factory
from utils import get_dummy_dataset, Args

class TestClient(unittest.TestCase):
    def setUp(self):
        # Set up dummy arguments and datasets for testing
        self.args = Args(
            model='mnistcnn',
            train_epochs=1,
            batch_size=2,
            optimizer='sgd',
            lr=0.01,
            loss_fn='cross_entropy'
        )
        self.train_dataset, self.test_dataset = get_dummy_dataset()

        # Initialize a Client object
        self.client = Client(self.train_dataset, self.test_dataset, self.args)

    def test_client_initialization(self):
        # Check if model is initialized correctly
        self.assertIsInstance(self.client.model, torch.nn.Module)
        
        # Check if optimizer and loss function are initialized correctly
        self.assertIsNotNone(self.client.optimizer)
        self.assertIsNotNone(self.client.loss_fn)

    def test_set_local_model(self):
        # Create a dummy state dict and set it as the local model state
        dummy_state_dict = model_factory[self.args.model]().state_dict()
        self.client.set_local_model(dummy_state_dict)

        # Verify the model state matches the dummy state
        for key in dummy_state_dict:
            self.assertTrue(torch.equal(self.client.model.state_dict()[key], dummy_state_dict[key]))

    def test_train(self):
        initial_params = {name: param.clone() for name, param in self.client.model.named_parameters()}
        self.client.train()
        updated_params = {name: param for name, param in self.client.model.named_parameters()}

        # Verify that training updated at least one parameter in the model
        parameter_changed = any((initial_params[name] != updated_params[name]).any() for name in initial_params)
        self.assertTrue(parameter_changed)

    def test_evaluate(self):
        # Make sure the evaluate method runs without errors
        try:
            self.client.evaluate()
            success = True
        except Exception as e:
            print(f"Evaluate failed with error: {e}")
            success = False
        self.assertTrue(success)

    def test_get_local_model(self):
        # Ensure that get_local_model() returns a state dict
        state_dict = self.client.get_local_model()
        self.assertIsInstance(state_dict, dict)
        self.assertTrue(all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in state_dict.items()))

if __name__ == '__main__':
    unittest.main()