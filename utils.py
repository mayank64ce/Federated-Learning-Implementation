import torch
from torch.utils.data import DataLoader, TensorDataset
from models import model_factory

def get_loader(dataset, batch_size, train=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=train)

def get_optimizer(optimizer_name, model, lr):
    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr)
    else:
        raise ValueError("Invalid optimizer name")

def get_loss_fn(loss_fn):
    if loss_fn == "cross_entropy":
        return torch.nn.CrossEntropyLoss()
    elif loss_fn == "mse":
        return torch.nn.MSELoss()
    else:
        raise ValueError("Invalid loss function name")

# Function to create a dummy dataset for testing
def get_dummy_dataset(num_samples=100, num_features=28*28, num_classes=10):
    # Create random input data resembling MNIST (28x28 pixels, flattened to 784 features)
    x = torch.randn(num_samples, 1, 28, 28)  # Shape (num_samples, channels, height, width)
    # Create random labels as integers between 0 and num_classes-1
    y = torch.randint(0, num_classes, (num_samples,))
    # Return the dataset wrapped in a TensorDataset for easy loading
    train_dataset = TensorDataset(x, y)
    test_dataset = TensorDataset(x, y)
    return train_dataset, test_dataset

def get_model(args):
    if args.model in model_factory:
        return model_factory[args.model]()
    else:
        raise ValueError("Invalid model name")

# Args class to hold model parameters and configuration options
class Args:
    def __init__(self, model='SimpleCNN', train_epochs=1, batch_size=32, optimizer='adam', lr=0.001, loss_fn='cross_entropy'):
        self.model = model
        self.train_epochs = train_epochs
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.lr = lr
        self.loss_fn = loss_fn