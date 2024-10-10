import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
from torchvision import datasets, transforms
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
    
def get_dataset(args):
    """
    Loads the specified dataset based on the arguments and returns the full training and test datasets.

    Args:
        args: A configuration object with attributes specifying the dataset type and other parameters.

    Returns:
        full_train_dataset: The entire training dataset.
        full_test_dataset: The entire test dataset.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalize dataset to range [-1, 1]
    ])

    if args.dataset.lower() == 'mnist':
        full_train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
        full_test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    elif args.dataset.lower() == 'cifar10':
        transform_cifar = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize CIFAR-10 images to range [-1, 1]
        ])
        full_train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_cifar)
        full_test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_cifar)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}. Please use 'mnist' or 'cifar10'.")

    return full_train_dataset, full_test_dataset

def distribute_data_uniformly(dataset, args):
    """
    Distributes the dataset uniformly among the specified number of clients.

    This function randomly shuffles the dataset and then divides it into approximately equal subsets
    for each client, ensuring that any remainder is evenly distributed among the first few clients.

    Args:
        dataset: The dataset to be distributed, typically a PyTorch dataset object.
        args: An object containing the configuration, specifically the attribute 'num_clients' 
              which indicates the number of clients among which the dataset should be distributed.

    Returns:
        client_datasets: A list of datasets (subsets), each corresponding to a client's share of the data.
                         The length of the list equals the number of clients.
    """
    # Shuffle the dataset indices
    dataset_size = len(dataset)
    indices = torch.randperm(dataset_size).tolist()

    # Calculate the size of each subset for clients
    split_size = dataset_size // args.num_clients
    remainder = dataset_size % args.num_clients  # Handle cases where dataset size isn't perfectly divisible

    # Distribute the data among clients
    client_datasets = []
    for i in range(args.num_clients):
        # Calculate the start and end indices for each client's subset
        start_index = i * split_size
        end_index = start_index + split_size
        if i < remainder:  # Distribute the remaining samples among the first clients
            end_index += 1

        client_indices = indices[start_index:end_index]
        client_subset = Subset(dataset, client_indices)
        client_datasets.append(client_subset)

    return client_datasets

def distribute_data_non_iid(dataset, args):
    """
    Distributes the dataset in a Non-IID manner among the specified number of clients.

    This function sorts the dataset by labels, divides it into shards, and then assigns 
    a specified number of shards to each client in a way that ensures a Non-IID distribution.

    Args:
        dataset: The dataset to be distributed, typically a PyTorch dataset object.
        args: An object containing the configuration, specifically the attributes 'num_clients',
              'num_shards', and 'shards_per_client'.

    Returns:
        client_datasets: A list of datasets (subsets), one for each client.
                         The length of the list equals the number of clients.
    """
    # Step 1: Sort the dataset by label
    targets = torch.tensor(dataset.targets)
    sorted_indices = targets.argsort().tolist()

    # Step 2: Divide the sorted data into shards
    shard_size = len(dataset) // args.num_shards
    shards = [sorted_indices[i * shard_size:(i + 1) * shard_size] for i in range(args.num_shards)]

    # Step 3: Randomly assign shards to clients
    client_datasets = [[] for _ in range(args.num_clients)]
    shard_indices = torch.randperm(args.num_shards).tolist()  # Shuffle the shards randomly

    for i in range(args.num_clients):
        assigned_shards = shard_indices[i * args.shards_per_client:(i + 1) * args.shards_per_client]
        for shard_idx in assigned_shards:
            client_datasets[i].extend(shards[shard_idx])

    # Convert client indices to Subsets of the original dataset
    client_subsets = [Subset(dataset, indices) for indices in client_datasets]

    return client_subsets

def distribute_train_test_data_uniformly(full_train_dataset, full_test_dataset, args):
    """
    Uniformly distributes the training and test datasets among the specified number of clients.

    This function takes the complete training and test datasets and splits them into smaller subsets,
    distributing these subsets uniformly across the clients specified in the 'args' configuration.

    Args:
        full_train_dataset: The complete training dataset to be distributed.
        full_test_dataset: The complete test dataset to be distributed.
        args: An object containing the configuration, specifically the attribute 'num_clients' 
              which indicates the number of clients among which the datasets should be distributed.

    Returns:
        client_train_datasets: A list of training datasets (subsets), one for each client.
        client_test_datasets: A list of test datasets (subsets), one for each client.
    """
    # Distribute the train and test data uniformly among the clients
    client_train_datasets = distribute_data_uniformly(full_train_dataset, args)
    client_test_datasets = distribute_data_uniformly(full_test_dataset, args)

    return client_train_datasets, client_test_datasets

def distribute_train_test_data_non_iid(full_train_dataset, full_test_dataset, args):
    """
    Uniformly distributes the training and test datasets in a Non-IID manner among the specified number of clients.

    This function takes the complete training and test datasets and splits them into smaller shards,
    distributing these shards uniformly across the clients specified in the 'args' configuration.

    Args:
        full_train_dataset: The complete training dataset to be distributed.
        full_test_dataset: The complete test dataset to be distributed.
        args: An object containing the configuration, specifically the attributes 'num_clients',
              'num_shards', and 'shards_per_client'.

    Returns:
        client_train_datasets: A list of training datasets (subsets), one for each client.
        client_test_datasets: A list of test datasets (subsets), one for each client.
    """
    # Distribute the train and test data in a Non-IID manner among the clients
    client_train_datasets = distribute_data_non_iid(full_train_dataset, args)
    client_test_datasets = distribute_data_non_iid(full_test_dataset, args)

    return client_train_datasets, client_test_datasets


# Args class to hold model parameters and configuration options
# Args class to hold model parameters and configuration options
class Args:
    def __init__(self, 
                 model='SimpleCNN', 
                 dataset='mnist', 
                 num_clients=5, 
                 num_comm_rounds=2, 
                 clients_each_round=0.4, 
                 distribution='uniform', 
                 num_shards=10,  # Number of shards for Non-IID distribution
                 shards_per_client=2,  # Number of shards assigned to each client
                 train_epochs=1, 
                 batch_size=32, 
                 optimizer='adam', 
                 lr=0.001, 
                 loss_fn='cross_entropy'):
        self.model = model  # Model type (e.g., 'SimpleCNN', 'mnist2nn')
        self.dataset = dataset  # Dataset type (e.g., 'mnist', 'cifar10')
        self.num_clients = num_clients  # Total number of clients in federated learning
        self.num_comm_rounds = num_comm_rounds  # Number of communication rounds
        self.clients_each_round = clients_each_round  # Fraction of clients to sample in each round
        self.distribution = distribution  # Data distribution type ('uniform' or 'non-iid')
        self.num_shards = num_shards  # Total number of shards for Non-IID distribution
        self.shards_per_client = shards_per_client  # Number of shards assigned to each client
        self.train_epochs = train_epochs  # Number of training epochs for each client
        self.batch_size = batch_size  # Batch size for client training
        self.optimizer = optimizer  # Optimizer type ('adam', 'sgd', etc.)
        self.lr = lr  # Learning rate for the optimizer
        self.loss_fn = loss_fn  # Loss function type (e.g., 'cross_entropy', 'mse')