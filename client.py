import torch
from utils import get_loader, get_optimizer, get_loss_fn, get_model

class Client:
    def __init__(self, train_dataset, test_dataset, args, id=None):
        # Initialize the client with the specified model and training parameters
        self.id = id
        self.model = get_model(args)
        self.train_epochs = args.train_epochs
        self.train_loader = get_loader(train_dataset, args.batch_size)
        self.test_loader = get_loader(test_dataset, args.batch_size, train = False)
        self.optimizer = get_optimizer(args.optimizer, self.model, args.lr)
        self.loss_fn = get_loss_fn(args.loss_fn)
        self.train_dataset_length = len(train_dataset)

    def set_local_model(self, global_model_state_dict):
        # Update the local model with the global model's state
        self.model.load_state_dict(global_model_state_dict)

    def train(self):
        model = self.model
        model.train()  # Set model to training mode
        optimizer = self.optimizer
        loss_fn = self.loss_fn

        for epoch in range(self.train_epochs):
            running_loss = 0.0  # Track the loss for logging
            for x, y in self.train_loader:
                optimizer.zero_grad()  # Clear gradients from the previous step
                out = model(x)  # Forward pass
                loss = loss_fn(out, y)  # Compute the loss

                loss.backward()  # Backward pass (compute gradients)
                optimizer.step()  # Update model parameters

                running_loss += loss.item()

            # Log training information for this epoch
            print(f"Client training - Epoch [{epoch+1}/{self.train_epochs}], Loss: {running_loss/len(self.train_loader):.4f}")

    def evaluate(self):
        model = self.model
        model.eval()  # Set model to evaluation mode
        loss_fn = self.loss_fn
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():  # Disable gradient computation for evaluation
            for x, y in self.test_loader:
                out = model(x)  # Forward pass
                loss = loss_fn(out, y)  # Compute the loss
                total_loss += loss.item()

                _, predicted = torch.max(out, 1)  # Get the class with the highest score
                correct += (predicted == y).sum().item()
                total += y.size(0)

        accuracy = 100 * correct / total
        avg_loss = total_loss / len(self.test_loader)
        print(f"Client evaluation - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")

    def get_local_model(self):
        # Return the current state of the local model
        return self.model.state_dict()
    
    def get_train_data_length(self):
        return self.train_dataset_length