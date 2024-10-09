import torch
import torch.nn as nn
import torch.nn.functional as F

class MNIST2NN(nn.Module):
    def __init__(self):
        super(MNIST2NN, self).__init__()
        # Input layer to first hidden layer (784 input features for 28x28 images)
        self.fc1 = nn.Linear(784, 200)
        # First hidden layer to second hidden layer
        self.fc2 = nn.Linear(200, 200)
        # Second hidden layer to output layer (10 classes for MNIST digits 0-9)
        self.fc3 = nn.Linear(200, 10)

    def forward(self, x):
        # Flatten the input tensor to (batch_size, 784)
        x = x.view(-1, 784)
        # First hidden layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Second hidden layer with ReLU activation
        x = F.relu(self.fc2(x))
        # Output layer (no activation function as it will be combined with softmax or cross-entropy loss later)
        x = self.fc3(x)
        return x

class MNISTCNN(nn.Module):
    def __init__(self):
        super(MNISTCNN, self).__init__()
        # First convolutional layer: input channels = 1 (for grayscale images), output channels = 32, kernel size = 5x5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
        # Second convolutional layer: input channels = 32, output channels = 64, kernel size = 5x5
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5)
        # Fully connected layer: input features will depend on the output size of the conv layers after pooling
        self.fc1 = nn.Linear(64 * 4 * 4, 512)  # Adjusted input size based on output dimensions after pooling
        # Output layer: 10 classes for MNIST digits
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # First convolutional layer with ReLU activation and 2x2 max pooling
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # Second convolutional layer with ReLU activation and 2x2 max pooling
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        # Flatten the tensor for the fully connected layer
        x = x.view(-1, 64 * 4 * 4)
        # Fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Output layer with softmax activation (optional if using CrossEntropyLoss)
        x = self.fc2(x)
        return x
    

model_factory = {
    "mnist2nn": MNIST2NN,
    "mnistcnn": MNISTCNN
}