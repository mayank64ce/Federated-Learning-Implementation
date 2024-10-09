from client import Client
from server import Server

class FedRunner:
    def __init__(self):
        # Initialize the server
            # Initialize the global_model in server
        # Initialize the clients
            # distribute the data to the clients
        pass

    def run(self):
        # for each round
            # sample clients
            # get the global model from server
            # for each client
                # set the local model in client
                # train the local model in client
                # get the local model from client
            # aggregate the local models in server
            # set the global model in server
            # evaluate the global model in server
        pass
