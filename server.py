import torch
from collections import OrderedDict
from models import model_factory

class Server:
    def __init__(self, args):
        self.model = model_factory[args.model]()

    def get_global_model(self):
        return self.model.state_dict()

    def aggregate(self, model_weights, scaling_factors):
        # Initialize an OrderedDict to store the aggregated weights
        averaged_weights = OrderedDict()

        # Iterate over each key in the model weights (assuming all models have the same keys)
        for k in model_weights[0].keys():
            # Start with a zero tensor for each parameter key
            averaged_weights[k] = torch.zeros_like(model_weights[0][k])

            # Perform the weighted aggregation for each model's parameter
            for i in range(len(model_weights)):
                local_weight = model_weights[i]
                averaged_weights[k] += local_weight[k] * scaling_factors[i]

        return averaged_weights

    def set_global_model(self, new_state_dict):
        self.model.load_state_dict(new_state_dict)
