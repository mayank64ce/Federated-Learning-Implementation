# Federated-Learning-Implementation

Implementing "Communication-Efficient Learning of Deep Networks from Decentralized Data by McMahan et. al."

## Instructions

```bash
options:
  -h, --help            show this help message and exit
  --model MODEL         Model type (e.g., SimpleCNN, mnist2nn)
  --dataset DATASET     Dataset type (e.g., mnist, cifar10)
  --num_clients NUM_CLIENTS
                        Total number of clients in federated learning
  --num_comm_rounds NUM_COMM_ROUNDS
                        Number of communication rounds
  --clients_each_round CLIENTS_EACH_ROUND
                        Fraction of clients to sample in each round
  --distribution DISTRIBUTION
                        Data distribution type (uniform or non-iid)
  --num_shards NUM_SHARDS
                        Total number of shards for Non-IID distribution
  --shards_per_client SHARDS_PER_CLIENT
                        Number of shards assigned to each client
  --train_epochs TRAIN_EPOCHS
                        Number of training epochs for each client
  --batch_size BATCH_SIZE
                        Batch size for client training
  --optimizer OPTIMIZER
                        Optimizer type (adam, sgd, etc.)
  --lr LR               Learning rate for the optimizer
  --loss_fn LOSS_FN     Loss function type (cross_entropy, mse)
```

To run, use the following command:

```bash
python main.py
```
