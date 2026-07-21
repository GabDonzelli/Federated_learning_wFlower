"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import Net, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()

#create a customized strategy to be able to get the node and partition relation
class CustomFedAvg(FedAvg):
    def __init__(self, partition_to_node, **kwargs):
        super().__init__(**kwargs)
        self.partition_to_node = partition_to_node  # Dictionary to store partition-to-node mapping

    def aggregate_train(self, server_round, replies):
        replies = list(replies)

        for reply in replies: 
            if reply.has_content(): 
                partition_id = int (reply.content["metrics"]["partition-id"]
                                    )
                node_id = reply.metadata.src_node_id
                self.partition_to_node[partition_id] = node_id  # Store the mapping
        print( "Mapping : ", self.partition_to_node,
              )
        return super().aggregate_train(server_round, replies)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # list of node and partition relation 
    partition_to_node = {} 

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    #Show SuperNodes available
    node_ids = list(grid.get_node_ids())
    print(f"SuperNodes available: {node_ids}")

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    # Start strategy, run FedAvg for `num_rounds`
    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord({"lr": lr}),
        num_rounds=num_rounds,
        evaluate_fn=global_evaluate,
    )

    if context.run_config["save-model"]:
        # Save final model to disk
        print("\nSaving final model to disk...")
        state_dict = result.arrays.to_torch_state_dict()
        torch.save(state_dict, "final_model.pt")


def global_evaluate(server_round: int, arrays: ArrayRecord) -> MetricRecord:
    """Evaluate model on central data."""

    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(arrays.to_torch_state_dict())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load entire test set
    test_dataloader = load_centralized_dataset()

    # Evaluate the global model on the test set
    test_loss, test_acc = test(model, test_dataloader, device)

    # Return the evaluation metrics
    return MetricRecord({"accuracy": test_acc, "loss": test_loss})


