"""pytorchexample: A Flower / PyTorch app."""

import torch
from flwr.app import ArrayRecord, ConfigRecord, Context, MetricRecord, MessageType, RecordDict
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from pytorchexample.task import Net, load_centralized_dataset, test

# Create ServerApp
app = ServerApp()

"""CLients have their artition of the dataset. They also have their node identity, which means where they are trained.
    The clients numbers are their partition number."""

#create a customized strategy to be able to get the node and partition relation
class CustomFedAvg(FedAvg):
    def __init__(self, client_to_node, **kwargs):
        super().__init__(**kwargs)
        # Dictionary to store partition-to-node mapping
        self.client_to_node = client_to_node  
        # Dictionary to store client names
        #self.client_names = {}  

    def configure_train(self, server_round, arrays, config, grid):
        print(f"Configuring training for server round {server_round}...")

        # first round == mapping
        if server_round == 1:

            #to name clients 
            #for client_number, (client_id, node_id) in enumerate ( 
                #sorted(self.client_to_node.items()), start = 1): 

                #client_name = f"Client {client_number}" 
                #self.client_names[node_id] = client_name

                #print(f"{client_name} is on Node {node_id} with Partition {client_id}")


            return super().configure_train(server_round, arrays, config, grid)
        
        if server_round == 2:
            selected_clients = [1,2,3,4,5]
            for i in range(len(selected_clients)):
                client_id = selected_clients[i]
                node_id = self.client_to_node[client_id]
                print(f"Client selected: {selected_clients[i]} and is on Node {node_id} ")
                                   
        if server_round == 3:
            selected_clients = [6,7,8,9,10]
            #pode vir a virar uma funcao print###################################
            for i in range(len(selected_clients)):
                client_id = selected_clients[i]
                node_id = self.client_to_node[client_id]
                print(f"Client selected: {selected_clients[i]} and is on Node {node_id} ")

        # get the node_id of the selected partitions and print them
        selected_nodes = [self.client_to_node[client_id] for client_id in selected_clients]

        # update the config with the current server_round   
        config["server_round"] =  server_round

        #sending the global model to the selected nodes
        record = RecordDict({self.arrayrecord_key: arrays, self.configrecord_key: config, })

        #record = what is sent to the selected nodes, selected_nodes = the nodes that will receive the record
        return self._construct_messages(record, selected_nodes,  MessageType.TRAIN)
        

    def aggregate_train(self, server_round, replies):
        replies = list(replies)

        for reply in replies: 
            if reply.has_content(): 
                client_id = int (reply.content["metrics"]["partition-id"])
                
                client_id = client_id + 1
                node_id = reply.metadata.src_node_id
                
                # Store the mapping
                # only save the mapping once per client
                if client_id not in self.client_to_node:
                    self.client_to_node[client_id] = node_id 

                # Remove partition-id from metrics
                reply.content["metrics"].pop("partition-id", None)  
        
        
        if server_round == 1:
            print(f"\nMapping round {server_round}:")
            for client_id, node_id in sorted(self.client_to_node.items()):
                print(f"Client {client_id}: Node {node_id}")

        return super().aggregate_train(server_round, replies)


@app.main()
def main(grid: Grid, context: Context) -> None:
    """Main entry point for the ServerApp."""

    # list of node and partition relation 
    client_to_node = {} 

    # Read run config
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["learning-rate"]

    # Show mapping of clients and nodes
    node_ids = list(grid.get_node_ids())

    # create a fixed mapping once
    client_to_node = {
        client_id: node_id
        for client_id, node_id in enumerate(node_ids, start=1)
    }

    print("Client mapping:")
    for client_id, node_id in sorted(client_to_node.items()):
        print(f"Client {client_id}: Node {node_id}")

    # Load global model
    global_model = Net()
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    #strategy = FedAvg(fraction_evaluate=fraction_evaluate)

    #initialize the custom strategy
    strategy = CustomFedAvg(
        client_to_node=client_to_node,
        fraction_evaluate=fraction_evaluate,
        min_train_nodes = 2,
        min_evaluate_nodes = 2,
        min_available_nodes = 2,
    )

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


