import argparse
import os
from pathlib import Path

import tensorflow as tf
import utils
import flwr as fl
from typing import Dict, Optional, Tuple

class ClientTrainer(object):
    def __init__(self):
        self.model = None
        self.x_train = None
        self.y_train = None
        self.x_test = None
        self.y_test = None

    def set_model(self, model):
        self.model = model
    
    def set_data(self, x_train, x_test, y_train, y_test):
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
    
    def set_ips(self, server_ip, monitor_ip, client_ip):
        self.server_ip = server_ip
        self.monitor_ip = monitor_ip
        self.client_ip = client_ip

    def train(self, fit_func):
        if self.x_train is None or self.model is None:
            raise Exception("Data or Model is not provided")
        client = NPClient(self.model, self.x_train, self.y_train, self.x_test, self.y_test, fit_func, self.monitor_ip, self.client_ip)
        fl.client.start_numpy_client(
            server_address= self.server_ip+":8080",
            client=client,
            #root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
        )
        

# Define Flower client
class NPClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, fit_func, monitor_ip, client_ip):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.fit_func = fit_func
        self.monitor_ip = monitor_ip
        self.client_ip = client_ip

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        server_round: int = config["server_round"]

        # Train the model using hyperparameters from config
        if self.fit_func is None:

            history = self.model.fit(
                self.x_train,
                self.y_train,
                batch_size,
                epochs,
                validation_split=0.1,
            )
        else: history = self.fit_func(self.model, self.x_train, self.y_train, batch_size, epochs)

        # Return updated model parameters and results
        parameters_prime = self.model.get_weights()
        num_examples_train = len(self.x_train)


        results = {
            "loss": history.history["loss"][0],
            "accuracy": history.history["accuracy"][0],
            "val_loss": history.history["val_loss"][0],
            "val_accuracy": history.history["val_accuracy"][0],
        }
        his_summary = history.history

        result = utils.post_metrics(
            monitor_ip = self.monitor_ip,
            type = "client",
            ip_addr = self.client_ip, 
            round = server_round,
            accuracy = his_summary["accuracy"], 
            loss = his_summary["loss"],
            val_accuracy = his_summary["val_accuracy"], 
            val_loss = his_summary["val_loss"],
        )
        print("Content Recieved: ", result.text)

        return parameters_prime, num_examples_train, results

    def evaluate(self, parameters, config):
        """Evaluate parameters on the locally held test set."""

        # Update local model with global parameters
        self.model.set_weights(parameters)

        # Get config values
        steps: int = config["val_steps"]

        # Evaluate global model parameters on the local test data and return results
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, 32, steps=steps)
        num_examples_test = len(self.x_test)
        return loss, num_examples_test, {"accuracy": accuracy}

class ServerController(object):
    def __init__(self):
        self.model = None
        self.X = None
        self.y = None

    def set_model(self, model):
        self.model = model
    
    def set_data(self, X, y):
        self.X, self.y = X, y
    
    def set_ips(self, server_ip, monitor_ip):
        self.server_ip = server_ip
        self.monitor_ip = monitor_ip

    def start(self, num_rounds, port='8080', evaluate_func=None):
        if self.X is None or self.model is None:
            raise Exception("Data or Model is not provided")
        
        self.strategy = fl.server.strategy.FedAvg(
            fraction_fit=0.3,
            fraction_evaluate=0.2,
            min_fit_clients=3,
            min_evaluate_clients=2,
            min_available_clients=3,
            evaluate_fn=self.get_evaluate_fn(evaluate_func),
            on_fit_config_fn=fit_config,
            on_evaluate_config_fn=evaluate_config,
            initial_parameters=fl.common.ndarrays_to_parameters(self.model.get_weights()),
        )

        # Start Flower server (SSL-enabled) for four rounds of federated learning
        fl.server.start_server(
            server_address="0.0.0.0:"+port,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=self.strategy,
            # certificates=(
            #     Path(".cache/certificates/ca.crt").read_bytes(),
            #     Path(".cache/certificates/server.pem").read_bytes(),
            #     Path(".cache/certificates/server.key").read_bytes(),
            # ),
        )

    def get_evaluate_fn(self, evaluate_func=None):
        """Return an evaluation function for server-side evaluation."""

        # Load data and model here to avoid the overhead of doing it in `evaluate` itself
        # (x_train, y_train), _ = tf.keras.datasets.cifar10.load_data()

        # # Use the last 5k training examples as a validation set
        # x_val, y_val = x_train[45000:50000], y_train[45000:50000]
        # x_tr, y_tr = x_train[:45000], y_train[:45000]
        
        # path = os.path.join('/data', filename)
        # x_tr, x_val, y_tr, y_val = load_cifar(path)

        # The `evaluate` function will be called after every round
        def evaluate(
            server_round: int,
            parameters: fl.common.NDArrays,
            config: Dict[str, fl.common.Scalar],
        ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
            self.model.set_weights(parameters)  # Update model with the latest parameters
            if evaluate_func is None: loss, accuracy = self.model.evaluate(self.X, self.y)
            else: loss, accuracy = evaluate_func(self.model, self.X, self.y)
            # val_loss, val_accuracy = model.evaluate(x_val, y_val)  

            result = utils.post_metrics(
                monitor_ip = self.monitor_ip,
                type = "server",
                ip_addr = None, 
                round = server_round,
                accuracy = accuracy, 
                loss = loss,
                val_accuracy = None, 
                val_loss = None,
            )
            print("Content Recieved: ", result.text)

            return loss, {"accuracy": accuracy}

        return evaluate

def fit_config(server_round: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 32,
        "local_epochs": 1 if server_round < 2 else 2,
        "server_round": server_round,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}