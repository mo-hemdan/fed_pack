import argparse
import os
from pathlib import Path

import tensorflow as tf
import utils
import flwr as fl

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
    
    def train(self, server_ip, fit_func):
        if self.x_train == None or self.model == None:
            raise Exception("Data or Model is not provided")
        client = CifarClient(self.model, self.x_train, self.y_train, self.x_test, self.y_test, fit_func)
        fl.client.start_numpy_client(
            server_address=server_ip+":8080",
            client=client,
            #root_certificates=Path(".cache/certificates/ca.crt").read_bytes(),
        )
        

# Define Flower client
class CifarClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train, x_test, y_test, fit_func):
        self.model = model
        self.x_train, self.y_train = x_train, y_train
        self.x_test, self.y_test = x_test, y_test
        self.fit_func = fit_func

    def get_properties(self, config):
        """Get properties of client."""
        raise Exception("Not implemented")

    def get_parameters(self):
        """Get parameters of the local model."""
        raise Exception("Not implemented (server-side parameter initialization)")

    def fit(self, parameters, config):
        """Train parameters on the locally held training set."""
        global client_id

        # Update local model parameters
        self.model.set_weights(parameters)

        # Get hyperparameters for this round
        batch_size: int = config["batch_size"]
        epochs: int = config["local_epochs"]
        server_round: int = config["server_round"]

        # Train the model using hyperparameters from config
        if fit_func == None:

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
            monitor_ip = monitor_ip,
            type = "client",
            ip_addr = client_id, 
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