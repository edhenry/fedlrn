import pickle
import tensorflow as tf
import uuid
import logging
import msgpack


import random
import codecs
import numpy as np
import json
import sys
import time

from flask import *
from flask_socketio import SocketIO
from flask_socketio import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GlobalModel(object):
    """
    Global model for distribution to clients
    """
    def __init__(self):
        self.model = self.build_model()

    
    def build_model(self):
        raise NotImplementedError()
    
    def update_weights(self):
        raise NotImplementedError()

    def aggregate_loss_accuracy(self):
        raise NotImplementedError()
    
    def aggregate_train_loss_accuracy(self):
        raise NotImplementedError()

    def aggregate_valid_loss_accuracy(self):
        raise NotImplementedError()
    
    def get_stats(self):
        raise NotImplementedError()

#TODO Should we have an object per global model to push to clients, per use case?
# Should we break that out logically here?

class FedLearnServer(object):

    # Federated Learning Options
    # TODO break this out into a configuration file
    MIN_NUMBER_WORKERS = 5
    MAX_NUMBER_ROUNDS = 50
    
    # This should be tunable according to the total number of clients
    # some ratio of T (arrays of measurements on a client) and G (groups of clients)
    # this ratio will likely guide the overall global convergence rates and define
    # the profile of the workload from a hardware perspective
    NUM_CLIENTS_CONTACTED_PER_SECOND = 5
    ROUNDS_BETWEEN_VALIDATION = 2

    def __init__(self):
        self.global_model = global_model()

    def register_handles(self):
        raise NotImplementedError()
    
    def train_next_round(self):
        raise NotImplementedError()


if __name__ == '__main__':

    # TODO: Pass Server configuration parameters in via a configuration file
    server = FedLearnServer()
    logger.INFO("Server Listening on {}")