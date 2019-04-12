import codecs
import json
import logging
import pickle
import random
import sys
import time
import uuid

import msgpack
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from flask_socketio import *
from flask_socketio import SocketIO

import utility

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Logging file handler
handler = logging.FileHandler('server.log')
handler.setLevel(logging.INFO)

# Logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class GlobalModel(object):
    """
    Global model for distribution to clients
    """
    def __init__(self):
        self.model = self.build_model()

        self.current_weights = self.model.get_weights()
        self.previous_training_loss = None

        # track all losses across rounds
        # losses[i] = [round_number, timestamp, loss]
        self.train_losses = []
        self.validation_losses = []
        self.training_accuracies = []
        self.validation_accuracies = []

        self.training_start_time = int(round(time.time()))

    
    def build_model(self):
        raise NotImplementedError()
    
    def update_weights(self, client_weights, client_size):
        new_weights = [np.zeros(w.shape) for w in self.current_weights]
        total_size = np.sum(client_size)
        for c in range(len(client_weights)):
            for i in range(len(new_weights)):
                client_weights_test = client_weights[c][i]
                client_size_test = client_size[c]
                new_weights[i] += np.true_divide((client_weights[c][i] * np.float32(client_size[c])), np.float32(total_size))
        
        self.current_weights = new_weights

    def aggregate_loss_accuracy(self, client_losses, client_accuracies, client_sizes):
        total_size = np.sum(client_sizes)

        aggregate_loss = np.sum(client_losses[i] / (total_size * client_sizes[i]) for i in range(len(client_sizes)))
        aggregate_accuracies = np.sum(client_accuracies[i] / (total_size * client_sizes[i]) for i in range(len(client_sizes)))
        
        return aggregate_loss, aggregate_accuracies


    def aggregate_train_loss_accuracy(self, client_losses, client_accuracies, client_sizes, current_round):
        """Aggregate the training loss and accuracies of clients participating in training round
        
        Arguments:
            client_losses {list} -- list of participating client losses
            client_accuracies {[type]} -- list of participating client accuracies
            client_sizes {int} -- number of participating clients
            current_round {int} -- current round of training
        """
        current_time = int(round(time.time())) - self.training_start_time
        aggregate_loss, aggregate_accuracies = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        self.train_losses += [[current_round, current_time, aggregate_loss]]
        self.training_accuracies += [[current_round, current_round, aggregate_accuracies]]
        with open('stat.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        
        return aggregate_loss, aggregate_accuracies

    def aggregate_valid_loss_accuracy(self, client_losses, client_accuracies, client_sizes, current_round):
        """Aggregate validation loss accuracies across participating client devices
        
        Arguments:
            client_losses {list} -- list of participating client losses
            client_accuracies {list} -- list of participating client accuracies
            client_sizes {int} -- number of participating clients
            current_round {int} -- current round of training
        """
        current_time = int(round(time.time())) - self.training_start_time
        aggregate_loss, aggregate_accuracies = self.aggregate_loss_accuracy(client_losses, client_accuracies, client_sizes)
        self.validation_losses += [[current_round, current_time, aggregate_loss]]
        self.validation_accuracies += [[current_round, current_time, aggregate_accuracies]]
        with open('stats.txt', 'w') as outfile:
            json.dump(self.get_stats(), outfile)
        
        return aggregate_loss, aggregate_accuracies        
    
    def get_stats(self):
        return {
            "train_loss": self.train_losses,
            "validation_loss": self.validation_losses,
            "train_accuracy": self.training_accuracies,
            "validation_accuracy": self.validation_accuracies
        }

#TODO Should we have an object per global model to push to clients, per use case?
# Should we break that out logically here?

class GlobalModel_MNIST_CNN(GlobalModel):
    def __init__(self):
        super(GlobalModel_MNIST_CNN, self).__init__()
    
    def build_model(self):
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Conv2D(32, kernel_size=(3,3),
                                         activation='relu',
                                         input_shape=(28, 28, 1), ))
        model.add(tf.keras.layers.Conv2D(64, (3,3), activation='relu'))
        model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
        model.add(tf.keras.layers.Dropout(0.25))
        model.add(tf.keras.layers.Flatten())
        model.add(tf.keras.layers.Dense(128, activation='relu'))
        model.add(tf.keras.layers.Dropout(0.5))
        model.add(tf.keras.layers.Dense(10, activation='softmax'))
        
        model.compile(loss=tf.keras.losses.categorical_crossentropy,
                      optimizer=tf.keras.optimizers.Adadelta(),
                      metrics=['accuracy'])
        
        return model

class FedLearnServer(object):

    # Federated Learning Options
    # TODO break this out into a configuration file
    MIN_NUMBER_WORKERS = 3
    MAX_NUMBER_ROUNDS = 50
    
    # This should be tunable according to the total number of clients
    # some ratio of T (arrays of measurements on a client) and G (groups of clients)
    # this ratio will likely guide the overall global convergence rates and define
    # the profile of the workload from a hardware perspective
    NUM_CLIENTS_CONTACTED_PER_ROUND = 3
    ROUNDS_BETWEEN_VALIDATION = 2

    def __init__(self, global_model, host, port):
        self.global_model = global_model()

        self.ready_client_sids = set()

        self.app = Flask(__name__)
        self.socketio = SocketIO(self.app)
        self.host = host
        self.port = port

        self.model_id = str(uuid.uuid4())

        # training states for participating clients and server logic
        self.current_round = -1
        self.current_round_client_updates = []
        self.eval_client_updates = []

        self.register_handles()

        @self.app.route('/')
        def dashboard():
            return render_template('dashboard.html')

        @self.app.route('/stats')
        def status_page():
            return json.dumps(self.global_model.get_stats())
        
    def register_handles(self):
        
        @self.socketio.on('connect')
        def handle_connect():
            logger.info(f"{request.sid} connected!")
        
        @self.socketio.on('reconnected')
        def handle_reconnect():
            logger.info(f"{request.sid} reconnected!")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            logger.info(f"{request.sid} disconnected!")
            # prune client from set of clients
            if request.sid in self.ready_client_sids:
                self.ready_client_sids.remove(request.sid)
        
        @self.socketio.on('client_wake_up')
        def handle_wake_up():
            logger.info(f"client_wake_up : {request.sid}")
            emit('init', {
                "model_json": self.global_model.model.to_json(),
                "model_id": self.model_id,
                "min_training_size": 1200,
                "data_split": (0.6, 0.3, 0.1),
                "epochs_per_round": 1,
                "batch_size": 10
            })

        @self.socketio.on('client_ready')
        def handle_client_ready(data):
            logger.info(f"Client {request.sid} Ready for Training using {data}")
            self.ready_client_sids.add(request.sid)
            if len(self.ready_client_sids) >= FedLearnServer.MIN_NUMBER_WORKERS and self.current_round == -1:
                self.train_next_round()
        
        @self.socketio.on('client_update')
        def handle_client_update(data):
            logger.info(f"Received Client Update of {sys.getsizeof(data)} bytes")
            logger.info(f"Handle Client Update for Client {request.sid}")
            # for x in data:
            #     if x != 'weights':
            #         #logger.info(f"{x}, {data[x]}")
            
            if data['round_number'] == self.current_round:
                self.current_round_client_updates += [data]
                self.current_round_client_updates[-1]['weights'] = utility.pickle_string_to_obj(data['weights'])

                if len(self.current_round_client_updates) > FedLearnServer.NUM_CLIENTS_CONTACTED_PER_ROUND * .7:
                    self.global_model.update_weights(
                        [x['weights'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                    )
                    aggregate_train_loss, aggregate_train_accuracy = self.global_model.aggregate_train_loss_accuracy(
                        [x['train_loss'] for x in self.current_round_client_updates],
                        [x['train_accuracy'] for x in self.current_round_client_updates],
                        [x['train_size'] for x in self.current_round_client_updates],
                        self.current_round
                    )

                    logger.info(f"aggregate_train_loss : {aggregate_train_loss}")
                    logger.info(f"aggregate_train_accuracy : {aggregate_train_accuracy}")

                    if 'validation_loss' in self.current_round_client_updates[0]:
                        aggregate_validation_loss, aggregate_validation_accuracy = self.global_model.aggregate_valid_loss_accuracy(
                            [x['valid_loss'] for x in self.current_round_client_updates],
                            [x['valid_accuracy'] for x in self.current_round_client_updates],
                            [x['valid_size'] for x in self.current_round_client_updates],
                            self.current_round
                        )

                        logger.info(f"aggregate_valid_loss : {aggregate_validation_loss}")
                        logger.info(f"aggregate_validation_accuracy : {aggregate_validation_accuracy}")

                    if self.global_model.previous_training_loss is not None and (self.global_model.previous_training_loss - aggregate_train_loss) / self.global_model.previous_training_loss < 0.01:
                        logger.info("Convergence! Starting test phase...")
                        self.stop_and_eval()
                        return
                    
                    self.global_model.previous_training_loss = aggregate_train_loss
                
                    if self.current_round >= FedLearnServer.MAX_NUMBER_ROUNDS:
                        self.stop_and_eval()
                    else:
                        self.train_next_round()
        
        @self.socketio.on('client_eval')
        def handle_client_eval(data):
            if self.eval_client_updates is None:
                return
            logger.info(f"Handle client {request.sid} evaluation")
            logger.info(f"Evaluation response {data}")
            self.eval_client_updates += [data]

            if len(self.eval_client_updates) > FedLearnServer.NUM_CLIENTS_CONTACTED_PER_ROUND * .7:
                aggregate_test_loss, aggregate_test_accuracy = self.global_model.aggregate_loss_accuracy(
                    [x['test_loss'] for x in self.eval_client_updates],
                    [x['test_accuracy'] for x in self.eval_client_updates],
                    [x['test_size'] for x in self.eval_client_updates],
                )
                logger.info(f"Aggregate Test Loss : {aggregate_test_loss}")
                logger.info(f"Aggregate Test Accuracy : {aggregate_test_accuracy}")
                logger.info(f"===== DONE =====")
                self.eval_client_updates = None
        
    def train_next_round(self):
        self.current_round += 1
        self.current_round_client_updates = []

        logger.info(f"===== ROUND {self.current_round} =====")
        logger.info(f"ready_client_sids {list(self.ready_client_sids)}")
        client_sids_selected = random.sample(list(self.ready_client_sids), FedLearnServer.NUM_CLIENTS_CONTACTED_PER_ROUND)
        logger.info(f"Request Updates from {client_sids_selected}")

        for rid in client_sids_selected:
            emit('request_update', {
                'model_id': self.model_id,
                'round_number': self.current_round,
                'current_weights': utility.obj_to_pickle_string(self.global_model.current_weights),
                'weights_format': 'pickle',
                'run_validation': self.current_round % FedLearnServer.ROUNDS_BETWEEN_VALIDATION == 0,
            }, room=rid)

    def stop_and_eval(self):
        self.eval_client_updates = []
        for rid in self.ready_client_sids:
            emit('stop_and_eval', {
                'model_id': self.model_id,
                'current_weights': utility.obj_to_pickle_string(self.global_model.current_weights),
                'weights_format': 'pickle'
            }, room=rid)

    def start(self):
        self.socketio.run(self.app, host=self.host, port=self.port)        

if __name__ == '__main__':

    # TODO: Pass Server configuration parameters in via a configuration file
    server = FedLearnServer(GlobalModel_MNIST_CNN, "127.0.0.1", 5000)
    logger.info(f"Server Listening on Host IP : 127.0.0.1 and Port : 5000")
    server.start()
