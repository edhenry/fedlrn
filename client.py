import codecs
import json
import logging
import pickle
import random
import time

import numpy as np
import socketio

import tensorflow as tf
import utility
import datasources
import threading

# Basic loggging config 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Logging file handler
handler = logging.FileHandler('client.log')
handler.setLevel(logging.INFO)

# Logging format
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

class LocalModel(object):
    """Local model definition for local training on data sources

    """
    def __init__(self, model_config, data_collected):
        self.model_config = model_config
        
        # TODO: Load model from tensorflow definition as defined by TF Server
        # We can have the server locally define a model and transport that model 
        # definition down to the client for instantiation

        # Multiple weight initalization schemes will need to be supported
        # those defined, instantiated, serialized, and transported to the
        # clients from the server and those initialized locally by the 
        # client logic itself
        self.model_json_graph = tf.Graph()

        with self.model_json_graph.as_default():
            self.session1 = tf.Session()
            with self.session1.as_default():
                self.model = tf.keras.models.model_from_json(model_config['model_json'])
                self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                                optimizer=tf.keras.optimizers.Adadelta(),
                                metrics=['accuracy'])

        # A tuple of data will be defined and passed into this class at 
        # instantiation -- need to figure out how to pipe this into the
        # model
        train_data, test_data, valid_data = data_collected
        self.x_train = np.array([tup[0] for tup in train_data])
        self.y_train = np.array([tup[1] for tup in train_data])
        self.x_test = np.array([tup[0] for tup in test_data])
        self.y_test = np.array([tup[1] for tup in test_data])
        self.x_valid = np.array([tup[0] for tup in valid_data])
        self.y_valid = np.array([tup[1] for tup in valid_data])
    
    def get_weights(self):
        """
        Return the parameters of the locally trained model
        """
        return self.model.get_weights()

    def set_weights(self, new_weights):
        """
        Initilize of reload weights of the model

        This can support the idea of a model that is being fine tuned or one that is being
        trained from scratch
        """
        # if fine_tune == False:
        #     # TODO: Implement method to intialize model params
        #     self.model.set_weights()
        #     logger.info("fine_tune set to {}, training new instance of our model".format(fine_tune))
        #     return False
        # else:
        with self.model_json_graph.as_default():
            with self.session1.as_default():
                self.model.set_weights(new_weights)
    
    def score(self):
         with self.model_json_graph.as_default():
            with self.session1.as_default():
                score = self.model.evaluate(self.x_train, self.y_train, verbose=0)
                logger.info("Train Loss : {}".format(score[0]))
                logger.info("Train Accuracy {}".format(score[1]))
                return self.model.get_weights(), score[0], score[1]

    def train_single_round(self):
        """
        Execute one round of training
        """
        with self.model_json_graph.as_default():
            with self.session1.as_default():
                self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                               optimizer=tf.keras.optimizers.Adadelta(),
                               metrics=['accuracy'])
                
                self.model.fit(self.x_train, self.y_train,
                           epochs=self.model_config['epochs_per_round'],
                           batch_size=self.model_config['batch_size'],
                           verbose=1,
                           validation_data=(self.x_valid, self.y_valid))
                
        return self.score()
                        

    def validate(self):
        """
        Pass over validation set

        """
        with self.model_json_graph.as_default():
            with self.session1.as_default():
                score = self.model.evaluate(self.x_valid, self.y_valid, verbose=0)
                logger.info("Validation Loss : {}".format(score[0]))
                logger.info("Validation Accuracy : {}".format(score[1]))
        
        return score

    def evaluate(self):
        """
        Pass over test set

        """
        with self.model_json_graph.as_default():
            with self.session1.as_default():
                score = self.model.evaluate(self.x_test, self.y_test, verbose=0)
                logger.info("Test Loss : {}".format(score[0]))
                logger.info("Test Accuracy : {}".format(score[1]))
        
        return score

class FederatedClient(object):
    """Federated Client

    This is a process that can wake up and go to sleep intermettently according to the parameter
    passing logic that is desired by the user

    This object will send/receive model definitions with the server and return model parameters
    back to the server in either a push or pull fashion

    """
    MAX_DATASET_SIZE_KEPT = 1200

    def __init__(self, server_host, server_port, datasource):
        self.local_model = None
        self.datasource = datasource()

        self.sio_client =  socketio.Client()
        # TODO : Make this configurable via a client configuration file
        self.sio_client.connect('http://127.0.0.1:5000')

        self.register_handlers()
        logger.info("Sent client wakeup")
        self.sio_client.emit('client_wake_up')
        self.sio_client.wait()

    def on_init(self, *args):
        model_config = args[0]
        logger.info(f"Preparing local data based on server model_config")

        fake_data, _ = self.datasource.fake_non_iid_data(
            min_train = model_config['min_training_size'],
            max_train = FederatedClient.MAX_DATASET_SIZE_KEPT,
            data_split = model_config['data_split']
        )
        
        self.local_model = LocalModel(model_config, fake_data)
        self.sio_client.emit('client_ready', {
            'train_size': self.local_model.x_train.shape[0],
        })

    def register_handlers(self):
        """
        Handlers used for SocketIO messaging
        """
        @self.sio_client.on('connect')
        def on_connect():
            logger.info("Client Connected!")
        
        @self.sio_client.on('disconnect')
        def on_disconnect():
            logger.info("Client Disconnected!")
        
        @self.sio_client.on('reconnect')
        def on_reconnect():
            logger.info("Client Reconnected!")
        
        def on_request_update(*args):
            logger.info('Server requesting update!')
            req = args[0]
            
            if req['weights_format'] == 'pickle':
                weights = utility.pickle_string_to_obj(req['current_weights'])
            
            self.local_model.set_weights(weights)
            my_weights, train_loss, train_accuracy = self.local_model.train_single_round()

            logger.info(f"Client Weights : {my_weights}")
            
            response = {
                'round_number': float(req['round_number']),
                'weights': utility.obj_to_pickle_string(my_weights),
                'train_size': self.local_model.x_train.shape[0],
                'valid_size': self.local_model.x_valid.shape[0],
                'train_loss': float(train_loss),
                'train_accuracy': float(train_accuracy)
            }

            if req['run_validation']:
                valid_loss, valid_accuracy = self.local_model.validate()
                response['valid_loss'] = float(valid_loss)
                response['valid_accuracy'] = float(valid_accuracy)
        
            self.sio_client.emit('client_update', response)
        
        def on_stop_and_eval(*args):
            req = args[0]
            if req['weights_format'] == 'pickle':
                weights = utility.pickle_string_to_obj(req['current_weights'])
            self.local_model.set_weights(weights)
            test_loss, test_accuracy = self.local_model.evaluate()

            response = {
                'test_size': self.local_model.x_test.shape[0],
                'test_loss': float(test_loss),
                'test_accuracy': float(test_accuracy)
            }

            self.sio_client.emit('client_evaluation', response)
        
        self.sio_client.on('connect', on_connect)
        self.sio_client.on('disconnect', on_disconnect)
        self.sio_client.on('reconnect', on_reconnect)
        self.sio_client.on('init', lambda *args: self.on_init(*args))
        self.sio_client.on('request_update', on_request_update)
        self.sio_client.on('stop_and_eval', on_stop_and_eval)


    def intermittent_sleep(self, p=.1, low=1, high=5):
        """Intermittently sleep the client with some probability < p for sime random time between 1 and 100 seconds
        
        Keyword Arguments:
            p {float} -- [probability of sleep firing] (default: {.1})
            low {int} -- [lower bound on sleep time] (default: {10})
            high {int} -- [upper bound on sleep time] (default: {100})
        """
        if (random.random() < p):
            time.sleep(random.randint(low, high))


if __name__ == '__main__':
    FederatedClient("127.0.0.1", 5000, datasources.Mnist)
