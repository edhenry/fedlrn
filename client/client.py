import codecs
import logging
import json
import pickle
import random
import time

import numpy as np
import socketio
import tensorflow as tf

import utility

# Basic loggging config 
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
        self.model = tf.keras.models.model_from_json(model_config['model_json'])
        
        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                           optimizer=tf.keras.optimizers.Adadelta(),
                           metrics=['accuracy'])

        # A tuple of data will be defined and passed into this class at 
        # instantiation -- need to figure out how to pipe this into the
        # model
        train_data, test_data, validate_data = data_collected
        self.x_train = np.array([tup[0] for tup in train_data])
        self.y_train = np.array([tup[1] for tup in train_data])
        self.x_test = np.array([tup[0] for tup in test_data])
        self.y_test = np.array([tup[1] for tup in test_data])
        self.x_valid = np.array([tup[1] for tup in validate_data])
        self.y_valid = np.array([tup[1] for tup in validate_data])

    def get_weights(self):
        """
        Return the parameters of the locally trained model
        """

    def set_weights(self, fine_tune: bool, new_weights):
        """
        Initilize of reload weights of the model

        This can support the idea of a model that is being fine tuned or one that is being
        trained from scratch
        """
        if fine_tune == False:
            # TODO: Implement method to intialize model params
            self.model.set_weights()
            logger.info("fine_tune set to {}, training new instance of our model".format(fine_tune))
            return False
        else:
            self.model.set_weights(new_weights)

    def train_single_round(self):
        """
        Execute one round of training
        """
        self.model.compile(loss=tf.keras.losses.categorical_crossentropy,
                           optimizer=tf.keras.optimizers.Adadelta(),
                           metrics=['accuracy'])
        
        self.model.fit(self.x_train, self.y_train,
                       epoch=self.model_config['epoch_per_round'],
                       batch_size=self.model_config['batch_size'],
                       verbose=1,
                       validation_data=(self.x_valid, self.y_valid))
        
        score = self.model.evaluate(self.x_train, self.y_train, verbose=0)
        logger.info("Train Loss : {}".format(score[0]))
        logger.info("Train Accuracy {}".format(score[1]))
        return self.model.get_weights(), score[0], score[1]

    def validate(self):
        """
        Pass over validation set

        """
        score = self.model.evaluate(self.x_valid, self.y_valid, verbose=0)
        logger.info("Validation Loss : {}".format(score[0]))
        logger.info("Validation Accuracy : {}".format(score[1]))
        
        return score

    def evaluate(self):
        """
        Pass over test set

        """
        score = self.model.evaluate(self.x_test, self.y_test, verboser=0)
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

    def __init__(self, server_host, server_port, datasource):
        self.local_model = None
        self.datasource = datasource

        self.sio_client =  socketio.Client()
        # TODO : Make this configurable via a client configuration file
        self.sio_client.connect('http://192.168.1.1:5000')

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
            logger.info('Client requesting update!')
            req = args[0]

            if req['weights_format'] == 'pickle':
                weights = utility.pickle_string_to_obj(req['current_weights'])
            
            self.local_model.set_weights(weights)
            my_weights, train_loss, train_accuracy = self.local_model.train_one_round()
            
            response = {
                'round_number': req['round_number'],
                'weights': utility.obj_to_pickle_string(my_weights),
                'train_size': self.local_model.x_train.shape[0],
                'valid_size': self.local_model.x_valid.shape[0],
                'train_loss': train_loss,
                'train_accuracy': train_accuracy
            }

            if req['run_validation']:
                valid_loss, valid_accuracy = self.local_model.validate()
                response['valid_loss'] = valid_loss
                response['valid_accuracy'] = valid_accuracy
        
            self.sio_client.emit('client_update', response)
        
        def on_stop_and_eval(*args):
            req = args[0]
            if req['weights_format'] == 'pickle':
                weights = utility.pickle_string_to_obj(req['current_weights'])
            self.local_model.set_weights(weights)
            test_loss, test_accuracy = self.local_model.evaluate()

            response = {
                'test_size': self.local_model.x_test.shape[0],
                'test_loss': test_loss,
                'test_accuracy': test_accuracy
            }

            self.sio_client.emit('client_evaluation', response)
        
        self.sio_client.on('connect', on_connect)
        self.sio_cleint.on('disconnect', on_disconnect)
        self.sio_client.on('reconnect', on_reconnect)
        self.sio_client.on('init', lambda *args: self.on_init(*args))
        self.sio_client.on('request_update', on_request_update)
        self.sio_client.on('stop_and_eval', on_stop_and_eval)


    def intermittent_sleep(self, p=.1, low=10, high=100):
        """Intermittently sleep the client with some probability < p for sime random time between 1 and 100 seconds
        
        Keyword Arguments:
            p {float} -- [probability of sleep firing] (default: {.1})
            low {int} -- [lower bound on sleep time] (default: {10})
            high {int} -- [upper bound on sleep time] (default: {100})
        """
        if (random.random() < p):
            time.sleep(random.randint(low, high))


if __name__ == '__main__':
    FederatedClient("127.0.0.1", 5000, datasource.Mnist)



