import codecs
import json
import logging
import pickle
import random
import threading
import time

import numpy as np
import socketio
import tensorflow as tf

import datasources
import utility
from client import LocalModel


def main():
local_model = LocalModel()


if __name__ == '__main__':
    