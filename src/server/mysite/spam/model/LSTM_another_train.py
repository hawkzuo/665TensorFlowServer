import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.contrib import legacy_seq2seq

import numpy as np


class Model():
    def __init__(self, config, training=True):
        self.config = config

        # cell function
        cell_fn = rnn.BasicLSTMCell

        cells = []

        # Generate the cells for the LSTM model
        for _ in range(config.num_layers):
            cell = cell_fn(config.rnn_size)
            if training and (config.output_keep_prob < 1.0 or config.input_keep_prob < 1.0):
                cell = rnn.DropoutWrapper(cell,
                                          input_keep_prob=config.input_keep_prob,
                                          output_keep_prob=config.output_keep_prob)
            cells.append(cell)

        # Combine into MultiRNNCell
        self.cell = cell = rnn.MultiRNNCell(cells, state_is_tuple=True)






