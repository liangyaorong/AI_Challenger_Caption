'''
author: Liang YaoRong
email: liangyaorong1995@outlook.com
'''


import tensorflow as tf
from tensorflow.python.ops import array_ops
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import variable_scope
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.rnn import RNNCell
from tensorflow.contrib.rnn import LSTMStateTuple
from collections import namedtuple
Linear = core_rnn_cell._Linear



_PeterAttentionLSTMStateTuple = namedtuple("PeterAttentionLSTMStateTuple", ["c1", "h1", "c2", "h2"])

class PeterAttentionStateTuple(_PeterAttentionLSTMStateTuple):
    __slots__ = ()

    @property
    def dtype(self):
        (c1, h1, c2, h2) = self
        if c1.dtype != h1.dtype:
            raise TypeError("Inconsistent internal state: %s vs %s"%(str(c1.dtype), str(c2.dtype)))
        return c1.dtype



class PeterAttentionWrapper(RNNCell):
    '''
    paper: 'Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering'
    '''

    def __init__(self, cells, visual_features, attention_vec_size, mode, name=None):
        '''

        :param cell: [LSTM1, LSTM2]
        :param visual_features: low layer of the net, size:[batch_size, width, height, depth]
        :param attention_vec_size: dim of w3 where w3*(W1*v_i + W2*h1)
        '''
        super(PeterAttentionWrapper, self).__init__(name=name)
        assert mode in ["train", "eval", "inference"]
        self._mode = mode
        self._cells = cells
        self._lstm_cell_1 = cells[0]
        self._lstm_cell_2 = cells[1]
        self._attention_vec_size = attention_vec_size

        self._v = visual_features
        [self.batch_size, self.height, self.width, self.depth] = self._v.get_shape()
        self._v = tf.reshape(self._v, [self.batch_size, self.height * self.width, 1, self.depth])
        self._v = tf.transpose(self._v, perm=[0, 3, 2, 1])  # size:[batch_size, depth, 1, height*width]
        self._v_bar = slim.flatten(tf.reduce_mean(self._v, 1))  # size:[batch_size, height*width]


    @property
    def state_size(self):
        return PeterAttentionStateTuple._make([size for cell in self._cells for size in cell.state_size])

    @property
    def output_size(self):
        return self._lstm_cell_2.output_size

    def zero_state(self, batch_size, dtype):
        return PeterAttentionStateTuple._make([state for cell in self._cells for state in cell.zero_state(batch_size, dtype)])



    def call(self, inputs, state):
        '''
        :param inputs: word_embedding_input, size [batch_size, embedding_length]
        :param state: namedtuple(PeterAttentionLSTMStateTuple, [c1, h1, c2, h2])
        :return:
        Decoder_output:
        new_state_1: state_1 after attention_Decoder, size as state_1
        new_state_2: state_2 after attention_Decoder, size as state_2

        '''

        state_1 = LSTMStateTuple(state[0], state[1])
        state_2 = LSTMStateTuple(state[2], state[3])

        # in train mode, tile_time=1 ensure the shape[0] of _v_bar is batch size. in inference mode tile_times mean beam width
        tile_times = 1 if self._mode == "train" else tf.shape(inputs)[0]

        with variable_scope.variable_scope("Top_down_LSTM"):
            lstm_cell_1_input = array_ops.concat([state_1.h, tf.tile(self._v_bar, [tile_times, 1]), inputs], 1)
            h1, new_state_1 = self._lstm_cell_1(lstm_cell_1_input, state_1) #size:[batch_size, num_units]
            self.num_units = h1.get_shape()[1]

        with variable_scope.variable_scope("Attention"):
            # attention, alpha_i = w3*(W1*v_i + W2*h1)

            # 1-by-1 convolution to realize W1*v_i
            W1 = tf.get_variable("Attention_W1", [1, 1, self.height*self.width, self._attention_vec_size])
            y1 = tf.nn.conv2d(input=self._v, filter=W1, strides=[1, 1, 1, 1], padding="SAME") #size:[batch_size, depth, 1, attention_vec_size]

            # matmul
            W2 = tf.get_variable("Attention_W2", [self.num_units, self._attention_vec_size])
            y2 = tf.matmul(h1, W2) #size:[batch_size, attention_vec_size]
            y2 = tf.reshape(y2, [-1, 1, 1, self._attention_vec_size])

            # W3
            W3 = tf.get_variable("Attention_W3", [self._attention_vec_size])

            # calculate weight alpha
            alpha = tf.reduce_sum(W3 * tf.tanh(y1 + y2), [2, 3]) #size:[batch_size, depth]
            alpha = tf.nn.softmax(alpha)

            # calculate attention vector d
            v_cat = tf.reduce_sum(tf.reshape(alpha, [-1, self.depth, 1, 1]) * self._v, [1, 2], name="v_cat") #size:[batch_size, height*width]

        with variable_scope.variable_scope("Language_LSTM"):
            lstm_cell_2_input = array_ops.concat([v_cat, h1], 1)
            h2, new_state_2 = self._lstm_cell_2(lstm_cell_2_input, state_2)

        new_state = PeterAttentionStateTuple(new_state_1.c, new_state_1.h, new_state_2.c, new_state_2.h)

        return h2, new_state










