import tensorflow as tf

import math

from .utils import flatten

class Embedding(object):
    def __init__(self, vocab_size, embedding_size):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        embedding_matrix_initializer = tf.random_uniform_initializer(-1/math.sqrt(embedding_size), 1/math.sqrt(embedding_size))
        self.embedding_matrix = tf.get_variable("embedding_matrix", (vocab_size, embedding_size),
                                       initializer=embedding_matrix_initializer)

    def __call__(self, idxes):
        return tf.nn.embedding_lookup(self.embedding_matrix, idxes)

class Linear(object):
    def __init__(self, input_sizes, hidden_size, bias_init=0.0, scope="Linear"):
        self.input_sizes = flatten(input_sizes)
        self.hidden_size = hidden_size
        self.scope=scope

        with tf.variable_scope(self.scope):
            initializer = tf.random_uniform_initializer(-1/math.sqrt(self.hidden_size), 1/math.sqrt(self.hidden_size))


            self.matrices = [tf.get_variable("matrix_%d" % (i,), (ipt_sz, self.hidden_size), initializer=initializer)
                             for i, ipt_sz in enumerate(self.input_sizes)]
            self.bias = tf.get_variable("bias", (self.hidden_size,), initializer=tf.constant_initializer(bias_init))

    def __call__(self, *inputs):
        inputs = flatten(inputs)
        with tf.variable_scope(self.scope):
            summands = [tf.matmul(ipt,weight) for ipt, weight in zip(inputs, self.matrices)]
            summands.append(self.bias)
            return sum(summands)


class GRU(object):
    def __init__(self, input_sizes, hidden_size, scope="GRU", final_nonlinearity=tf.tanh):
        self.input_sizes = flatten(input_sizes)
        self.hidden_size = hidden_size
        self.scope=scope
        self.final_nonlinearity = final_nonlinearity
        with tf.variable_scope(self.scope):
            self.reset_gate = Linear(self.input_sizes + [self.hidden_size], self.hidden_size, bias_init=1.0, scope="reset_gate")
            self.update_gate = Linear(self.input_sizes + [self.hidden_size], self.hidden_size, bias_init=1.0, scope="update_gate")
            self.memory_interpolation = Linear(self.input_sizes + [self.hidden_size], self.hidden_size, scope="memory_interpolation")

    def __call__(self, inputs, state):
        inputs = flatten(inputs)
        with tf.variable_scope(self.scope):
            r = self.reset_gate(inputs + [state])
            u = self.update_gate(inputs + [state])
            r, u = tf.sigmoid(r), tf.sigmoid(u)
            c = self.memory_interpolation(inputs + [r * state])
            c = self.final_nonlinearity(c)
            return u * state + (1-u) * c

    def zero_state(self, batch_size):
        zeros = tf.zeros(tf.pack([batch_size, self.hidden_size]))
        zeros.set_shape([None, self.hidden_size])
        return zeros


class SentenceParser(object):
    def __init__(self, embedding, rnn_cell):
        self.embedding = embedding
        self.rnn_cell  = rnn_cell

    def step_fun(self, prev_out, embedding_for_ts):
        return self.rnn_cell(embedding_for_ts, prev_out)

    def hiddens(self, input_idxes):
        "Expects input_idxes to be input_idxes of size TIMESTEPS * BATCH_SIZE"
        # embed input encoded sentences
        embedded_timesteps = self.embedding(input_idxes)
        batch_size = tf.shape(input_idxes)[1]
        initial_state = gru_cell.zero_state(batch_size)
        return  functional_ops.scan(self.step_fun, embedded_timesteps, initializer=initial_state)

    def final_hidden(self, input_idxes):
        rnn_hiddens = self.hiddens(input_idxes)
        # execute our rnn using scan function
        # extract final timestep's hidden
        rnn_hiddens_reverse = tf.reverse(rnn_hiddens, [True,False,False])
        rnn_final_hidden = rnn_hiddens_reverse[0,:,:]
        return rnn_final_hidden


class Classifier(object):
    def __init__(self, input_sizes, num_classes):
        self.decoder = Linear(input_sizes, num_classes, scope="classifier")

    def unscaled_probs(self, inputs):
        return self.decoder(inputs)

    def error(self, inputs, output_onehots):
        errors = tf.nn.softmax_cross_entropy_with_logits(self.unscaled_probs(inputs), output_onehots)
        return tf.reduce_sum(errors, 0)

    def num_correct(self, inputs, output_onehots):
        predicted_answer = tf.arg_max(self.unscaled_probs(inputs), 1)
        correct_answer = tf.arg_max(output_onehots, 1)
        is_correct = tf.equal(predicted_answer, correct_answer)
        num_correct = tf.reduce_sum(tf.cast(is_correct, tf.int32), 0)
        return num_correct
