import tensorflow as tf
import math

from tensorflow.python.ops import functional_ops # for scan!

from .utils import flatten

class Embedding(object):
    def __init__(self, vocab_size, embedding_size, embedding_matrix=None):
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        if embedding_matrix is None:
            embedding_matrix_initializer = tf.random_uniform_initializer(-1/math.sqrt(embedding_size), 1/math.sqrt(embedding_size))
            self.embedding_matrix = tf.get_variable("embedding_matrix", (vocab_size, embedding_size),
                                           initializer=embedding_matrix_initializer)
        else:
            self.embedding_matrix = embedding_matrix

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

class StackedGRU(object):
    def __init__(self, input_sizes, hidden_sizes, final_nonlinearity, scope="stacked_gru"):
        self.input_sizes = flatten(input_sizes)
        self.hidden_sizes = flatten(hidden_sizes)
        self.scope = scope
        with tf.variable_scope(self.scope):
            self.grus = []
            self.grus.append(GRU(self.input_sizes, self.hidden_sizes[0], final_nonlinearity=final_nonlinearity, scope="gru_0"))
            for i, (in_size, out_size) in enumerate(zip(self.hidden_sizes[:-1], self.hidden_sizes[1:])):
                self.grus.append(GRU(in_size, out_size, final_nonlinearity=final_nonlinearity, scope="gru_%d" % (i+1,)))

    def __call__(self, inputs, state):
        new_state = []
        level_input = flatten(inputs)
        with tf.variable_scope(self.scope):
            offset = 0
            for i, gru in enumerate(self.grus):
                level_state = tf.slice(state, [0, offset], [-1, self.hidden_sizes[i]])
                level_output = gru(level_input, level_state)
                new_state.append(level_output)
                level_input = level_output
                offset += self.hidden_sizes[i]
            return tf.concat(1, new_state)

    def zero_state(self, batch_size):
        with tf.variable_scope(self.scope):
            return tf.concat(1, list(gru.zero_state(batch_size) for gru in self.grus))

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
        initial_state = self.rnn_cell.zero_state(batch_size)
        return  functional_ops.scan(self.step_fun, embedded_timesteps, initializer=initial_state)

    def final_hidden(self, input_idxes):
        rnn_hiddens = self.hiddens(input_idxes)
        # execute our rnn using scan function
        # extract final timestep's hidden
        rnn_hiddens_reverse = tf.reverse(rnn_hiddens, [True,False,False])
        rnn_final_hidden = rnn_hiddens_reverse[0,:,:]
        return rnn_final_hidden

class BidirectionalSentenceParser(object):
    def __init__(self, embedding, rnn_forward, rnn_backward):
        self.parser_forward  = SentenceParser(embedding, rnn_forward)
        self.parser_backward = SentenceParser(embedding, rnn_backward)

    def final_hidden(self, input_idxes):
        sentence_hidden_forward  = self.parser_forward.final_hidden(input_idxes)
        input_idxes_reversed = tf.reverse(input_idxes, [True, False])
        sentence_hidden_backward = self.parser_backward.final_hidden(input_idxes_reversed)
        return tf.concat(1, [sentence_hidden_forward, sentence_hidden_backward])

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
