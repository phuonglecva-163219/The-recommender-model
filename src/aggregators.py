import tensorflow as tf
from abc import abstractmethod

LAYER_IDS = {}


def get_layer_id(layer_name=''):
    if layer_name not in LAYER_IDS:
        LAYER_IDS[layer_name] = 0
        return 0
    else:
        LAYER_IDS[layer_name] += 1
        return LAYER_IDS[layer_name]


class Aggregator(object):
    def __init__(self, batch_size, dim, dropout, act, name):
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_id(layer))
        self.name = name
        self.dropout = dropout
        self.act = act
        self.batch_size = batch_size
        self.dim = dim

    def __call__(self, self_vectors, neighbor_vectors, neighbor_relations):
        outputs = self._call(self_vectors, neighbor_vectors, neighbor_relations)
        return outputs

    @abstractmethod
    def _call(self, self_vectors, neighbor_vectors, neighbor_relations):
        # dimension:
        # self_vectors: [batch_size, -1, dim]
        # neighbor_vectors: [batch_size, -1, n_neighbor, dim]
        # neighbor_relations: [batch_size, -1, n_neighbor, dim]
        # user_embeddings: [batch_size, dim]
        pass

    def _mix_neighbor_vectors(self, neighbor_vectors, neighbor_relations, sel_vectors):
        avg = False
        if not avg:
            # [batch_size, -1, 1, dim]
            sel_vectors = tf.reshape(sel_vectors, [self.batch_size, -1, 1, self.dim])


            # [batch_size, -1, n_neighbor]
            sel_vectors_neighbors_relations = tf.reduce_mean(sel_vectors * neighbor_vectors, axis=-1)
            sel_vectors_neighbors_relations_nomalized = tf.nn.softmax(sel_vectors_neighbors_relations, dim=-1)

            # [batch_size, -1, n_neighbor, 1]
            sel_vectors_neighbors_relations_nomalized = tf.expand_dims(sel_vectors_neighbors_relations_nomalized, axis=-1)
            # [batch_size, -1, dim]
            # neighbors_aggregated = tf.reduce_mean(user_relation_scores_normalized * neighbor_vectors, axis=2)
            neighbors_aggregated_sofmax = tf.nn.softmax(sel_vectors_neighbors_relations_nomalized * neighbor_vectors * neighbor_relations, axis=-1)
            neighbors_aggregated = tf.reduce_mean(neighbors_aggregated_sofmax, axis=2)
            # neighbors_aggregated = tf.reduce_sum(neighbors_aggregated_sofmax, axis = 2)
        else:
            # [batch_size, -1, dim]
            neighbors_aggregated = tf.reduce_mean(neighbor_vectors, axis=2)

        return neighbors_aggregated


class SumAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(SumAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

            self.weights_1 = tf.get_variable(
                shape = [self.dim, self.dim], initializer = tf.contrib.layers.xavier_initializer(), name = 'weights_1')
            self.bias_1 = tf.get_variable(shape = [self.dim], initializer = tf.zeros_initializer(), name = 'bias_1')

            self.weights_2 = tf.get_variable(
                shape = [self.dim, self.dim], initializer = tf.contrib.layers.xavier_initializer(), name = 'weights_2')
            self.bias_2 = tf.get_variable(shape = [self.dim], initializer = tf.zeros_initializer(), name = 'bias_2')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations):
        # [batch_size, -1, dim]
        # neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, self_vectors)

        # [-1, dim]
        output = tf.reshape(self_vectors + neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        out = tf.reshape(self_vectors * neighbors_agg, [-1, self.dim])
        out = tf.nn.dropout(out, keep_prob = 1 - self.dropout)
        out = tf.matmul(out, self.weights) + self.bias
        out = tf.reshape(out, [self.batch_size, -1, self.dim])

        # return self.act(output + out)
        return self.act(output)


class ConcatAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(ConcatAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim * 2, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # [batch_size, -1, dim * 2]
        output = tf.concat([self_vectors, neighbors_agg], axis=-1)

        # [-1, dim * 2]
        output = tf.reshape(output, [-1, self.dim * 2])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)

        # [-1, dim]
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)


class NeighborAggregator(Aggregator):
    def __init__(self, batch_size, dim, dropout=0., act=tf.nn.relu, name=None):
        super(NeighborAggregator, self).__init__(batch_size, dim, dropout, act, name)

        with tf.variable_scope(self.name):
            self.weights = tf.get_variable(
                shape=[self.dim, self.dim], initializer=tf.contrib.layers.xavier_initializer(), name='weights')
            self.bias = tf.get_variable(shape=[self.dim], initializer=tf.zeros_initializer(), name='bias')

    def _call(self, self_vectors, neighbor_vectors, neighbor_relations, user_embeddings):
        # [batch_size, -1, dim]
        neighbors_agg = self._mix_neighbor_vectors(neighbor_vectors, neighbor_relations, user_embeddings)

        # [-1, dim]
        output = tf.reshape(neighbors_agg, [-1, self.dim])
        output = tf.nn.dropout(output, keep_prob=1-self.dropout)
        output = tf.matmul(output, self.weights) + self.bias

        # [batch_size, -1, dim]
        output = tf.reshape(output, [self.batch_size, -1, self.dim])

        return self.act(output)
