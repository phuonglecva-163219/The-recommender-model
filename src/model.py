import tensorflow as tf
from aggregators import SumAggregator, ConcatAggregator, NeighborAggregator
from sklearn.metrics import f1_score, roc_auc_score, roc_curve, auc
import numpy as np
# import matplotlib.pyplot as plt
class MyModel(object):
    def __init__(self, args, n_user, n_entity, n_relation, adj_entity, adj_relation, user_hist_dict, max_entity, item_hist_dict, adj_entity_exp_score):
        self.item_hist_dict = item_hist_dict
        self.adj_entity_exp_score = adj_entity_exp_score
        self._parse_args(args, adj_entity, adj_relation, user_hist_dict, max_entity)
        self._build_inputs()
        self._build_model(n_user, n_entity, n_relation)
        self._build_train()

    @staticmethod
    def get_initializer():
        return tf.contrib.layers.xavier_initializer()

    def _parse_args(self, args, adj_entity, adj_relation, user_hist_dict, max_entity):
        # [entity_num, neighbor_sample_size]
        self.max_entity = max_entity
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        self.user_hist_dict = user_hist_dict
        self.n_iter = args.n_iter
        self.batch_size = args.batch_size
        self.n_neighbor = args.neighbor_sample_size
        self.dim = args.dim
        self.l2_weight = args.l2_weight
        self.lr = args.lr
        if args.aggregator == 'sum':
            self.aggregator_class = SumAggregator
        elif args.aggregator == 'concat':
            self.aggregator_class = ConcatAggregator
        elif args.aggregator == 'neighbor':
            self.aggregator_class = NeighborAggregator
        else:
            raise Exception("Unknown aggregator: " + args.aggregator)

    def _build_inputs(self):
        self.user_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='user_indices')
        self.item_indices = tf.placeholder(dtype=tf.int64, shape=[None], name='item_indices')
        self.labels = tf.placeholder(dtype=tf.float32, shape=[None], name='labels')
    def _build_model(self, n_user, n_entity, n_relation):
        self.user_emb_matrix = tf.get_variable(
            shape=[n_user , self.dim], initializer=MyModel.get_initializer(), name='user_emb_matrix')
        self.entity_emb_matrix = tf.get_variable(
            shape=[n_entity + n_user + 1, self.dim], initializer=MyModel.get_initializer(), name='entity_emb_matrix')
        self.relation_emb_matrix = tf.get_variable(
            shape=[n_relation + 3, self.dim], initializer=MyModel.get_initializer(), name='relation_emb_matrix')


        # [batch_size, ]
        self.items_score = tf.reshape(tf.gather(self.item_hist_dict, self.item_indices), [self.batch_size, -1])
        # self.items_score = tf.reshape(tf.gather(self.adj_entity_exp_score, self.item_indices), [self.batch_size, -1])

        # items_score = [score for item, score in self.item_hist_dict.items() if item in ]
        # [batch_size, dim]
        # max_entity = 9365
        # max_entity = 102568
        # self.old_user_indices = self.user_indices - max_entity - 100
        self.user_embeddings_1 = tf.nn.embedding_lookup(self.user_emb_matrix, self.user_indices - self.max_entity - 1)
        self.item_embeddings_1 = tf.nn.embedding_lookup(self.entity_emb_matrix, self.item_indices)
        self.scores_1 = tf.reduce_mean(tf.sigmoid(self.items_score)  * self.user_embeddings_1 * self.item_embeddings_1, axis = 1)

        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items
        # dimensions of entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        entities, relations = self.get_neighbors(self.item_indices)
        # [batch_size, dim]
        self.item_embeddings, self.aggregators = self.aggregate(entities, relations, False)

        entities_users, relations_users = self.get_neighbors(self.user_indices)
        self.user_embeddings, self.aggregators_users = self.aggregate(entities_users, relations_users, True)

        # [batch_size]
        # self.scores = tf.reduce_sum(self.user_embeddings * self.item_embeddings, axis=1)
        # self.user_embeddings += self.user_embeddings_1
        # self.item_embeddings += self.item_embeddings_1
        self.scores = tf.reduce_mean(tf.sigmoid(self.items_score) * self.user_embeddings * self.item_embeddings , axis=1)
        self.scores += self.scores_1
        self.scores_normalized = tf.sigmoid(self.scores)
        # self.scores_normalized = tf.sigmoid(self.scores + self.scores_1)

    def get_neighbors(self, seeds):
        seeds = tf.expand_dims(seeds, axis=1)
        entities = [seeds]
        relations = []
        for i in range(self.n_iter):
            neighbor_entities = tf.reshape(tf.gather(self.adj_entity, entities[i]), [self.batch_size, -1])
            neighbor_relations = tf.reshape(tf.gather(self.adj_relation, entities[i]), [self.batch_size, -1])
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def aggregate(self, entities, relations, is_user):
        aggregators = []  # store all aggregators
        if not is_user:
            entity_vectors = [tf.nn.embedding_lookup(self.entity_emb_matrix, i) for i in entities]
            relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]
        else:
            res = []
            for j in range(len(entities)):
                if j == 0:
                    res.append(tf.nn.embedding_lookup(self.user_emb_matrix, entities[j] - self.max_entity - 1))
                else:
                    res.append(tf.nn.embedding_lookup(self.entity_emb_matrix, entities[j]))
            entity_vectors = res
            relation_vectors = [tf.nn.embedding_lookup(self.relation_emb_matrix, i) for i in relations]

        for i in range(self.n_iter):
            if i == self.n_iter - 1:
                aggregator = self.aggregator_class(self.batch_size, self.dim, act=tf.nn.leaky_relu)
            else:
                aggregator = self.aggregator_class(self.batch_size, self.dim)
            aggregators.append(aggregator)

            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = [self.batch_size, -1, self.n_neighbor, self.dim]
                vector = aggregator(self_vectors=entity_vectors[hop],
                                    neighbor_vectors=tf.reshape(entity_vectors[hop + 1], shape),
                                    neighbor_relations=tf.reshape(relation_vectors[hop], shape),
                                    )
                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        res = tf.reshape(entity_vectors[0], [self.batch_size, self.dim])

        return res, aggregators

    def _build_train(self):
        self.base_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=self.labels, logits=self.scores))

        self.l2_loss = tf.nn.l2_loss(self.user_emb_matrix) + tf.nn.l2_loss(
            self.entity_emb_matrix) + tf.nn.l2_loss(self.relation_emb_matrix)
        for aggregator in self.aggregators:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)

        for aggregator in self.aggregators_users:
            self.l2_loss = self.l2_loss + tf.nn.l2_loss(aggregator.weights)


        self.loss = self.base_loss + self.l2_weight * self.l2_loss

        self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def train(self, sess, feed_dict):
        return sess.run([self.optimizer, self.loss], feed_dict)

    def eval(self, sess, feed_dict):
        labels, scores = sess.run([self.labels, self.scores_normalized], feed_dict)
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)

        accuracy = [i for i in range(len(labels)) if labels[i] == scores[i]]
        accuracy = len(accuracy) / len(scores)
        return auc, f1, accuracy

    def get_scores(self, sess, feed_dict):
        return sess.run([self.item_indices, self.scores_normalized], feed_dict)
