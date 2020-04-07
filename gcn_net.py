from loadData import DataSet
import tensorflow as tf
import numpy as np
from utils import *
import time


class gcnNet(object):

    def __init__(self, category='Mnist', num_supports=1, hidden_dim=[], weight_decay=5e-3, lr=0.01, dropout=0.5,
                 history_dir='./', checkpoint_dir='./', logs_dir='./', Train_Epochs=200):
        self.category = category
        self.num_supports = num_supports
        self.hidden_dim = hidden_dim
        self.weight_decay = weight_decay
        self.dropout = dropout

        self.lr = lr
        self.epoch = Train_Epochs
        self.history_dir = history_dir
        self.checkpoint_dir = checkpoint_dir
        self.logs_dir = logs_dir

        self.data = DataSet(category=self.category)
        self.adj, self.features, self.y_train, self.y_val, self.y_test, self.train_mask, self.val_mask, self.test_mask = self.data.Load()
        self.num_features_nonzero = np.shape(self.features[1])
        self.adj = [preprocess_adj(self.adj)]

        self.kernel = [tf.sparse_placeholder(tf.float32) for _ in range(self.num_supports)]
        self.input = tf.sparse_placeholder(tf.float32, shape=tf.constant(self.features[2], dtype=tf.int64))
        self.labels = tf.placeholder(tf.float32, shape=(None, self.y_train.shape[1]))
        self.labels_mask = tf.placeholder(tf.int32)

    def train(self, sess):
        self.build()

        sess.run(tf.global_variables_initializer())

        saver = tf.train.Saver(max_to_keep=10)

        writer = tf.summary.FileWriter(self.logs_dir, sess.graph)
        start = 0
        latest_checkpoint = tf.train.latest_checkpoint(self.checkpoint_dir)

        if latest_checkpoint:
            latest_checkpoint.split('-')
            start = int(latest_checkpoint.split('-')[-1])
            saver.restore(sess, latest_checkpoint)
            print('Loading checkpoint {}.'.format(latest_checkpoint))

        tf.get_default_graph().finalize()

        cost_val = []

        for epoch in range(start + 1, self.epoch):
            t = time.time()
            feed_dict = dict()
            feed_dict.update({self.labels: self.y_train})
            feed_dict.update({self.labels_mask: self.train_mask})
            feed_dict.update({self.input: self.features})
            feed_dict.update({self.kernel[i]: self.adj[i] for i in range(len(self.adj))})

            # Training step
            outs = sess.run([self.opt, self.loss, self.accuracy], feed_dict=feed_dict)

            cost, acc, duration = self.evaluate(sess, self.features, self.adj, self.y_val, self.val_mask)
            cost_val.append(cost)

            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
                  "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
                  "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

        print("Optimization Finished!")

        test_cost, test_acc, test_duration = self.evaluate(sess, self.features, self.adj, self.y_test, self.test_mask)
        print("Test set results:", "cost=", "{:.5f}".format(test_cost),
              "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

    def build(self):
        self.vars = self.build_weights(self.input, self.hidden_dim, self.labels)
        self.output = self.classifier(self.kernel, self.input, self.vars, self.hidden_dim, self.num_features_nonzero, self.dropout)
        self.loss = self.loss_func(self.labels, self.labels_mask, self.output, self.vars, self.weight_decay)

        self.accuracy = self.masked_accuracy(self.output, self.labels, self.labels_mask)

        self.opt = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def build_weights(self, input, hidden_dim, output):
        with tf.variable_scope("gcn"):
            vars = {}
            for i in range(len(hidden_dim) + 1):
                if i == 0:
                    vars['weight_' + str(i)] = glorot([tf.shape(input)[1], hidden_dim[0]], name='weight_' + str(i))
                    continue
                if i == len(hidden_dim):
                    vars['weight_' + str(i)] = glorot([hidden_dim[-1], output.get_shape().as_list()[1]], name='weight_' + str(i))
                    continue
                vars['weight_' + str(i)] = glorot([hidden_dim[i - 1], hidden_dim[i]], name='weight_' + str(i))
            return vars

    def classifier(self, kernel, input, vars, hidden_dim, num_features_nonzero, dropout):
        layers = []
        layers.append(input)
        for i in range(len(hidden_dim) + 1):
            if i == len(hidden_dim):
                output = tf.nn.dropout(layers[i], 1-self.dropout)
                output = tf.matmul(output, vars['weight_' + str(i)])
                output = tf.sparse_tensor_dense_matmul(kernel[0], output)
                return output
            layers[i] = self.sparse_dropout(layers[i], 1 - dropout, num_features_nonzero)
            h = tf.sparse_tensor_dense_matmul(layers[i], vars['weight_' + str(i)])
            h = tf.sparse_tensor_dense_matmul(kernel[0], h)

            h = tf.nn.relu(h)
            layers.append(h)

    def loss_func(self, labels, labels_mask, outputs, vars, weight_decay):
        loss = weight_decay * tf.nn.l2_loss(vars['weight_0'])

        loss_cross = tf.nn.softmax_cross_entropy_with_logits(logits=outputs, labels=labels)
        mask = tf.cast(labels_mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        loss_cross *= mask

        loss += tf.reduce_mean(loss_cross)
        return loss

    def masked_accuracy(self, preds, labels, mask):
        """Accuracy with masking."""
        correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
        accuracy_all = tf.cast(correct_prediction, tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        mask /= tf.reduce_mean(mask)
        accuracy_all *= mask
        return tf.reduce_mean(accuracy_all)

    def sparse_dropout(self, x, keep_prob, noise_shape):
        """Dropout for sparse tensors."""
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(x, dropout_mask)
        return pre_out * (1. / keep_prob)

    def evaluate(self, sess, features, adj, labels, mask):
        t_test = time.time()
        feed_dict = dict()
        feed_dict.update({self.labels: labels})
        feed_dict.update({self.labels_mask: mask})
        feed_dict.update({self.input: features})
        feed_dict.update({self.kernel[i]: adj[i] for i in range(len(adj))})
        outs_val = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
        return outs_val[0], outs_val[1], (time.time() - t_test)
