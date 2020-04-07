# _author_ = "jiali cui"
# Email : cuijiali961224@gmail.com
# Data:

from gcn_net import gcnNet
import tensorflow as tf
import os
import shutil


FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string('category', 'cora', 'DataSet category')

tf.flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
tf.flags.DEFINE_list('hidden_dim', [16,], 'hidden layers')
tf.flags.DEFINE_integer('Train_Epochs', 200, 'Number of Epochs to train')
tf.flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
tf.flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
tf.flags.DEFINE_float('lr', 0.01, 'learning rate')



tf.flags.DEFINE_string('history_dir', './output/history/', 'history')
tf.flags.DEFINE_string('checkpoint_dir', './output/checkpoint/', 'checkpoint')
tf.flags.DEFINE_string('logs_dir', './output/logs/', 'logs')

def main(_):

    model = gcnNet(
        category=FLAGS.category,
        Train_Epochs=FLAGS.Train_Epochs,
        lr=FLAGS.lr,
        history_dir=FLAGS.history_dir,
        checkpoint_dir=FLAGS.checkpoint_dir,
        logs_dir=FLAGS.logs_dir,
        weight_decay=FLAGS.weight_decay,
        dropout=FLAGS.dropout,
        hidden_dim=FLAGS.hidden_dim
    )

    continueTrain = False
    # continueTrain = True
    with tf.Session() as sess:
        if not continueTrain:
            if os.path.exists(FLAGS.checkpoint_dir):
                shutil.rmtree(FLAGS.checkpoint_dir[:-1])
            os.makedirs(FLAGS.checkpoint_dir)

        if os.path.exists(FLAGS.logs_dir):
            shutil.rmtree(FLAGS.logs_dir[:-1])
        os.makedirs(FLAGS.logs_dir)

        if not os.path.exists(FLAGS.history_dir):
            os.makedirs(FLAGS.history_dir)

        model.train(sess)


if __name__ == '__main__':
    tf.app.run()
