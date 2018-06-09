import utils
import lenet
import os
import time
import tensorflow as tf
import math

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SESSION_STATE = './sess/msdo/model.ckpt'
GRAPH = './graphs/msdo'

N_CLASSES = 43
N_TRAINS = 32760
N_EVALS = 6449
N_TESTS = 12569
N_EPOCHS = 240
STEP = 40

# LR = (0.03, 0.03, 0.01, 0.01, 0.003, 0.003, 0.003, 0.001)
LR = (0.03, 0.03, 0.01, 0.01, 0.01, 0.003)
BATCH_SIZE = 128
RATE = 0.1

onscreen_format = 'Ep {0:3d}: loss = {1:.6f} accuracy = {2:.6f} time: {3:.3f}s'
log_format = '{0};{1:.6f};{2:.6f};{3:.3f}\n'


class LeNet5:

    def __init__(self):
        self.n_train_batches = math.ceil(N_TRAINS / BATCH_SIZE)
        self.n_valid_batches = math.ceil(N_EVALS / BATCH_SIZE)
        self.training = None
        self.lr = None
        self.train_init = None
        self.val_init = None
        self.test_init = None
        self.loss = None
        self.epoch_loss = None
        self.epoch_accuracy = None
        self.optimize = None
        self.accuracy = None
        self.loss_summary = None
        self.acc_summary = None

    def build_graph(self):

        # Create a boolean placeholder for batch norm or drop out phase
        self.training = tf.placeholder(tf.bool, name='phase')

        # Create a placeholder for learning rate
        self.lr = tf.placeholder(tf.float32, shape=[], name='learningrate')

        # Create placeholders for summarize epoch loss and epoch accuracy
        self.epoch_loss = tf.placeholder(tf.float32, shape=[], name='eloss')
        self.epoch_accuracy = tf.placeholder(tf.float32, shape=[], name='eacc')

        # Create initializer for train, eval and test data
        with tf.name_scope('data'):
            train_data, val_data, test_data = utils.get_tf_dataset(BATCH_SIZE)

            # Create an iterator to iterate through train, val or test set
            iterator = tf.data.Iterator.from_structure(
                train_data.output_types,
                train_data.output_shapes)

            # Get the next batch of data
            img, label = iterator.get_next()
            img = tf.reshape(img, [-1, 32, 32, 1])

            # Create initializer for iterator from the specific dataset
            self.train_init = iterator.make_initializer(train_data)
            self.val_init = iterator.make_initializer(val_data)
            self.test_init = iterator.make_initializer(test_data)

        # Get logits from input images
            logits = lenet.LN_MS_DO(img, N_CLASSES, RATE, self.training)

        # Calculate loss
        with tf.name_scope('loss'):
            entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
                labels=label, logits=logits)
            self.loss = tf.reduce_mean(entropy, name='loss')

        # Define optimize operation
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.lr)
        self.optimize = optimizer.minimize(self.loss)

        # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        # with tf.control_dependencies(update_ops):
        #   optimize = optimizer.minimize(loss)

        # Evaluate accuracy
        with tf.name_scope('eval'):
            preds = tf.nn.softmax(logits)
            correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
            self.accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

        # Create a FileWriter for writing event log
        # with tf.name_scope('summaries'):
        #     tf.summary.scalar('loss', self.loss)
        #     tf.summary.scalar('accuracy', self.accuracy)
        #     self.summary_op = tf.summary.merge_all()
        self.loss_summary = tf.summary.scalar('loss', self.epoch_loss)
        self.acc_summary = tf.summary.scalar('accuracy', self.epoch_accuracy)

    def train_once(self, sess, writer, epoch):
        total_loss = 0

        # Run the train initializer
        sess.run(self.train_init)

        summary = None

        for i in range(self.n_train_batches):

            _, loss_batch = sess.run(
                [self.optimize, self.loss],
                feed_dict={self.training: True,
                           self.lr: LR[int(epoch/STEP)]}
            )

            total_loss += loss_batch

        summary = sess.run(
            self.loss_summary,
            feed_dict={self.epoch_loss: total_loss / self.n_train_batches}
        )
        writer.add_summary(summary, epoch)

        return total_loss / self.n_train_batches

    def eval_once(self, sess, writer, epoch):
        total_correct_preds = 0

        # Run the test or eval initializer
        sess.run(self.val_init)

        summary = None

        for i in range(self.n_valid_batches):

            accuracy_batch = sess.run(
                self.accuracy,
                feed_dict={self.training: False}
            )

            total_correct_preds += accuracy_batch

        summary = sess.run(
            self.acc_summary,
            feed_dict={self.epoch_accuracy: total_correct_preds / N_EVALS}
        )
        writer.add_summary(summary, epoch)
        return total_correct_preds / N_EVALS

    def train(self, restore=False, savesess=True):

        # Create a saver to save and restore sesstion state
        saver = tf.train.Saver()

        with tf.Session() as sess:
            if restore:
                # Use a recorded session
                saver.restore(sess, SESSION_STATE)
            else:
                # Initialize a new graph
                sess.run(tf.global_variables_initializer())

            # Create a writer for tracking loss and accuracy of the model
            writer = tf.summary.FileWriter(GRAPH, tf.get_default_graph())

            for epoch in range(N_EPOCHS):
                start = time.time()

                epoch_loss = self.train_once(sess, writer, epoch)

                epoch_accuracy = self.eval_once(
                    sess, writer, epoch
                )

                info = (epoch,
                        epoch_loss,
                        epoch_accuracy,
                        time.time()-start)

                print(onscreen_format.format(*info))
            if savesess:
                saver.save(sess, SESSION_STATE)

    def test(self, state=SESSION_STATE):
        with tf.Session() as sess:
            total_correct_preds = 0

            # Load the saved session
            tf.train.Saver().restore(sess, state)

            # Run the test initializer
            sess.run(self.test_init)

            try:
                while True:
                    accuracy_batch = sess.run(
                        self.accuracy,
                        feed_dict={self.training: False}
                    )

                    total_correct_preds += accuracy_batch

            except tf.errors.OutOfRangeError:
                pass

        print('Accuracy on test set: {0:.3f}%'.format(
            total_correct_preds / N_TESTS * 100)
        )


if __name__ == '__main__':
    ln = LeNet5()
    ln.build_graph()
    # ln.train(restore=False)
    ln.test()
