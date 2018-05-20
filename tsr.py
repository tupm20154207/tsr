import utils
import lenet
import os
import time
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SESSION_STATE = 'sess/msdo3/model.ckpt'
TRAINING_INFO = 'log/msdo3'

N_CLASSES = 43
N_TRAINS = 32760
N_EVALS = 6449
N_TESTS = 12569
N_EPOCHS = 240
STEP = 40
LR = (0.03, 0.03, 0.01, 0.01, 0.01, 0.003, 0.003, 0.001)
BATCH_SIZE = 128
RATE = 0.1

onscreen_format = 'Ep {0:3d}: loss = {1:.6f} accuracy = {2:.6f} time: {3:.3f}s'
log_format = '{0};{1:.6f};{2:.6f};{3:.3f}\n'


def my_cnn_fn(mode, restore, savesess, writelog):

    # Create a placeholder for dropout phase
    phase = tf.placeholder(tf.bool, name='phase')

    # Create a placeholder for learning rate

    lr = tf.placeholder(tf.float32, shape=[], name='lr')

    # Create iterator initializer
    with tf.name_scope('data'):
        train_data, val_data, test_data = utils.get_tf_dataset(BATCH_SIZE)

        # Create an iterator to iterate through train, val or test set
        iterator = tf.data.Iterator.from_structure(
            train_data.output_types,
            train_data.output_shapes)

        # For each batch of element of the data set assign it to self.img
        # and self.label
        img, label = iterator.get_next()
        img = tf.reshape(img, [-1, 32, 32, 1])

        # Create initializer for iterator from the specific dataset
        if mode == 'train':
            train_init = iterator.make_initializer(train_data)
            val_init = iterator.make_initializer(val_data)
        else:
            test_init = iterator.make_initializer(test_data)

    # Build graph
    # logits = lenet.LN_SS(img, N_CLASSES)
    # logits = lenet.LN_MS(img, N_CLASSES)
    # logits = lenet.LN_MS_BN(img, phase, N_CLASSES)
    logits = lenet.LN_MS_DO(img, N_CLASSES, RATE, phase)

    # Calculate loss
    with tf.name_scope('loss'):
        entropy = tf.nn.softmax_cross_entropy_with_logits_v2(
            labels=label, logits=logits)
        loss = tf.reduce_mean(entropy, name='loss')

    # Define optimize operation
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=lr)
    optimize = optimizer.minimize(loss)

    # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    # with tf.control_dependencies(update_ops):
    #   optimize = optimizer.minimize(loss)

    # Evaluate accuracy
    with tf.name_scope('eval'):
        preds = tf.nn.softmax(logits)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(label, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))

    # Create a FileWriter for writing event log

    with tf.name_scope('summaries'):
        tf.summary.scalar('loss', loss)
        tf.summary.scalar('accuracy', accuracy)
        merged = tf.summary.merge_all()

    # Create a saver to save session state to binary file

    saver = tf.train.Saver()

    # Start session

    with tf.Session() as sess:

        if restore:
            saver.restore(sess, SESSION_STATE)
            start_epoch = len(open(TRAINING_INFO, 'r').readlines())
        else:
            sess.run(tf.global_variables_initializer())
            start_epoch = 0

        writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())

        if mode == 'train':
            with open(TRAINING_INFO, 'a') as f:
                for epoch in range(N_EPOCHS):
                    start = time.time()

                    epoch_loss = train(
                        sess, train_init, optimize, writer, merged,
                        loss, lr, epoch, phase
                    )
                    epoch_accuracy = eval(
                        sess, val_init, accuracy, writer, merged,
                        N_EVALS, phase
                    )

                    info = (start_epoch + epoch,
                            epoch_loss,
                            epoch_accuracy,
                            time.time()-start)

                    print(onscreen_format.format(*info))
                    if writelog:
                        f.write(log_format.format(*info))

                if savesess and input('Save y/n: ') == 'y':
                    saver.save(sess, SESSION_STATE)
        else:
            acc = eval(sess, test_init, accuracy, N_TESTS, phase)
            print('Accuracy on test set: {0:.3f}%'.format(acc * 100))


def train(sess, train_init, optimize, writer, merged, loss, lr, epoch, phase):
    total_loss = 0
    n_train_batches = 0
    sess.run(train_init)

    try:
        while True:
            _, loss_batch, summary = sess.run(
                [optimize, loss, merged],
                feed_dict={lr: LR[int(epoch/STEP)], phase: True}
            )
            total_loss += loss_batch
            n_train_batches += 1

    except tf.errors.OutOfRangeError:
        pass

    return total_loss / n_train_batches


def eval(sess, init, accuracy, writer, merged, n_samples, phase):
    total_correct_preds = 0
    sess.run(init)
    try:
        while True:
            accuracy_batch = sess.run(accuracy, feed_dict={phase: False})
            total_correct_preds += accuracy_batch

    except tf.errors.OutOfRangeError:
        pass

    return total_correct_preds / n_samples


if __name__ == '__main__':
    my_cnn_fn('test', True, True, True)
    # my_cnn_fn('train', False, True, True)
