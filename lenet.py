import tensorflow as tf


def LN_SS(input_layer, training, n_classes):
    """Apply Lenet5 Single-scale"""
    conv1 = conv_rel(input_layer, 6, 'conv1')  # 28x28x6
    pool1 = max_pool(conv1, 'pool1')  # 14x14x6

    conv2 = conv_rel(pool1, 16, 'conv2')  # 10x10x16
    pool2 = max_pool(conv2, 'pool2')  # 5x5x16

    pool2_flat = tf.reshape(pool2, [-1, 400])
    fc1 = fc_rel(pool2_flat, 120, 'fc1')
    fc2 = fc_rel(fc1, 84, 'fc2')
    logits = tf.layers.dense(fc2, n_classes, name='logits')

    return logits


def LN_MS(input_layer, n_classes):
    """Apply Lenet5 Multi-scale"""
    conv1 = conv_rel(input_layer, 6, 'conv1')  # 28x28x6
    pool1 = max_pool(conv1, 'pool1')  # 14x14x6

    conv2 = conv_rel(pool1, 16, 'conv2')  # 10x10x16
    pool2 = max_pool(conv2, 'pool2')  # 5x5x16

    pool1_flat = tf.reshape(pool1, [-1, 14 * 14 * 6])
    pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])

    fc1 = fc_rel(tf.concat([pool1_flat, pool2_flat], axis=1), 120, 'fc1')

    fc2 = fc_rel(fc1, 84, 'fc2')

    logits = tf.layers.dense(fc2, n_classes, name='logits')

    return logits


def LN_MS_BN(input_layer, phase, n_classes):
    """Apply Lenet5 Multi-scale with a batchnorm layer AFTER activation of
    each convolution layer"""
    # phase is a boolean placeholder that need to be fed during train/test time

    conv1 = conv_rel(input_layer, 6, 'conv1')
    pool1 = max_pool(conv1, name='pool1')
    bn1 = tf.layers.batch_normalization(
        inputs=pool1,
        training=phase,
        name='bn1'
    )

    conv2 = conv_rel(bn1, 16, 'conv2')
    pool2 = max_pool(conv2, name='pool2')
    bn2 = tf.layers.batch_normalization(
        inputs=pool2,
        training=phase,
        name='bn2'
    )

    bn1_flat = tf.reshape(bn1, [-1, 14 * 14 * 6])
    bn2_flat = tf.reshape(bn2, [-1, 5 * 5 * 16])

    fc1 = fc_rel(tf.concat([bn1_flat, bn2_flat], axis=1), 120, 'fc1')

    fc2 = fc_rel(fc1, 84, 'fc2')

    logits = tf.layers.dense(fc2, n_classes, name='logits')

    return logits


def LN_MS_BN2(input_layer, phase, n_classes):
    """Apply Lenet5 Multi-scale with a batchnorm layer BEFORE activation of
    each convolution layer"""
    # phase is a boolean placeholder that need to be fed during train/test time

    conv1 = tf.layers.conv2d(
        inputs=input_layer,
        strides=1,
        filters=6,
        kernel_size=[5, 5],
        padding='valid',
        name='conv1'
    )
    bn1 = tf.layers.batch_normalization(
        inputs=conv1,
        training=phase,
        name='bn1'
    )
    pool1 = max_pool(tf.nn.relu(bn1), name='pool1')

    conv2 = tf.layers.conv2d(
        inputs=pool1,
        strides=1,
        filters=16,
        kernel_size=[5, 5],
        padding='valid',
        name='conv2'
    )
    bn2 = tf.layers.batch_normalization(
        inputs=conv2,
        training=phase,
        name='bn2'
    )
    pool2 = max_pool(tf.nn.relu(bn2), name='pool2')

    pool1_flat = tf.reshape(pool1, [-1, 14 * 14 * 6])
    pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])

    fc1 = fc_rel(tf.concat([pool1_flat, pool2_flat], axis=1), 120, 'fc1')

    fc2 = fc_rel(fc1, 84, 'fc2')

    logits = tf.layers.dense(fc2, n_classes, name='logits')

    return logits


def conv_rel(input_tensor, n_filters, name):
    return tf.layers.conv2d(
        inputs=input_tensor,
        strides=1,
        filters=n_filters,
        kernel_size=[5, 5],
        padding="valid",
        activation=tf.nn.relu,
        name=name
    )


def max_pool(input_tensor, name):
    return tf.layers.max_pooling2d(
        input_tensor,
        pool_size=[2, 2],
        strides=2,
        name=name
    )


def fc_rel(input_tensor, n_units, name):
    return tf.layers.dense(
        input_tensor,
        units=n_units,
        activation=tf.nn.relu,
        name=name
    )
