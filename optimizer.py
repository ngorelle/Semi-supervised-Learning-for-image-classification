import tensorflow as tf

from config import HyperparameterArgs

hp = HyperparameterArgs()


def adam_optimizer(cost, global_step, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, name=None):
    with tf.name_scope(name, "adam_optimizer") as scope:
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate,
                                             beta1=beta1,
                                             beta2=beta2,
                                             epsilon=epsilon)
        return optimizer.minimize(cost, global_step=global_step, name=scope)


def step_rampup(global_step, rampup_length):
    if global_step < rampup_length:
        result = tf.constant(0.0)
    else:
        result = tf.constant(1.0)
    return tf.identity(result, name="step_rampup")


def sigmoid_rampup(global_step, rampup_length):
    global_step = tf.cast(global_step, tf.float32)
    rampup_length = tf.cast(rampup_length, tf.float32)

    def ramp():
        phase = 1.0 - tf.maximum(0.0, global_step) / rampup_length
        return tf.exp(-5.0 * phase * phase)

    result = tf.cond(global_step < rampup_length, ramp, lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampup")


def sigmoid_rampdown(global_step, rampdown_length, training_length):
    global_step = tf.cast(global_step, dtype=tf.float32)
    rampdown_length = tf.cast(rampdown_length, dtype=tf.float32)
    training_length = tf.cast(training_length, dtype=tf.float32)

    def ramp():
        phase = 1.0 - tf.maximum(0.0, training_length - global_step) / rampdown_length
        return tf.exp(-12.5 * phase * phase)

    result = tf.cond(global_step >= training_length - rampdown_length,
                     ramp,
                     lambda: tf.constant(1.0))
    return tf.identity(result, name="sigmoid_rampdown")
