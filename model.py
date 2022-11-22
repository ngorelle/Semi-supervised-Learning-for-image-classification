# -*- coding: utf-8 -*-
# Mean teacher

import tensorflow as tf
import tensorflow_addons as tfa
from mean_teacher.optimizer import step_rampup, sigmoid_rampup, sigmoid_rampdown
from config import HyperparameterArgs

hp = HyperparameterArgs()
mse = tf.keras.losses.MeanSquaredError()

with tf.name_scope("ramps"):
    sigmoid_rampup_value = sigmoid_rampup(hp.global_step, hp.rampup_length)
    sigmoid_rampdown_value = sigmoid_rampdown(hp.global_step,
                                              hp.rampdown_length,
                                              hp.training_length)
    learning_rate = tf.multiply(sigmoid_rampup_value * sigmoid_rampdown_value,
                                hp.max_learning_rate,
                                name='learning_rate')
    adam_beta_1 = tf.add(sigmoid_rampdown_value * hp.adam_beta_1_before_rampdown,
                         (1 - sigmoid_rampdown_value) * hp.adam_beta_1_after_rampdown,
                         name='adam_beta_1')

    cons_coefficient = tf.multiply(sigmoid_rampup_value,
                                   hp.max_consistency_cost,
                                   name='consistency_coefficient')

    step_rampup_value = step_rampup(hp.global_step, hp.rampup_length)
    adam_beta_2 = tf.add((1 - step_rampup_value) * hp.adam_beta_2_during_rampup,
                         step_rampup_value * hp.adam_beta_2_after_rampup,
                         name='adam_beta_2')


    ema_decay = tf.add((1 - step_rampup_value) * hp.ema_decay_during_rampup,
                       step_rampup_value * hp.ema_decay_after_rampup,
                       name='ema_decay')



optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=adam_beta_1, beta_2=adam_beta_2,
                                     epsilon=1e-8)


# Model implementation
class Model(tf.keras.Model):
    def __init__(self, hp: HyperparameterArgs, path_best_model_weights):
        super(Model, self).__init__()
        self.hp = hp
        self.best_acc = float("-inf")
        self.path_best_model_weights = path_best_model_weights
        # Block1
        self.normalize1 = tf.keras.layers.BatchNormalization()

        self.random_flip = tf.keras.layers.RandomFlip(mode='horizontal', seed=None, name='random_flip')
        self.random_translate = tf.keras.layers.RandomTranslation(0.2, 0.2, name='random_translate')
        self.gaussian_noise = tf.keras.layers.GaussianNoise(0.15, seed=None, name='gaussian_noise')

        # First Convolutional Layer
        # Block2
        self.conv1_1 = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same'))
        self.normalize2_1 = tf.keras.layers.BatchNormalization()
        self.conv1_2 = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same'))
        self.normalize2_2 = tf.keras.layers.BatchNormalization()
        self.conv1_3 = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv2D(filters=128, kernel_size=3, padding='same'))
        self.normalize2_3 = tf.keras.layers.BatchNormalization()
        self.pool1 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.dropout1 = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None, name='dropout1')

        # Block3
        self.conv2_1 = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same'))
        self.normalize3_1 = tf.keras.layers.BatchNormalization()
        self.conv2_2 = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same'))
        self.normalize3_2 = tf.keras.layers.BatchNormalization()
        self.conv2_3 = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv2D(filters=256, kernel_size=3, padding='same'))
        self.normalize3_3 = tf.keras.layers.BatchNormalization()
        self.pool2 = tf.keras.layers.MaxPool2D(pool_size=(2, 2))
        self.dropout2 = tf.keras.layers.Dropout(0.5, noise_shape=None, seed=None, name='dropout2')

        # Block4
        self.conv3_1 = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv2D(filters=512, kernel_size=3, padding='valid'))
        self.normalize4_1 = tf.keras.layers.BatchNormalization()
        self.conv3_2 = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv2D(filters=256, kernel_size=1, padding='same'))
        self.normalize4_2 = tf.keras.layers.BatchNormalization()
        self.conv3_3 = tfa.layers.WeightNormalization(
            tf.keras.layers.Conv2D(filters=128, kernel_size=1, padding='same'))
        self.normalize4_3 = tf.keras.layers.BatchNormalization()
        self.pool3 = tf.keras.layers.AveragePooling2D(pool_size=(6, 6))

        self.flatten = tf.keras.layers.Flatten()

        # dropout
        self.normalize5_1 = tf.keras.layers.BatchNormalization()
        self.linear1 = tf.keras.layers.Dense(128)
        self.normalize5_2 = tf.keras.layers.BatchNormalization()
        self.linear2 = tf.keras.layers.Dense(10)

    def call(self, x):
        if self.hp.aug:
            if self.hp.translation:
                x = self.random_translate(x)
            if self.hp.flip:
                x = self.random_flip(x)
            if self.hp.gaussian:
                x = self.gaussian_noise(x)
        # print("Enter: ", x.shape)
        x = self.conv1_1(x)
        x = lrelu(x)
        x = self.normalize2_1(x)
        x = lrelu(x)
        x = self.conv1_2(x)
        x = lrelu(x)
        x = self.normalize2_2(x)
        x = self.conv1_3(x)
        x = lrelu(x)
        x = self.normalize2_3(x)
        x = self.pool1(x)
        x = self.dropout1(x)

        # print("Block 1:", x.shape)
        x = self.conv2_1(x)
        x = lrelu(x)
        x = self.normalize3_1(x)
        x = self.conv2_2(x)
        x = lrelu(x)
        x = self.normalize3_2(x)
        x = self.conv2_3(x)
        x = lrelu(x)
        x = self.normalize3_3(x)

        x = self.pool2(x)
        x = self.dropout2(x)
        # print("Block 2:", x.shape)
        x = self.conv3_1(x)
        x = lrelu(x)
        x = self.normalize4_1(x)
        x = self.conv3_2(x)
        x = lrelu(x)
        x = self.normalize4_2(x)
        x = self.conv3_3(x)
        x = lrelu(x)
        x = self.normalize4_3(x)

        x = self.pool3(x)
        # print("Block 3:", x.shape)
        x = self.flatten(x)
        x = self.normalize5_1(x)
        x = self.linear1(x)
        x = lrelu(x)
        x = self.normalize5_2(x)
        x = self.linear2(x)
        # print("Block 4:", x.shape)
        return x

    def save_weights_models(self, acc):
        # print("Model:", acc, self.best_acc, self.path_best_model_weights)
        if acc > self.best_acc:
            # print("enter")
            self.best_acc = acc
            try:
                self.save_weights(self.path_best_model_weights, save_format="h5")
            except:
                ...

    def load_weights_models(self):
        # print(self.get_weights())
        try:
            self.load_weights(self.path_best_model_weights)
        except:
            ...


def lrelu(x):
    x = tf.keras.layers.LeakyReLU(alpha=0.1)(x)
    return x


# update the teacher's weights with the ema function
def ema(student_model, teacher_model, alpha):
    # formular: weights[i] = alpha * weights_1[i] + (1 - alpha)* weight_2[i]

    for i in range(len(student_model.trainable_variables)):
        weights = alpha * teacher_model.trainable_variables[i] + (1 - alpha) * student_model.trainable_variables[i]
        teacher_model.trainable_variables[i].assign(weights)
    return teacher_model


def classification_cost(labels, predictions):
    costs = tf.nn.sparse_softmax_cross_entropy_with_logits(labels, predictions)
    costs = tf.reduce_mean(costs)
    return costs


def consistency_cost(prediction_student, predictions_teacher, cons_coefficient):
    softmax1 = tf.nn.softmax(prediction_student)
    softmax2 = tf.nn.softmax(predictions_teacher)
    costs = tf.reduce_mean((softmax1 - softmax2) ** 2)
    costs = costs * cons_coefficient

    return costs


def overall_cost(classification_cost, consistency_cost, ratio=0.5):
    return (ratio * classification_cost) + ((1 - ratio) * consistency_cost)


# @tf.function
def train_step(model, images, labels, images_ul=None, p_labels=None):
    with tf.GradientTape() as tape:
        predictions = model(images)  # Forward Pass

        consistency_loss = 0

        # Calculate Loss
        if images_ul is not None and p_labels is not None:
            new_predictions = model(images_ul)  # Forward Pass
            consistency_loss = consistency_cost(p_labels, new_predictions, cons_coefficient)

        classification_loss = classification_cost(labels, predictions)
        overall_loss = classification_loss + consistency_loss
    gradients = tape.gradient(overall_loss, model.trainable_variables)  # Calculate Gradients (Backward Pass)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))  # Apply Gradients

    return [overall_loss, classification_loss, consistency_loss]


def evaluation(model, dataset):
    val_accuracy = tf.keras.metrics.Accuracy()
    val_accuracy.reset_states()
    for x, y in iter(dataset):
        predictions = model(x)
        predictions = tf.nn.softmax(predictions)
        predictions = tf.argmax(predictions, axis=-1)
        val_accuracy.update_state(y, predictions)
    return val_accuracy.result().numpy()
