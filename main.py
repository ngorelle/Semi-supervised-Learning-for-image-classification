import os
import sys
import numpy as np

root = os.path.dirname(os.getcwd())
for directory in os.listdir(root):
    sys.path.append(os.path.join(root, directory))
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
import matplotlib.pyplot as plt
from collections import defaultdict
from icecream import ic, IceCreamDebugger
from config import HyperparameterArgs
from plot_graphs import plot_loss, plot_accuracy
from dataset.prepare_data import DataGen, split_data
from mean_teacher.model import Model, train_step, evaluation, ema

TIME_FORMAT = "%Y-%m-%d--%H-%M-%S"
WEIGHTS_DIR = "experiments/weights"
RESULTS_DIR = "experiments/results"
CONFIG_DIR = "experiments/config"

# Make directory
os.makedirs(WEIGHTS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CONFIG_DIR, exist_ok=True)


def train(hp: HyperparameterArgs):
    seed = 12345
    tf.random.set_seed(seed)
    training_id = datetime.now().strftime(TIME_FORMAT) + f"{'_supervised' if hp.training == 'supervised' else ''}"

    # get data per class
    (x_labels, y_labels), (x_unlabels, y_unlabels), (x_test, y_test) = split_data(hp.image_per_class)

    # generate train data
    train_ds_labeled = DataGen(x_labels, y_labels, hp.batch_size_labels)
    train_ds_unlabeled = DataGen(x_unlabels, y_unlabels, hp.batch_size_unlabels, with_label=False)
    # generate test data
    test_ds = DataGen(x_test, y_test, hp.batch_size)

    # Create two instances of the model
    student_model = Model(hp, os.path.join(WEIGHTS_DIR, training_id + "_student.h5"))
    teacher_model = Model(hp, os.path.join(WEIGHTS_DIR, training_id + "_teacher.h5"))
    # perform a simple test of the two model
    m = tf.ones((1, 32, 32, 3))
    _ = student_model(m)
    _ = teacher_model(m)
    # Initialize the trainable variables of teacher
    for i in range(len(student_model.trainable_variables)):
        teacher_model.trainable_variables[i].assign(student_model.trainable_variables[i])

    step = hp.global_step
    log_results = defaultdict(list)
    for epoch in tqdm(range(hp.epoch)):
        total_loss = []
        consistency_loss = []
        classification_loss = []

        for _, (x, y) in enumerate(train_ds_labeled, 1):
            step += 1
            if hp.training == "supervised":
                loss = train_step(student_model, x, y)
            else:
                if step < hp.warm_up_steps:
                    loss = train_step(student_model, x, y)
                else:
                    # get unlabeled data
                    x_unlabeled = train_ds_unlabeled.get_next()
                    # Pass it through teacher network to get pseudo-label
                    y_unlabeled = teacher_model(x_unlabeled)
                    loss = train_step(student_model, x, y, x_unlabeled, y_unlabeled)
                # update teacher_model at each training_step
                teacher_model = ema(student_model, teacher_model, alpha=hp.ema_consistency)

            total_loss.append(loss[0])
            consistency_loss.append(loss[2])
            classification_loss.append(loss[1])
            hp.global_step += step
            # print(hp.global_step)

        # evaluate the student_model
        acc_test = evaluation(student_model, test_ds)
        acc_train = evaluation(student_model, train_ds_labeled)
        classification_loss = tf.reduce_mean(classification_loss).numpy()

        # save the model weights
        student_model.save_weights_models(acc_test)

        log_results["student_acc_test"].append(acc_test)
        log_results["student_acc_train"].append(acc_train)
        log_results["student_loss"].append(classification_loss)
        log_msg = f"Epoch: {epoch + 1} Step: {step} TrA: {acc_train:.4f} TeA: {acc_test:.4f} "
        log_msg += f"CLF_L: {classification_loss:.4f} "

        # Train with unlabeled data
        if hp.training != "supervised":
            # evaluate the teacher_model
            acc_test = evaluation(teacher_model, test_ds)
            acc_train = evaluation(teacher_model, train_ds_labeled)
            log_results["teacher_acc_test"].append(acc_test)
            log_results["teacher_acc_train"].append(acc_train)
            total_loss = tf.reduce_mean(total_loss).numpy()
            consistency_loss = tf.reduce_mean(consistency_loss).numpy()
            log_results["teacher_loss"].append(consistency_loss)
            log_results['total_loss'].append(total_loss)
            # save the model weights
            teacher_model.save_weights_models(acc_test)

            log_msg += f"TTrA: {acc_train:.4f} TTeA: {acc_test:.4f} ConsL: {consistency_loss: .4f} TTL: {total_loss: .4f}"

        print(log_msg)

    df = pd.DataFrame.from_dict(log_results)
    df.to_csv(os.path.join(RESULTS_DIR, training_id + '.csv'), index=False)

    with open(os.path.join(CONFIG_DIR, training_id + ".txt"), "w") as file:
        for k, v in vars(hp).items():
            file.write(f"{k}: {v} \n")



if __name__ == '__main__':
    hp = HyperparameterArgs()
    # hp.epoch = 2
    train(hp)
    # Supervised
    # hp.training = "supervised"
    # for epoch, image_per_class in zip([100, 150], [100, 200]):
    #     hp.epoch = epoch
    #     hp.image_per_class = image_per_class
    #     for translation in [True]:
    #         hp.translation = translation
    #         for flip in [True]:
    #             hp.flip = flip
    #             for lr in [1e-3]:
    #                 hp.lr = lr
    #                 train(hp)

    # hp = HyperparameterArgs()
    # # Semi-Supervised
    # hp.training = "semi-supervised"
    # for epoch in [5]:
    #     hp.epoch = epoch
    #     for batch_size in [100]:
    #         hp.batch_size_unlabels = batch_size
    #         for image_per_class in [100]:
    #             hp.image_per_class = image_per_class
    #             for translation in [True]:
    #                 hp.translation = translation
    #                 for gaussian in [True]:
    #                     hp.gaussian = gaussian
    #                     for lr in [1e-3]:
    #                         hp.lr = lr
    #                         train(hp)
