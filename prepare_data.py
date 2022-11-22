"""## **Load and prepare a Dataset**
"""
import tensorflow as tf
from icecream import IceCreamDebugger
import numpy as np

ic = IceCreamDebugger("Prepare Data|> ")

# laod and prepare the dataset

# split the dataset
def split_data(number_per_class):
    ic = IceCreamDebugger("Split Data|> ")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    labels = np.unique(y_train)
    indexes = [np.where(y_train == label)[0] for label in labels]
    for index in indexes:
        np.random.shuffle(index)
    indexes = np.array([index[: number_per_class] for index in indexes]).flatten()[:]
    x_labels, y_labels = x_train[indexes], y_train[indexes]
    indexes = np.delete(y_train, indexes)
    x_unlabels, y_unlabels = x_train[indexes], y_train[indexes]

    # scale the images in the dataset to [0,1], add a single dimension and convert the data to float32
    x_labels = tf.image.convert_image_dtype(x_labels, tf.float32)
    x_unlabels = tf.image.convert_image_dtype(x_unlabels, tf.float32)
    x_test = tf.image.convert_image_dtype(x_test, tf.float32)

    # Convert Labels to Integer and reshape the Labels
    y_labels = y_labels.astype("int32").reshape(-1, )
    y_unlabels = y_unlabels.astype("int32").reshape(-1, )
    y_test = y_test.astype("int32").reshape(-1, )

    ic(x_labels.shape)
    ic(y_labels.shape)
    ic(x_unlabels.shape)
    ic(y_unlabels.shape)

    return (x_labels, y_labels), (x_unlabels, y_unlabels), (x_test, y_test)

# generate data
class DataGen(tf.keras.utils.Sequence):

    def __init__(self, x, y, batch_size, shuffle=True, with_label=True):
        self.x = x
        self.y = y
        self.with_label = with_label
        self.n = len(x)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.index: int = 0
        self.length: int = None

    def on_epoch_end(self):
        if self.shuffle:
            indexes = np.random.choice(np.arange(self.n), self.n, replace=False)
            self.x, self.y = self.x[indexes], self.y[indexes]

    def __getitem__(self, index):
        i = index * self.batch_size
        x, y = self.x[i: i + self.batch_size], self.y[i: i + self.batch_size]

        if index + 1 == self.__len__():
            indexes = np.random.choice(np.arange(self.n), self.n, replace=False)
            self.x, self.y = tf.gather(self.x, indexes), tf.gather(self.y, indexes)

        return (x, y) if self.with_label else x

    def get_next(self):
        self.index = (self.index + 1) % self.__len__()
        return self.__getitem__(self.index)

    def __len__(self):
        if self.length is None:
            self.length = self.n // self.batch_size
        return self.length


if __name__ == "__main__":
    (x_labels, y_labels), (x_unlabels, _), (_, _) = split_data(400)
    ic(x_labels.shape)
    ic(y_labels.shape)
    ic(x_unlabels.shape)
    generator = DataGen(x_labels, y_labels, 400)
    for i in range(2):
        for (batchx, batchy) in generator:
            print(batchy.shape, batchx.shape)
        print()
    # ic(y_train.shape)
