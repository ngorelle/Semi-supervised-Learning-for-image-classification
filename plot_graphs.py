import matplotlib.pyplot as plt

"""#Plot the Loss and accuracy graphs"""


# plot the loss
def plot_loss(epochs, loss):
    fig_1 = plt.figure(figsize=(6, 5))
    plt.plot(range(1, epochs + 1), loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train_loss'], loc='upper right')
    plt.show()
    fig_1.savefig('loss.png')


# plot the acuuracy
def plot_accuracy(epochs, accuracy_student, accuracy_teacher=None):
    fig_2 = plt.figure(figsize=(6, 5))
    plt.plot(range(1, epochs + 1), accuracy_student)
    if accuracy_teacher is not None:
        plt.plot(range(1, epochs + 1), accuracy_teacher)
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['student', 'teacher'], loc='lower right')
        plt.show()
        fig_2.savefig('accuracy.png')
    else:
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['val_accuracy'], loc='lower right')
        plt.show()
        fig_2.savefig('accuracy.png')
