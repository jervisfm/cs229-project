import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import keras
import numpy as np

class TrainingPlot(keras.callbacks.Callback):
    """
    Based on code from
    https://github.com/kapil-varshney/utilities/blob/master/training_plot/trainingplot.py
    """

    def __init__(self, filename='output/training_plot.jpg'):
        self.filename = filename

    # This function is called when the training begins
    def on_train_begin(self, logs={}):
        # Initialize the lists for holding the logs, losses and accuracies
        self.losses = []
        self.acc = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    def normalize_values(self, x):
        max_value = max(x)
        scaled_x = []
        for item in enumerate(x):
            scaled_x.append(item/max_value)
        return scaled_x

    # This function is called at the end of each epoch
    def on_epoch_end(self, epoch, logs={}):

        # Append the logs, losses and accuracies to the lists
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))
        self.val_losses.append(logs.get('val_loss'))
        self.val_acc.append(logs.get('val_acc'))

        # Before plotting ensure at least 2 epochs have passed
        if len(self.losses) > 1:

            N = np.arange(0, len(self.losses))

            # You can chose the style of your preference
            #print(plt.style.available)
            plt.style.use("bmh")

            # Plot train loss, train acc, val loss and val acc against epochs passed
            plt.figure()
            plt.plot(N, self.losses, label = "train_loss")
            plt.plot(N, self.acc, label = "train_acc")
            plt.plot(N, self.val_losses, label = "val_loss")
            plt.plot(N, self.val_acc, label = "val_acc")
            plt.title("Training Loss and Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig(self.filename + "_loss_accuracy.png")
            plt.close()

            # Plot only accuracies.
            plt.figure()
            plt.plot(N, self.acc, label="train_acc")
            plt.plot(N, self.val_acc, label="val_acc")
            plt.title("Training and Validation Accuracy [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Accuracy")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig(self.filename + "_accuracy.png")
            plt.close()

            # Plot ony loss.
            plt.figure()
            plt.plot(N, self.losses, label="train_loss")
            plt.plot(N, self.val_losses, label="val_loss")
            plt.title("Training and Validation Loss [Epoch {}]".format(epoch))
            plt.xlabel("Epoch #")
            plt.ylabel("Loss")
            plt.legend()
            # Make sure there exists a folder called output in the current directory
            # or replace 'output' with whatever direcory you want to put in the plots
            plt.savefig(self.filename + "_loss.png")
            plt.close()