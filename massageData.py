import numpy as np
import os
import myConstants as mc

from sklearn.model_selection import train_test_split

class massageData():

        def getData(self):
                """ Returns the full dataset. This is 100% of the loaded data. """
                return (self.X, self.Y)

        def getTrain(self):
                """ Returns a tuple x,y for the training dataset. """
                return (self.X_train, self.y_train)

        def getTest(self):
                """ Returns a tuple x,y for the test dataset. """
                return (self.X_test, self.y_test)

        def getDev(self):
                """ Returns a tuple of x,y for the developmetn dataset. """
                return (self.X_dev, self.y_dev)

        def __init__(self, folder='data/numpy_bitmap/'):
                """
                Creates is new instsance of massageData.

                This class is responsible for loading in the image datase and generating

        Arguments:
                - folder: Path to the folder with the npy image array data. Important
                Path should end with a '/' at the end.
                If not set, defaults to 'data/numpy_bitmap/'
                """
                self.folder = folder
                self.X = np.zeros((0, mc.BITMAP_DIM))
                self.Y = []

                # Split into train / test / dev. We use a split ratio of 80% train, 10% dev, 10% test.
                # To get the 3 pieces, we'll do the splits as follows:
                # - Split dataset into 80/20 train/eval to get train set.
                # - Split eval piece 50/50 to get dev/test sets which each makes 10% of the overall data.
                #
                # Further, to ensure that the split is done uniformly across all the various categories, we
                # will do the split for each categories (e.g. airplane) as we load it. This avoids any
                # potential issues train/test/dev splits not being fairly balanced.

                self.X_train = np.zeros((0, mc.BITMAP_DIM))
                self.y_train = []

                self.X_dev = np.zeros((0, mc.BITMAP_DIM))
                self.y_dev = []

                self.X_test= np.zeros((0, mc.BITMAP_DIM))
                self.y_test = []

                random_seed = 42
                max_num_examples_per_class = 70000
                for index, filename in enumerate(os.listdir(self.folder)):
                        fullpath = self.folder + filename
                        # use filename as Y label.
                        print("Processing file #: {} - Name={}".format(index, filename))
                        name = filename[:-4]
                        x = np.load(fullpath)
                        x = x[:max_num_examples_per_class,:]
                        print("X has shape", x.shape)
                        y = [name] * x.shape[0]
                        # Full dataset.
                        #self.X = np.concatenate((self.X, x), axis=0)
                        #self.Y += y

                        # Split into train/test/dev. See also comment above.
                        X_train, X_eval, y_train, y_eval = train_test_split(x, y, test_size=0.2,random_state=random_seed)
                        X_dev, X_test, y_dev, y_test = train_test_split(X_eval, y_eval, test_size=0.5, random_state=random_seed)

                        # Train split.
                        self.X_train = np.concatenate((self.X_train, X_train), axis=0)
                        self.y_train += y_train

                        # Dev split.
                        self.X_dev = np.concatenate((self.X_dev, X_dev), axis=0)
                        self.y_dev += y_dev

                        # Test split.
                        self.X_test = np.concatenate((self.X_test, X_test), axis=0)
                        self.y_test += y_test



                # Convert into NP Arrays.
                self.Y = np.array(self.Y)
                self.y_train = np.array(self.y_train)
                self.y_test = np.array(self.y_test)
                self.y_dev = np.array(self.y_dev)

                print("All: ", self.X.shape)
                print("train: ", self.X_train.shape)
                print("Test: ", self.X_test.shape)
                print("Dev: ", self.X_dev.shape)


def main():
        data = massageData()
        print("x values: ", data.getX().shape)
        print("y values: ", data.getY().shape)

if __name__ == '__main__':
        main()
