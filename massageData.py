import numpy as np
import os
import utils

from sklearn.model_selection import train_test_split

class massageData():
        def binarize(self, x):
                """ Set all pixel with value > 0 to be 255. 
                This means we only differentiate between 'have ink' and
                'does not have ink'"""
                x[x>0] = 255
                return x

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

        def __init__(self, folder='data/numpy_bitmap/', binarize=False):
                """
                Creates is new instsance of massageData.

                This class is responsible for loading in the image datase and generating

        Arguments:
                - folder: Path to the folder with the npy image array data. Important
                Path should end with a '/' at the end.
                If not set, defaults to 'data/numpy_bitmap/'

                - convertTo3Channels: Converts the input data into 3 rgb channels. Useful to transfer
                learning with imagenet models that expected images with rgb channels.
                """
                self.folder = folder
                self.X = np.zeros((0, utils.BITMAP_DIM))
                self.Y = []

                # Split into train / test / dev. We use a split ratio of 80% train, 10% dev, 10% test.
                # To get the 3 pieces, we'll do the splits as follows:
                # - Split dataset into 80/20 train/eval to get train set.
                # - Split eval piece 50/50 to get dev/test sets which each makes 10% of the overall data.
                #
                # Further, to ensure that the split is done uniformly across all the various categories, we
                # will do the split for each categories (e.g. airplane) as we load it. This avoids any
                # potential issues train/test/dev splits not being fairly balanced.

                self.X_train = np.zeros((0, utils.BITMAP_DIM))
                self.y_train = []

                self.X_dev = np.zeros((0, utils.BITMAP_DIM))
                self.y_dev = []

                self.X_test= np.zeros((0, utils.BITMAP_DIM))
                self.y_test = []

                random_seed = 42
                # Some classes have as much as 100-250K examples. This leads to OOMS when loading data via numpy. So limit number of examples per class.
                max_num_examples_per_class = 20000
                for index, filename in enumerate(os.listdir(self.folder)):
                        fullpath = self.folder + filename
                        # use filename as Y label.
                        print("Processing file #: {} - Name={}".format(index, filename))
                        name = filename[:-4]
                        x = np.load(fullpath)
                        x = x[:max_num_examples_per_class,:]
                        print("X has shape", x.shape)
                        
                        if binarize:
                                print('USING BINARIZED DATA')
                                x = self.binarize(x)

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

if __name__ == '__main__':
        main()
