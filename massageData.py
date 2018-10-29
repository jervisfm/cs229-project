import numpy as np
import os

from sklearn.model_selection import train_test_split

# Dimension for the input feature bitmap vectors.
BITMAP_DIM = 784

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
		self.X = np.zeros((0, BITMAP_DIM))
		self.Y = []
		for filename in os.listdir(self.folder):
			fullpath = self.folder + filename
			# use filename as Y label.
			name = filename[:-4]
			x = np.load(fullpath)
			y = [name] * x.shape[0]
			self.X = np.concatenate((self.X, x), axis=0)
			self.Y += y
		self.Y = np.array(self.Y)

		# Split into train / test / dev. We use a split ratio of 80% train, 10% dev, 10% test.
		# To get the 3 pieces, we'll do the splits as follows:
		# - Split dataset into 80/20 train/eval to get train set.
		# - Split eval piece 50/50 to get dev/test sets which each makes 10% of the overall data.

		random_seed = 42
		self.X_train, self.X_eval, self.y_train, self.y_eval = train_test_split(self.X, self.Y, test_size=0.2,
																				random_state=random_seed)


		self.X_dev, self.X_test, self.y_dev, self.y_test = train_test_split(self.X_eval, self.y_eval, test_size=0.5,


																			random_state=random_seed)



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