import numpy as np
import os

# Dimension for the input feature bitmap vectors.
BITMAP_DIM = 784

class massageData():

	def getX(self):
		return self.X

	def getY(self):
		return self.Y
	
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
			# use filename as Y
			name = filename[:-4]
			x = np.load(fullpath)
			y = [name] * x.shape[0]
			self.X = np.concatenate((self.X, x), axis=0)
			self.Y += y
		self.Y = np.array(self.Y)

def main():
	data = massageData()
	print("x values: ", data.getX().shape)
	print("y values: ", data.getY().shape)

if __name__ == '__main__':
	main()