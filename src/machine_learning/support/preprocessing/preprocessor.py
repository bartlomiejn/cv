import cv2
from keras.preprocessing.image import img_to_array


class ImageToArrayPreprocessor:
	def __init__(self, data_format=None):
		self.data_format = data_format

	def process(self, image):
		"""
        Returns the image with channel ordering based on Keras configuration
        json.
        """
		return img_to_array(image, data_format=self.data_format)


class ResizePreprocessor:
	def __init__(self, width, height, interpolation=cv2.INTER_AREA):
		self.width = width
		self.height = height
		self.interpolation = interpolation

	def preprocess(self, image):
		return cv2.resize(
			image,
			(self.width, self.height),
			interpolation=self.interpolation
		)