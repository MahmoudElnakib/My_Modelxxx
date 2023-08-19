import numpy as np
from PIL import Image

img = Image.open('demo.jpg')
img_data = preprocess(img)
Image.fromarray(img_data).save("preprocess.jpg")


def preprocess(image):
	image = np.array(img.getdata()).reshape(img.size[1], img.size[0], 3)
	image = np.expand_dims(img_data.astype(np.uint8), axis=0)

	return image
img = Image.open('000409.png')
img_data = preprocess(img)
Image.fromarray(img_data).save("preprocess.png")
	
