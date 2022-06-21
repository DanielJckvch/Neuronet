import numpy as np

from PIL import Image

img = np.asarray(Image.open('img.jpg').convert('RGB'))

list = img.tolist()

print(0)