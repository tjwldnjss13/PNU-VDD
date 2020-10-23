import numpy as np
from PIL import Image

from model1 import Model1

if __name__ == '__main__':
    img_path = ''
    img = Image.open(img_path).convert('L')
    img = np.array(img)

    model = Model1(10)

    y_ = model.forward(img)
    cls = np.argmax(y_)
