import logging
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO)

def display_4_images(images, labels):

    if images.shape[0] < 4:
        logging.error('Need at least 4 images to display')
        return

    fig = plt.figure()

    for idx in range(4):
        ax = fig.add_subplot(2, 2, idx + 1, xticks=[], yticks=[])
        ax.imshow(np.squeeze(images[idx]), cmap='gray')
        ax.set_title(str(labels[idx]))

    plt.show()    
