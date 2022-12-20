import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import os
from PIL import Image

if __name__ == "__main__":
    edge_images, images = [], []

    folder_dir = "./data/"
    for image_name in os.listdir(folder_dir):
        if (image_name.endswith(".png") or image_name.endswith(".jpg") or image_name.endswith(".jpeg")):
            image = np.array(Image.open(folder_dir + image_name).convert("L"))
            if "E" in image_name:
                edge_images.append([image, image_name])
            else:
                images.append([image, image_name])


    # Sort by name
    edge_images = sorted(edge_images, key = lambda x: x[1])
    images = sorted(images, key = lambda x: x[1])


    fig = plt.figure(figsize=(10, 15))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, len(images) // 3),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

    idx = 0
    for ax, im in zip(grid, edge_images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im[0], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])

        idx += 1

    plt.savefig('./imgs/edge_grid.pdf', bbox_inches='tight')
    plt.show()

    fig = plt.figure(figsize=(10, 15))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(3, len(images) // 3),  # creates 2x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

    idx = 0
    for ax, im in zip(grid, images):
        # Iterating over the grid returns the Axes.
        ax.imshow(im[0], cmap="gray")
        ax.set_xticks([])
        ax.set_yticks([])

        idx += 1

    plt.savefig('./imgs/image_grid.pdf', bbox_inches='tight')
    plt.show()