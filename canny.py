import argparse
import math

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import signal
from scipy.ndimage import gaussian_filter


def nonmaxima_supression(mag, angle):
    g_n = np.zeros_like(angle)
    m, n = g_n.shape

    for i in range(1, m - 1, 1):
        for j in range(1, n - 1, 1):
            
            # check if vertical edge
            if (angle[i,j] >= -22.5 and angle[i,j] <= 22.5) or (angle[i, j] < -157.5 and angle[i, j] >=-180) or (angle[i, j] >= 157.5 and angle[i, j] <= 180):
                pix1 = mag[i, j + 1]
                pix2 = mag[i, j - 1]

            # check if +45 edge
            elif (angle[i,j] >= 22.5 and angle[i,j] <= 67.5) or (angle[i,j] <-112.5 and angle[i,j] >= -157.5):
                pix1 = mag[i - 1, j - 1]
                pix2 = mag[i + 1, j + 1]

            # check if horizontal edge
            elif (angle[i,j] >= 67.5 and angle[i,j] <= 112.5) or (angle[i,j] < -67.5 and angle[i,j] >= -112.5):
                pix1 = mag[i - 1, j]
                pix2 = mag[i + 1, j]

            # check if -45 edge
            elif (angle[i,j] >= 112.5 and angle[i,j] <= 157.5) or (angle[i,j] < -22.5 and angle[i,j] >= -67.5):
                pix1 = mag[i + 1, j - 1]
                pix2 = mag[i - 1, j + 1]
            
            if mag[i, j] >= pix1 and mag[i, j] >= pix2:
                g_n[i, j] = mag[i, j]

    return g_n


def otsus_method(img):
    max_var, best_thresh = 0.0, 0.0

    thresholds = np.arange(np.max(img), step=1.0/256)

    for thresh in thresholds:
        # Compute the between class variance
        background = img[img < thresh]
        foreground = img[img >= thresh]

        w_1 = background.size / img.size
        w_2 = foreground.size / img.size

        # Ignore cases where w_1 ro w_2 is zero
        if w_1 == 0 or w_2 == 0:
            continue

        mean_1 = background.mean()
        mean_2 = foreground.mean()

        var = w_1 * w_2 * ((mean_1 - mean_2) ** 2)

        if var > max_var:
            max_var = var 
            best_thresh = thresh

    return best_thresh


MAX_DEPTH = 10000
def find_conn_weak_edges(edges, row, col, depth=0):
    if depth > MAX_DEPTH:
        return

    m, n = edges.shape

    # Loak at the surrounding 8 pixels
    for i in range(-1, 2, 1):
        for j in range(-1, 2, 1):
            
            # Check if there is a weak edge nearby
            if row + i >= 0 and row + i < m and col + j >= 0 and col + j < n:
                if edges[row + i, col + j] > 0 and edges[row + i, col + j] < 1:
                    depth += 1

                    edges[row + i, col + j] = 1 # label weak edge as legit edge
                    find_conn_weak_edges(edges, row, col, depth)


def canny(img, display=True, threshold_scale=1.0):
    SIGMA = 2

    if display:
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.title(f"(a) Original image")
        plt.xticks([])
        plt.yticks([])
        plt.savefig('./imgs/intermediate1.png', bbox_inches='tight')
        plt.show()

    # (1) Apply Gaussian smoothing
    img = gaussian_filter(img, sigma=SIGMA)

    if display:
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.title(f"(b) Smoothed image")
        plt.xticks([])
        plt.yticks([])
        plt.savefig('./imgs/intermediate2.png', bbox_inches='tight')
        plt.show()

    # (2) Compute the gradient (i.e partial derivative w.r.t to x and y axis)
    kernel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    kernel_y = np.array([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ])
    
    g_x = signal.convolve2d(img, kernel_x, mode="same")
    g_y = signal.convolve2d(img, kernel_y, mode="same")

    # (3) Compute magnitude and angle image
    mag = np.sqrt(g_x ** 2 + g_y ** 2)
    angle = np.arctan2(g_y, g_x) * 180 / np.pi

    if display:
        plt.figure()
        plt.imshow(mag, cmap="gray")
        plt.title(f"(c) Magnitude image")
        plt.xticks([])
        plt.yticks([])
        plt.savefig('./imgs/intermediate3.png', bbox_inches='tight')
        plt.show()

        plt.figure()
        plt.imshow(angle, cmap="gray")
        plt.title(f"(d) Angle image")
        plt.xticks([])
        plt.yticks([])
        plt.savefig('./imgs/intermediate4.png', bbox_inches='tight')
        plt.show()

    # (4) Thin edges using non maxima supression
    g_n = nonmaxima_supression(mag, angle)

    if display:
        plt.figure()
        plt.imshow(g_n, cmap="gray")
        plt.title(f"(e) Edges after non-maxima supression")
        plt.xticks([])
        plt.yticks([])
        plt.savefig('./imgs/intermediate5.png', bbox_inches='tight')
        plt.show()


    # (5) Determine the optimal threshold using Otsu's method
    g_n = (g_n - g_n.min()) / (g_n.max() - g_n.min()) # normalize the image
    thresh_high = threshold_scale * otsus_method(g_n)
    thresh_low = 0.5 * thresh_high

    if display:
        print(f"thresh_high: {thresh_high}, thresh_low: {thresh_low}")

    # (6) Label strong and weak edges
    edges = g_n.copy()

    edges[edges > thresh_high] = 1  # label all edges above thresh_high as legit edges
    edges[edges < thresh_low] = 0   # remove all edges bellow thresh_low

    # (7) Link edges using 8-connectivity
    rows, cols = np.where(edges == 1)
    for s_row, s_col in zip(rows, cols):
        find_conn_weak_edges(edges, s_row, s_col)

    edges[(edges > 0) & (edges < 1)] = 0 # remove all remaining weak edges 

    if display:
        plt.figure()
        plt.imshow(edges, cmap="gray")
        plt.title(f"(f) Detected edges")
        plt.xticks([])
        plt.yticks([])
        plt.savefig('./imgs/intermediate6.png', bbox_inches='tight')
        plt.show()

    return edges


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'Canny Edge Detector',
                    description = 'Detects edges in the image and (possibly) display intermediate results')


    parser.add_argument("in_image", type=str, help="Input image path")
    parser.add_argument("out_image", type=str, help="Output image path")
    parser.add_argument("dispay", type=int, default=0, help="Display / Don't display intermediate results")
    parser.add_argument("threshold_scale", type=float, default=1.0, help="Factor between 0 and 1 by which the high threshold (determined using the Otsu's method) will be scaled.")


    args = parser.parse_args()
    # print(args.dispay)

    image = np.array(Image.open(args.in_image).convert("L"))
    edges = canny(image, args.dispay, args.threshold_scale) * 255

    edges = Image.fromarray(edges.astype('uint8'))
    edges.save(args.out_image)