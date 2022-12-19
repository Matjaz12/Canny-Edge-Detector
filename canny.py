import numpy as np
import matplotlib.pyplot as plt
import math
from scipy.ndimage import gaussian_filter
from PIL import Image
from scipy import signal
import argparse


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

    thresholds = range(int(np.max(img)) + 1)

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


def link_edges(g_nh, g_nl):
    g_nh = np.pad(g_nh, ((1, 1), (1, 1)), "constant")
    m, n = g_nh.shape

    for i in range(1, m - 1, 1):
        for j in range(1, n - 1, 1):
            if g_nh[i, j] == 0.0:
                continue

            # Take a sub region of the (g_nl) image
            win = g_nl[i - 1 : i + 2, j - 1 : j + 2]

            # Find where pixel with a non zero value in the sub region
            idxs = np.where(win != 0.0)
            idxs = (idxs[0] + (i - 1), idxs[1] + (j - 1))

            # Mark as the strong pixel
            g_nh[idxs] = 1.0

    g_nh = g_nh[1: m - 1, 1 : n - 1]

    return g_nh 


def canny(img, display=True):
    SIGMA = 2

    # (1) Apply Gaussian smoothing
    img = gaussian_filter(img, sigma=SIGMA)

    if display:
        plt.figure()
        plt.imshow(img, cmap="gray")
        plt.title(f"Image filtered with a gaussian kernel (sigma: {SIGMA})")
        plt.xticks([])
        plt.yticks([])
        plt.show()

    # (2) Compute the gradient (i.e partial derivative w.r.t to x and y)
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
        fig, (ax1, ax2) = plt.subplots(1, 2)

        ax1.imshow(mag, cmap="gray");
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title("magnitude image")

        ax2.imshow(angle, cmap="gray");
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("angle image")

        plt.show()

    # (4) Remove "weak" edges using non maxima supression
    g_n = nonmaxima_supression(mag, angle)

    if display:
        plt.figure()
        plt.imshow(g_n, cmap="gray")
        plt.title(f"Output of non-maxima supression")
        plt.xticks([])
        plt.yticks([])
        plt.show()

    # (5) determine the optimal threshold using Otsu's method
    thresh_high = otsus_method(g_n)
    thresh_low = 0.5 * thresh_high

    if display:
        print(f"thresh_high: {thresh_high}, thresh_low: {thresh_low}")

    # (6) compute strong and weak edge pixels

    g_nh = np.zeros_like(g_n) # strong edges
    g_nh[np.where(g_n > thresh_high)] = 1.0

    g_nl = np.zeros_like(g_n) # weak edges
    g_nl[np.where(g_n > thresh_low)] = 1.0
    g_nl = g_nl - g_nh

    if display:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        
        ax1.imshow(g_nh, cmap="gray");
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title("Strong edges")

        ax2.imshow(g_nl, cmap="gray");
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("Weak edges")

        plt.show()

    # (7) link edges using 8-connectivity
    edges = link_edges(g_nh, g_nl)

    if display:
        plt.figure()
        plt.imshow(edges, cmap="gray")
        plt.title(f"Detected edges")
        plt.xticks([])
        plt.yticks([])
        plt.show()

    return edges

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog = 'Canny Edge Detector',
                    description = 'Detects edges in the image and (possibly) display intermediate results')


    parser.add_argument("in_image", type=str, help="Input image path")
    parser.add_argument("out_image", type=str, help="Output image path")
    parser.add_argument("dispay", type=int, help="Display / Don't display intermediate results")

    args = parser.parse_args()
    print(args.dispay)

    image = np.array(Image.open(args.in_image).convert("L"))
    edges = canny(image, args.dispay) * 255

    edges = Image.fromarray(edges.astype('uint8'))
    edges.save(args.out_image)