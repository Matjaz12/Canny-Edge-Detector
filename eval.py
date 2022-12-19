import os
from os import listdir
from canny import canny
import numpy as np
from PIL import Image


if __name__ == "__main__":

    folder_dir = "./data/"
    print("Detecting edges...")
    
    for image_name in os.listdir(folder_dir):
        if (image_name.endswith(".png") or image_name.endswith(".jpg") or image_name.endswith(".jpeg")):
            if "E" not in image_name:
                out_name = image_name.split(".")
                out_name[0] += "E."
                out_name = "".join(out_name)
                print(f"image_name: {image_name}, out_name: {out_name}")

                image = np.array(Image.open(folder_dir + image_name).convert("L"))
                edges = canny(image, display=0) * 255
                edges = Image.fromarray(edges.astype('uint8'))
                edges.save(folder_dir + out_name)
