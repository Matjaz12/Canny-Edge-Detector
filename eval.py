import json
import os
from os import listdir

import numpy as np
from PIL import Image

from canny import canny



if __name__ == "__main__":
    print("Detecting edges...")

    # Load evaluation settings
    with open("./eval_settings.json", "r") as file:
        eval_settings = json.load(file)

    # print(eval_settings)
    
    # Detect and save edges for all images
    folder_dir = "./data/"
    for image_name in os.listdir(folder_dir):
        if (image_name.endswith(".png") or image_name.endswith(".jpg") or image_name.endswith(".jpeg")):
            if "E" not in image_name:
                out_name = image_name.split(".")
                image_setting = eval_settings[out_name[0]]
                threshold_scale = image_setting["threshold_scale"]
                out_name[0] += "E."
                out_name = "".join(out_name)
                # print(f"image_name: {image_name}, out_name: {out_name}, threshold_scale: {threshold_scale}")

                image = np.array(Image.open(folder_dir + image_name).convert("L"))
                edges = canny(image, display=0, threshold_scale=threshold_scale) * 255
                edges = Image.fromarray(edges.astype('uint8'))
                edges.save(folder_dir + out_name)
