# Canny Edge Detector

In this work we implement a Canny Edge Detector. We use the Otsu's method to roughly determine the optimal threshold and display detections on a set of 6 images from the CTMRI database. For further detail please read [Canny Edge Detector](https://github.com/Matjaz12/Canny-Edge-Detector/blob/main/report.pdf).

## Setup

In order to run the program user must do the following:

1. Create a virtual environment:

    `python3 -m venv venv`

2. Activate virtual environment:

    `. venv/bin/activate`

3. Install dependencies:

    `pip install -r requirements.txt`


## Usage

You can run the code using:

`python3 canny.py <in_image_path> <in_image_path> <display>`.

For example the following reads image ./data/0014.png and saves detected edges into ./data/0014E.png

`python3 canny.py ./data/0014.png ./data/0014E.png 0`
