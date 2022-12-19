# Canny Edge Detector

Implementation of a canny edge detector.

## Setup

1. Create a virtual environment: `python3 -m venv venv`
1. Activate virtual environment: `. venv/bin/activate`
2. Install dependencies: `pip install -r requirements.txt`


## Usage
You can run the code using: `python3 canny.py <in_image_path> <in_image_path> <display>`.

For example the following reads image ./data/0014.png and saves detected edges into ./data/0014E.png

`python3 canny.py ./data/0014.png ./data/0014E.png 0`

## Evaluation

To evaluate the detector simply run `python3 eval.py`. This computes and saves edges
for each image in the ./data directory.