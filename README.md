# Hand Segmentation and Action Classification using U-Net

This repository provides code for hand segmentation and action classification using the U-Net architecture. It allows you to train a hand segmentation model and perform inference on images and videos. Additionally, it includes an action classification model to classify actions in cooking videos, such as stirring or adding ingredients.

## Installation

1. Create a virtual environment:
   ```
   python -m venv env
   ```

2. Activate the virtual environment:
   - For Windows:
     ```
     env\Scripts\activate
     ```
   - For Linux/Mac:
     ```
     source env/bin/activate
     ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## A) Training Hand Segmentation Model

We utilize two publicly available datasets for training the segmentation model:
  - EYTH (EgoYoutubeHands)
  - HOF (HandOverFace)

For detailed training instructions, refer to the `Hand_Seg_Unet.ipynb` notebook. [Click here to access the notebook](Hand_Seg_Unet.ipynb).

## B) Running Segmentation Inference

To perform segmentation inference on images, use the following command:
```
python images_inference.py --source [path to image folder] --output [path to results base folder (default: results)]
```

To perform segmentation inference on videos, use the following command:
```
python video_inference.py --input_path [path to video] --output_path [path to results base folder (default: output_video)] --show_video [1 if you want to see live inference (default: 0)]
```

Example:
```
python video_inference.py --input_path ./video_2.mp4
```

The segmentation inference outputs will be stored in the `output_video` folder.

## C) Action Classification

### Problem
The problem consists of classifying actions in cooking videos, specifically identifying whether the action involves stirring or adding ingredients.

### Solution
We have developed an image classification model that takes segmented hand images as input and classifies them into three categories: adding ingredients, stirring, or background.

For detailed instructions on action classification, refer to the `action_classification.ipynb` notebook. [Click here to access the notebook](action_classification.ipynb).

## D) Action Classification Inference

To perform end-to-end inference with hand segmentation and action classification, use the following command:
```
python video_action_inference.py --input_path [path to video file] --output_path [path to results base folder (default: output_video_action)] --show_video [1 if you want to see live inference (default: 0)]
```

The inference outputs will be stored in the `output_video_action` folder.
