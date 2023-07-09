import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from utils import *
import argparse
import numpy as np


def preprocess_img(image_path, size=256):
    """Preprocess image for feeding into the model.

    Args:
        image_path: Path to the input image.
        size (int, optional): Image scale. Defaults to 256.

    Returns:
        img_tensor: (1, size, size, 3) image tensor.
    """
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    input_image = tf.image.resize(image, (256, 256), method="nearest")
    input_image = input_image / 255.0
    input_image = tf.expand_dims(input_image, axis=0)
    return input_image


def get_inference(model, input_img):
    """Run model inference.

    Args:
        model: U-Net model.
        input_img: Input image tensor.

    Returns:
        out_img: Predicted binary image (lane mask).
    """
    pred = model.predict(input_img)
    pred = pred[0]
    # Create segmentation mask
    mask = np.squeeze(pred) > 0.5
    out_img = mask.astype(np.uint8) * 255
    return out_img


def overlay_mask(image, mask):
    """Overlay the mask on the input image.

    Args:
        image: Input image.
        mask: Predicted binary mask.

    Returns:
        overlayed: Image with mask overlaid.
    """
    overlayed = cv2.addWeighted(image, 1, mask, 0.5, 0)
    return overlayed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference")
    parser.add_argument("--source", type=str, help="Path to image or folder")
    parser.add_argument("--output", type=str, help="Path to results folder", default="./results")
    parser.add_argument("--weights", type=str, help="Weights path", default='./unet_weights.h5')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    model = load_model(
        args.weights,
        custom_objects={
            "bce_dice_loss": bce_dice_loss,
            "dice_loss": dice_loss,
            "dice_coeff": dice_coeff,
        },
    )

    if os.path.isfile(args.source):
        print(args.source)
        img_path = args.source
        input_img = preprocess_img(img_path)
        prediction = get_inference(model, input_img)
        output_folder = os.path.join(args.output, os.path.basename(os.path.dirname(img_path)))
        os.makedirs(output_folder, exist_ok=True)

        # Save predicted mask
        predicted_mask_folder = os.path.join(output_folder, "predicted_mask")
        os.makedirs(predicted_mask_folder, exist_ok=True)
        img_name = os.path.basename(img_path)
        predicted_mask_path = os.path.join(predicted_mask_folder, img_name)
        cv2.imwrite(predicted_mask_path, prediction)

        # Overlay mask on the input image and save it
        input_img = cv2.imread(img_path)
        mask = cv2.resize(prediction, (input_img.shape[1], input_img.shape[0]))
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        overlayed = cv2.addWeighted(input_img, 0.6, mask_rgb, 0.4, 0)
        masked_img_folder = os.path.join(output_folder, "masked_image")
        os.makedirs(masked_img_folder, exist_ok=True)
        masked_img_path = os.path.join(masked_img_folder, img_name)
        cv2.imwrite(masked_img_path, overlayed)

    elif args.source:
        for root, dirs, files in os.walk(args.source):
            for filename in files:
                img_path = os.path.join(root, filename)
                print("Processing", img_path)
                input_img = preprocess_img(img_path)
                prediction = get_inference(model, input_img)
                output_folder = os.path.join(args.output, os.path.basename(root))
                os.makedirs(output_folder, exist_ok=True)

                # Save predicted mask
                predicted_mask_folder = os.path.join(output_folder, "predicted_mask")
                os.makedirs(predicted_mask_folder, exist_ok=True)
                output_path = os.path.join(predicted_mask_folder, filename)
                cv2.imwrite(output_path, prediction)

                # Overlay mask on the input image and save it
                input_img = cv2.imread(img_path)
                mask = cv2.resize(prediction, (input_img.shape[1], input_img.shape[0]))
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
                overlayed = cv2.addWeighted(input_img, 0.6, mask_rgb, 0.4, 0)
                masked_img_folder = os.path.join(output_folder, "masked_image")
                os.makedirs(masked_img_folder, exist_ok=True)
                #print('hi')
                masked_img_path = os.path.join(masked_img_folder, filename)
                cv2.imwrite(masked_img_path, overlayed)
