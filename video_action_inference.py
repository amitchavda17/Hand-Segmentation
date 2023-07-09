import cv2
import numpy as np
import argparse
import tensorflow as tf
from utils import *
import os



def classify_action(classifier_model,image):
    classes = {0:'Adding Ingrident',1:"",2:"Stirring"} # 1 is other
    # Resize the image to the input shape of the model
    image = cv2.resize(image, (224, 224))
    # Convert the image to float32 and normalize the pixel values
    image = image.astype(np.float32) / 255.0
    # Expand the dimensions to match the expected input shape
    image = np.expand_dims(image, axis=0)
    
    pred = classifier_model.predict(image)
    pred = np.argmax(pred[0])
    
    action = classes[pred]
    
    return action

def overlay_text_on_image(image, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1.0, font_color=(255, 255, 255), thickness=2):
    overlayed_image = image.copy()

    # Get the dimensions of the text
    text_size, _ = cv2.getTextSize(text, font, font_scale, thickness)

    # Calculate the position to center the text on the top side
    position_x = int((image.shape[1] - text_size[0]) / 2)
    position_y = text_size[1] + 10  # Add a small margin

    # Overlay the text on the image
    cv2.putText(overlayed_image, text, (position_x, position_y), font, font_scale, font_color, thickness)

    return overlayed_image

def main(args):
    # Load the U-Net model
    model =  tf.keras.models.load_model(
        './unet_weights.h5',
        custom_objects={
            "bce_dice_loss":bce_dice_loss ,
            "dice_loss": dice_loss,
            "dice_coeff": dice_coeff,
        },
    )
    
    classifier_model  =  tf.keras.models.load_model('./action_classifier.h5') 

    # Open the video file
    video = cv2.VideoCapture(args.input_path)
    out_file_name = 'out_'+args.input_path.split('/')[-1]
    
    # Get video properties
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video.get(cv2.CAP_PROP_FPS)
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create a VideoWriter object to save the output video
    os.makedirs(args.output_folder,exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_video = cv2.VideoWriter(os.path.join(args.output_folder,out_file_name), fourcc, fps, (width, height))
    
    # Process each frame in the video
    for frame_num in range(num_frames):
        ret, frame = video.read()
        if not ret:
            break

        # Preprocess the frame
        input_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #tf_image = tf.convert_to_tensor(image, dtype=tf.float32)
        # image = tf.io.read_file(image_path)
        # image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        input_image = tf.image.resize(image, (256, 256), method="nearest")
        input_image = input_image / 255.0
        input_image = tf.expand_dims(input_image, axis=0)

        # Perform inference
        pred = model.predict(input_image)
        pred=pred[0]
        # Create segmentation mask
        mask = np.squeeze(pred) > 0.9
        mask = mask.astype(np.uint8) * 255
        mask = cv2.resize(mask, (width, height))

        # Overlay mask on the frame
        mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)


        output_frame = cv2.addWeighted(frame, 0.7, mask_rgb, 0.3, 0)
        
        #Predict action
        action_type = classify_action(classifier_model,mask_rgb)
       
         

        output_frame = overlay_text_on_image(output_frame,action_type)
        # Write the output frame to the video file
        output_video.write(output_frame)

        #Display the output frame (optional)
        if args.show_video==1:
            cv2.imshow('Output', output_frame)
            if cv2.waitKey(1) == ord('q'):
                break

    # Release resources
    video.release()
    output_video.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Video Segmentation using U-Net')
    
    parser.add_argument('--input_path', type=str, help='Path to input video file', required=True)
    parser.add_argument('--output_folder', type=str, help='Path to save output video file', default = './output_video_action')
    parser.add_argument('--show_video', type=int, help='Path to save output video file', required=False,default=0)
    args = parser.parse_args()

    # Call the main function
    main(args)
