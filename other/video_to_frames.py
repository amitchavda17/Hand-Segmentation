import os
import cv2
import argparse

def convert_video_to_frames(video_path, output_dir):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(video_path)
    file_name = video_path.split('/')[-1].split('.')[0]
    # Get video properties
    num_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Convert video to frames
    for frame_num in range(num_frames):
        ret, frame = video.read()
        if not ret:
            break

        # Save the frame as an image file
        frame_name = f"{file_name}_{frame_num:06d}.jpg"
        frame_path = os.path.join(output_dir, frame_name)
        cv2.imwrite(frame_path, frame)

    # Release resources
    video.release()

def main(args):
    # Get the video file name without extension
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]

    # Create the output directory path
    output_dir = os.path.join("frames", video_name.replace('.mp4',''))

    # Convert video to frames
    convert_video_to_frames(args.video_path, output_dir)

    print(f"Video frames saved to: {output_dir}")

if __name__ == '__main__':
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Video to Frames Converter')
    parser.add_argument('--video_path', type=str, help='Path to input video file', required=True)
    args = parser.parse_args()

    # Call the main function
    main(args)
