import cv2
import os

def frames_to_video(input_folder, output_video_path, output_frame_rate=30.0):
    # Get the list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.endswith((".jpg", ".png", ".jpeg"))]

    # Sort the image files based on their names (assuming the names represent frame order)
    image_files.sort()

    # Get the dimensions of the first image to set the video size
    first_image_path = os.path.join(input_folder, image_files[0])
    first_image = cv2.imread(first_image_path)
    height, width, channels = first_image.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # For MP4 format, use "XVID" for AVI format
    out = cv2.VideoWriter(output_video_path, fourcc, output_frame_rate, (width, height))

    # Write frames to the video
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        frame = cv2.imread(image_path)
        out.write(frame)

    # Release the video writer
    out.release()

# Replace "path/to/input/frames/folder" and "output_video.mp4" with your input frames folder and desired output video file name, respectively
frames_folder = "test\\038_9"
output_video_file = "038_9.mp4"
frames_to_video(frames_folder, output_video_file)
