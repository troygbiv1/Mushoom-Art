import cv2
import os
import numpy as np
import glob
import subprocess 

# Paths
input_folder = 'mushroom_images1'
output_folder = 'outlines1'

# Create the output folder if it doesn't exist
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process each image in the input folder
for file_path in glob.glob(input_folder + '/*.jpeg'):  # Assumes .jpeg format
    # Read the image
    image = cv2.imread(file_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Apply Gaussian blur
    blurred_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
    # Binary thresholding after Gaussian blur
    threshold_value = 15
    _, thresh_image = cv2.threshold(blurred_image, threshold_value, 255, cv2.THRESH_BINARY)
    # Find contours from the binary image
    contours, hierarchy = cv2.findContours(thresh_image.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # Minimum area threshold for contours (adjust as needed)
    min_area_threshold = 1500 
    # Create an empty BGRA image (B, G, and R channels are 255 for white, Alpha channel is 0 for transparent)
    contour_image = np.zeros((gray_image.shape[0], gray_image.shape[1], 4), dtype=np.uint8)
    contour_image[:, :, 0:3] = 0  # Set B, G, and R channels to 255 (white)

    # Draw the contours on the empty image
    for i, contour in enumerate(contours):
          if cv2.contourArea(contour) > min_area_threshold:
               if hierarchy[0][i][3] == -1:
                    # fill external contours
                    cv2.drawContours(contour_image, [contour], -1, (255, 255, 255, 255), -1)
               else:
                    # delete Fill internal contours
                    cv2.drawContours(contour_image, [contour], -1, (0, 0, 0, 0), -1)
    # Save the contour image   
    base_name = os.path.basename(file_path)
    output_path = os.path.join(output_folder, 'contour_' + os.path.splitext(base_name)[0] + '.png')
    cv2.imwrite(output_path, contour_image)

image_dir = "outlines1"

# Change the current directory to the image directory
os.chdir(image_dir)

# FFmpeg command
start_number = 273  # Adjust this to your starting image number
ffmpeg_command = f"ffmpeg -framerate 24 -start_number {start_number} -i contour_MushroomRen_%04d.png -c:v libx264 -r 30 -pix_fmt yuv420p OutlineTimelapse.mp4"
# Run the FFmpeg command
subprocess.run(ffmpeg_command, shell=True)

print("All images processed and contours extracted.")