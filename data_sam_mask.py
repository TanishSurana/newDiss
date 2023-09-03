import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def detect_sobel_edges(image_path):
    # Read the image
    
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply Sobel edge detection
    sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Detect edges in X direction
    sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Detect edges in Y direction

    # Compute the magnitude of gradients
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)

    # Convert to 8-bit image (for display purposes)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return magnitude

import cv2
import numpy as np

def detect_sobel_edges_pil(image, threshold=30):
    
    image_np = np.array(image)
    sobel_x = cv2.Sobel(image_np, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(image_np, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
    magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply morphological dilation to increase the thickness of edges
    kernel = np.ones((3, 3), np.uint8)
    thick_edges = cv2.dilate(magnitude, kernel, iterations=5)
    
    # Binary thresholding
    _, binary_image = cv2.threshold(thick_edges, threshold, 255, cv2.THRESH_BINARY)

    return Image.fromarray(binary_image)


# Use the function
image_path = 'semi_optic_features\\test\\055_5\\0010.jpg'
opticimage = Image.open(image_path).convert("L")
edges = detect_sobel_edges_pil(opticimage)

#edges = detect_sobel_edges(image_path)

# Display the result
plt.imshow(edges, cmap='gray')
plt.title('Sobel Edge Detection')
plt.show()



