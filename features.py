import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

# Configuration
IMG_SIZE = (256, 256) # Standardized size 256,256

def extract_features(image_path):
    """
    1. Loads image.
    2. Performs K-Means Segmentation to isolate disease spots[cite: 38].
    3. Extracts Color, Texture (GLCM), and Shape features [cite: 52-53].
    """
    #  PREPROCESSING 
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    #  SEGMENTATION K-Means 
    # Reshape image to a list of pixels
    pixel_values = img_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # number of clusters k=2: Healthy vs Diseased/Background
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    k = 2
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Reconstruct the segmented image
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_image = segmented_data.reshape(img_rgb.shape)

    # FEATURE EXTRACTION 
    features = []

    # Color Features: Mean & Std Dev for R, G, B channels 
    for i in range(3):
        channel = segmented_image[:,:,i]
        features.append(np.mean(channel))
        features.append(np.std(channel))

    # Texture Features: GLCM Contrast, Correlation, Energy 
    gray = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
    # Calculate GLCM (distance=1, angle=0 degrees)
    glcm = graycomatrix(gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    features.append(graycoprops(glcm, 'contrast')[0, 0])
    features.append(graycoprops(glcm, 'correlation')[0, 0])
    features.append(graycoprops(glcm, 'energy')[0, 0])

    # Shape Features: Ratio of diseased area to total area 
    # We are assumming the "disease" part is the non-zero or non-background part after segmentation
    non_zero_pixels = cv2.countNonZero(gray)
    total_pixels = gray.size
    area_ratio = non_zero_pixels / total_pixels
    features.append(area_ratio)

    return np.array(features)