# import cv2
# import numpy as np

# def calculate_gradient(blurred_image):
#     raise NotImplementedError

# def custom_edge_detection(image):
#     # Step 1: Preprocessing
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.5)

#     # Step 2: Gradient Calculation
#     gradient_magnitude = calculate_gradient(blurred_image)

#     # Step 3: Non-Maximum Suppression
#     thinned_edges = non_maximum_suppression(gradient_magnitude)

#     # Step 4: Thresholding
#     strong_edges, weak_edges = thresholding(thinned_edges)

#     # Step 5: Edge Tracking by Hysteresis
#     final_edges = edge_tracking(strong_edges, weak_edges)

#     return final_edges

# # Example usage
# image = cv2.imread('a.jpg')
# edges = custom_edge_detection(image)
# cv2.imshow('Detected Edges', edges)
# cv2.waitKey(0)
# cv2.destroyAllWindows()


import cv2
import numpy as np

def calculate_gradient(blurred_image):
    # Compute gradients using Sobel operator
    gx = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)

    # Calculate gradient magnitude and direction
    magnitude = np.sqrt(gx**2 + gy**2)
    direction = np.arctan2(gy, gx)

    return magnitude, direction

def non_maximum_suppression(gradient_data):
    magnitude, direction = gradient_data
    M, N = magnitude.shape
    output = np.zeros((M, N), dtype=np.uint8)
    angle = direction * 180. / np.pi
    angle[angle < 0] += 180

    for i in range(1, M - 1):
        for j in range(1, N - 1):
            q = 255
            r = 255

            # Angle 0
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                q = magnitude[i, j + 1]
                r = magnitude[i, j - 1]
            # Angle 45
            elif (22.5 <= angle[i, j] < 67.5):
                q = magnitude[i + 1, j - 1]
                r = magnitude[i - 1, j + 1]
            # Angle 90
            elif (67.5 <= angle[i, j] < 112.5):
                q = magnitude[i + 1, j]
                r = magnitude[i - 1, j]
            # Angle 135
            elif (112.5 <= angle[i, j] < 157.5):
                q = magnitude[i - 1, j - 1]
                r = magnitude[i + 1, j + 1]

            if (magnitude[i, j] >= q) and (magnitude[i, j] >= r):
                output[i, j] = magnitude[i, j]
            else:
                output[i, j] = 0

    return output

def thresholding(image, low_ratio=0.05, high_ratio=0.15):
    high_threshold = image.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    strong = np.zeros_like(image, dtype=np.uint8)
    weak = np.zeros_like(image, dtype=np.uint8)

    strong_edges = (image >= high_threshold)
    weak_edges = ((image >= low_threshold) & (image < high_threshold))

    strong[strong_edges] = 255
    weak[weak_edges] = 75

    return strong, weak

def edge_tracking(strong, weak):
    M, N = strong.shape
    result = np.copy(strong)
    for i in range(1, M - 1):
        for j in range(1, N - 1):
            if weak[i, j] == 75:
                if np.any(strong[i-1:i+2, j-1:j+2] == 255):
                    result[i, j] = 255
                else:
                    result[i, j] = 0
    return result

def custom_edge_detection(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1.5)

    gradient_data = calculate_gradient(blurred_image)
    thinned_edges = non_maximum_suppression(gradient_data)
    strong_edges, weak_edges = thresholding(thinned_edges)
    final_edges = edge_tracking(strong_edges, weak_edges)

    return final_edges

# Example usage
image = cv2.imread('l5.jpg')
edges = custom_edge_detection(image)
cv2.imshow('Detected Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
