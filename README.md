# EXP-12-IMAGE-PROCESSING-AND-OBJECT-DETECTION

## Name
Thamizh S

## Register Number
212224040450

## Objective

To implement different image processing and computer vision techniques using OpenCV for region of interest segmentation, handwriting detection, and object detection.

The techniques used are:

- ROI Segmentation using Bitwise AND
- Handwriting Detection using Canny Edge Detection
- Contour Detection
- Object Detection using MobileNet-SSD
- Bounding Box and Label Detection

## Requirements

- Python 3
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- MobileNet-SSD model files
- Input images

## Methodology

1. Import the required Python libraries.
2. Read the input image and convert it from BGR to RGB.
3. Define a Region of Interest (ROI) using specific coordinates.
4. Create a blank mask and place the selected ROI on it.
5. Perform bitwise AND operation to segment the ROI.
6. Convert the image to grayscale for handwriting detection.
7. Apply Gaussian Blur to reduce image noise.
8. Apply Canny Edge Detection to identify edges.
9. Detect contours from the edge image.
10. Filter contours based on their area and draw bounding boxes.
11. Load the pretrained MobileNet-SSD model using the configuration and weights files.
12. Define the class labels for object detection.
13. Read the input image and create a blob for DNN processing.
14. Pass the blob through the MobileNet-SSD network.
15. Detect objects with confidence greater than 0.5.
16. Draw bounding boxes and labels around the detected objects.
17. Display the results using Matplotlib.

## Implementation

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# I) ROI Segmentation in an Image using Bitwise AND

# Read the image
image = cv2.imread('3img.jpg')

# Convert BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Display the original image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('on')
plt.show()

# Define the Region of Interest (ROI)
roi = image[100:420, 200:550]

# Create a blank mask
mask = np.zeros_like(image)

# Place the ROI on the mask
mask[100:420, 200:550] = roi

# Perform bitwise AND
segmented_roi = cv2.bitwise_and(image, mask)

# Display the segmented ROI
segmented_roi_rgb = cv2.cvtColor(
    segmented_roi,
    cv2.COLOR_BGR2RGB
)

plt.imshow(segmented_roi_rgb)
plt.title("Segmented ROI")
plt.axis('off')
plt.show()


# II) Handwriting Detection in an Image

# Read the image
image = cv2.imread('your_image_1.jpg')

# Convert BGR to RGB
image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# Display the original image
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')
plt.show()

# Convert image to grayscale
gray_image = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

# Apply Gaussian Blur
blurred_image = cv2.GaussianBlur(
    gray_image,
    (5, 5),
    0
)

# Apply Canny Edge Detection
edges = cv2.Canny(
    blurred_image,
    50,
    150
)

# Display Canny Edge Detection
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis('off')
plt.show()

# Find contours
contours, _ = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# Create a copy of the original image
result_image = image.copy()

# Filter contours and draw bounding boxes
for contour in contours:
    if cv2.contourArea(contour) > 50:
        x, y, w, h = cv2.boundingRect(contour)

        cv2.rectangle(
            result_image,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            2
        )

# Display handwriting detection result
plt.imshow(
    cv2.cvtColor(
        result_image,
        cv2.COLOR_BGR2RGB
    )
)

plt.title("Handwriting Detection")
plt.axis('off')
plt.show()


# III) Object Detection with Labels using MobileNet-SSD

# Set the configuration and weights files
config_file = 'deploy.prototxt'
weights = 'mobilenet_iter_73000.caffemodel'

# Load the pretrained MobileNet-SSD model
net = cv2.dnn.readNetFromCaffe(
    config_file,
    weights
)

# Define class labels
class_labels = {
    0: 'background',
    1: 'aeroplane',
    2: 'bicycle',
    3: 'bird',
    4: 'boat',
    5: 'bottle',
    6: 'bus',
    7: 'car',
    8: 'cat',
    9: 'chair',
    10: 'cow',
    11: 'diningtable',
    12: 'dog',
    13: 'horse',
    14: 'motorbike',
    15: 'person',
    16: 'pottedplant',
    17: 'sheep',
    18: 'sofa',
    19: 'train',
    20: 'tvmonitor'
}

# Read the image
image = cv2.imread('download.webp')

# Get image dimensions
(h, w) = image.shape[:2]

# Convert image to RGB
image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)

# Create a blob for DNN processing
blob = cv2.dnn.blobFromImage(
    image,
    0.007843,
    (300, 300),
    127.5
)

# Set the input and perform detection
net.setInput(blob)
detections = net.forward()

# Process detections
for i in range(detections.shape[2]):

    confidence = detections[0, 0, i, 2]

    if confidence > 0.5:

        index = int(
            detections[0, 0, i, 1]
        )

        label = class_labels[index]

        box = (
            detections[0, 0, i, 3:7]
            * np.array([w, h, w, h])
        )

        (startX, startY, endX, endY) = box.astype("int")

        # Draw bounding box
        cv2.rectangle(
            image_rgb,
            (startX, startY),
            (endX, endY),
            (0, 255, 0),
            2
        )

        # Draw object label
        cv2.putText(
            image_rgb,
            label,
            (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 0, 0),
            2
        )

# Display object detection result
plt.imshow(image_rgb)
plt.title("Object Detection with MobileNet-SSD")
plt.axis("off")
plt.show()
```
## Output

### Original Image

<img width="457" height="537" alt="image" src="https://github.com/user-attachments/assets/25a18fc4-2573-43d8-a746-58c442648206" />

### Segmented ROI

<img width="388" height="513" alt="image" src="https://github.com/user-attachments/assets/13626c2e-8d05-4bcb-be7a-467b79f70f28" />

### Canny Edge Detection

<img width="685" height="289" alt="image" src="https://github.com/user-attachments/assets/072bebc0-b060-40c5-91a3-d37e4f5b587c" />

### Handwriting Detection

<img width="660" height="325" alt="image" src="https://github.com/user-attachments/assets/7723c8de-6c12-4cd8-8996-be24fd1c4a9c" />

### Object Detection with MobileNet-SSD

<img width="468" height="487" alt="image" src="https://github.com/user-attachments/assets/98339ac9-95be-40c5-9803-63d0714ce847" />
