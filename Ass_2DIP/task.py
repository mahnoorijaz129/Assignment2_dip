import cv2
import numpy as np

# Load the grayscale image
image = cv2.imread('Image.bmp', cv2.IMREAD_GRAYSCALE)

# Define neighboring pixel positions
neighbors = [(-1, -1), (-1, 0), (-1, 1), (0, -1)]

def labelObjects(image):
    # Function to find the root label in the union-find data structure
    def find_root(label):
        while parent[label] != label:
            label = parent[label]
        return label

    # Function to merge two labels in the union-find data structure
    def union(label1, label2):
        root1 = find_root(label1)
        root2 = find_root(label2)
        if root1 != root2:
            parent[root1] = root2

    # Get the image dimensions
    height, width = image.shape

    # Initialize an array to store labels
    labels = np.zeros((height, width), dtype=int)
    parent = list(range(height * width))
    label = 0

    for x in range(height):
        for y in range(width):
            if image[x, y] == 255:
                neighborLabels = []

                # Check neighboring pixels
                for dx, dy in neighbors:
                    if 0 <= x + dx < height and 0 <= y + dy < width:
                        neighborLabel = labels[x + dx, y + dy]
                        if neighborLabel > 0:
                            neighborLabels.append(neighborLabel)

                if not neighborLabels:
                    label += 1
                    labels[x, y] = label
                else:
                    minLabel = min(neighborLabels)
                    labels[x, y] = minLabel
                    for neighborLabel in neighborLabels:
                        if neighborLabel != minLabel:
                            union(minLabel, neighborLabel)

    # Update labels to have root labels
    for x in range(height):
        for y in range(width):
            if image[x, y] == 255:
                labels[x, y] = find_root(labels[x, y])

    return labels, label

# Label objects in the image
labeledImage, numObjects = labelObjects(image)

# Generate random colors for objects
colors = np.random.randint(0, 255, size=(numObjects, 3), dtype=np.uint8)

# Create an output image with colored objects
outputImage = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

for label in range(1, numObjects + 1):
    outputImage[labeledImage == label] = colors[label - 1]

# Display the labeled objects
cv2.imshow('Labeled Objects', outputImage)
cv2.waitKey(0)
cv2.destroyAllWindows()











