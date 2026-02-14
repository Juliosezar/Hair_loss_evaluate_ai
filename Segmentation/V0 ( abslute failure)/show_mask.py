import matplotlib.pyplot as plt
import cv2 # or use PIL
import numpy as np

# Load your mask (ensure it is loaded in grayscale mode)
mask = cv2.imread('./masks/001.png', 0) 

# Plot it
plt.imshow(mask)
plt.colorbar() # Shows which color corresponds to 0, 1, 2
plt.show()
