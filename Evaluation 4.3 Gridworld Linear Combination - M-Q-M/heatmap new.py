import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

# Define the image directory
image_dir = Path(__file__).resolve().parent / 'Dollar-Euro' / '0'

# List of image filenames
image_files = [
    'heatmap_DE_0_1_new.png',
    'heatmap_DE_0_2_new.png',
    'heatmap_DE_0_4_new.png'
]

# Create the figure and axes
fig, axes = plt.subplots(1, 3, figsize=(20, 5))
a=[1,2,4]
c=0
# Loop through each image and plot it
for ax, img_file in zip(axes, image_files):
    img_path = image_dir / img_file
    img = mpimg.imread(img_path)
    ax.imshow(img)
    ax.set_title(f"SBF = {a[c]}", fontsize=15)
    c+=1
    ax.axis('off')  # Hide axis ticks

plt.tight_layout()
plt.savefig('DE new heatmap.jpeg', bbox_inches='tight', dpi=600)

plt.show()
