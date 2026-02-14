import xml.etree.ElementTree as ET
import numpy as np
import cv2
import os

def decode_cvat_rle(rle_string, box_height, box_width):
    """
    Decodes CVAT RLE string into a 2D binary mask for the bounding box.
    CVAT RLE format: value1, value2, value3... 
    where value1 = count of 0s, value2 = count of 1s, etc.
    """
    if not rle_string:
        return np.zeros((box_height, box_width), dtype=np.uint8)

    # Convert string "1, 2, 3" to list [1, 2, 3]
    rle_counts = [int(x) for x in rle_string.split(',')]
    
    # Total pixels in the bounding box
    total_pixels = box_height * box_width
    
    # Create the flat array
    flat_mask = np.zeros(total_pixels, dtype=np.uint8)
    
    current_idx = 0
    current_val = 0 # CVAT RLE always starts with Background (0)
    
    for count in rle_counts:
        if current_val == 1:
            # Fill with 255 (White) so we can see it, or 1 for logical mask
            flat_mask[current_idx : current_idx + count] = 255
        
        current_idx += count
        current_val = 1 - current_val # Toggle between 0 and 1
        
    # Reshape into the box dimensions
    try:
        return flat_mask.reshape((box_height, box_width))
    except ValueError:
        print(f"Error: RLE size {len(flat_mask)} doesn't match box {box_width}x{box_height}")
        return np.zeros((box_height, box_width), dtype=np.uint8)

def process_cvat_xml(xml_file, output_folder):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Define visible colors (Blue, Green, Red)
    colors = {
        'skin': (0, 255, 0),    # Green
        'default': (0, 0, 255)  # Red
    }

    for image_tag in root.findall('image'):
        file_name = image_tag.get('name')
        
        # 1. Get Full Image Dimensions
        img_w = int(image_tag.get('width'))
        img_h = int(image_tag.get('height'))
        
        # 2. Create Black Canvas for the Full Image
        full_canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)
        
        print(f"Processing: {file_name}")
        
        for mask_tag in image_tag.findall('mask'):
            label = mask_tag.get('label')
            rle_data = mask_tag.get('rle')
            
            # 3. Get Bounding Box Dimensions (Crucial for CVAT!)
            box_top = int(mask_tag.get('top'))
            box_left = int(mask_tag.get('left'))
            box_w = int(mask_tag.get('width'))
            box_h = int(mask_tag.get('height'))
            
            # 4. Decode the RLE into the small box shape
            mini_mask = decode_cvat_rle(rle_data, box_h, box_w)
            
            # 5. Determine Color
            color = colors.get(label, colors['default'])
            
            # 6. Paste the mini-mask onto the full canvas
            # We look at the region of interest (ROI) in the full canvas
            roi = full_canvas[box_top : box_top + box_h, box_left : box_left + box_w]
            
            # Where the mini_mask is White (255), paint the color on the ROI
            # We create a boolean mask where mini_mask is 255
            mask_boolean = mini_mask == 255
            
            # Apply color to the canvas ROI
            roi[mask_boolean] = color
            
            # Update the full canvas with the modified ROI
            full_canvas[box_top : box_top + box_h, box_left : box_left + box_w] = roi

        # Save
        save_path = os.path.join(output_folder, f"{file_name}")
        # Ensure extension
        if not save_path.lower().endswith(('.jpg', '.png')):
            save_path += ".png"
            
        cv2.imwrite(save_path, full_canvas)
        print(f"Saved: {save_path}")

# --- EXECUTION ---
# Change 'annotations.xml' to your actual file name
xml_path = 'annotations.xml' 

if os.path.exists(xml_path):
    process_cvat_xml(xml_path, 'masks')
else:
    print(f"File {xml_path} not found. Please verify the path.")
