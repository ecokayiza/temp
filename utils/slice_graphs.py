import os
import glob
from PIL import Image, ImageOps
import numpy as np

def slice_image(image_path, output_dir):
    filename = os.path.basename(image_path)
    name, ext = os.path.splitext(filename)
    
    try:
        img = Image.open(image_path)
    except Exception as e:
        print(f"Failed to open {image_path}: {e}")
        return

    # Convert to grayscale
    gray = img.convert('L')
    # Invert so ink is white (255) and background is black (0) for easier calculation
    # Assuming white background
    inverted = ImageOps.invert(gray)
    
    # Get data as numpy array
    data = np.array(inverted)
    
    # Calculate horizontal projection (sum of pixels in each row)
    # Rows with content will have high sum. Rows that are empty (white background) will have near 0 sum.
    row_sums = np.sum(data, axis=1)
    
    # Threshold to consider a row as "empty"
    # A purely white row in original image is 0 in inverted.
    # Allow some noise.
    threshold = 100 # Arbitrary small value
    
    is_content = row_sums > threshold
    
    # Find start and end indices of content blocks
    blocks = []
    start = None
    
    # We want to merge blocks that are close together (e.g. title and chart)
    # and separate blocks that are far apart (different charts).
    # But simpler approach: Find all continuous content segments.
    
    for i, has_content in enumerate(is_content):
        if has_content and start is None:
            start = i
        elif not has_content and start is not None:
            blocks.append((start, i))
            start = None
            
    if start is not None:
        blocks.append((start, len(is_content)))
        
    # Now merge blocks that are very close (likely parts of the same chart)
    # Gap threshold: e.g., 10 pixels?
    # The charts in the example seemed to have significant gaps.
    
    merged_blocks = []
    if not blocks:
        print(f"No content found in {filename}")
        return

    current_start, current_end = blocks[0]
    
    merge_threshold = 20 # pixels
    
    for i in range(1, len(blocks)):
        next_start, next_end = blocks[i]
        if next_start - current_end < merge_threshold:
            # Merge
            current_end = next_end
        else:
            # Save current and start new
            merged_blocks.append((current_start, current_end))
            current_start, current_end = next_start, next_end
            
    merged_blocks.append((current_start, current_end))
    
    # Filter out very small blocks (noise)
    min_height = 2
    valid_blocks = [b for b in merged_blocks if (b[1] - b[0]) > min_height]
    
    print(f"Found {len(valid_blocks)} blocks in {filename}")
    
    # Identify Chart Units
    # A unit consists of a Large Block (Body) optionally preceded by a Small Block (Title)
    # and optionally followed by a Small Block (Legend).
    # We use proximity to group them.
    
    chart_slices = []
    min_body_height = 100
    max_gap = 60
    
    # Identify indices of large blocks
    body_indices = [i for i, b in enumerate(valid_blocks) if (b[1] - b[0]) > min_body_height]
    
    if len(body_indices) < 5:
        print(f"Warning: Found only {len(body_indices)} charts, expected at least 5.")
    
    # Initialize starts and ends with the body blocks
    chart_slices = []
    for idx in body_indices:
        chart_slices.append([valid_blocks[idx][0], valid_blocks[idx][1]])
        
    # Distribute small blocks between charts
    # Map of "Next Chart Has Title"
    # 0->1: False
    # 1->2: True
    # 2->3: True
    # 3->4: False
    next_has_title = {0: False, 1: False, 2: False, 3: False}
    
    for i in range(len(body_indices) - 1):
        # Indices of small blocks between body i and body i+1
        start_idx = body_indices[i] + 1
        end_idx = body_indices[i+1]
        
        small_blocks = valid_blocks[start_idx:end_idx]
        
        if not small_blocks:
            continue
            
        if next_has_title.get(i, False):
            # Next chart has title, so the last small block belongs to next chart
            title_block = small_blocks[-1]
            chart_slices[i+1][0] = title_block[0] # Update start of next chart
            
            # The rest belong to current chart
            if len(small_blocks) > 1:
                chart_slices[i][1] = small_blocks[-2][1]
        else:
            # Next chart has NO title, so all small blocks belong to current chart
            chart_slices[i][1] = small_blocks[-1][1]
            
    # Handle the last chart (and any blocks after it)
    last_body_idx = body_indices[-1]
    if last_body_idx + 1 < len(valid_blocks):
        remaining_blocks = valid_blocks[last_body_idx+1:]
        if remaining_blocks:
            chart_slices[-1][1] = remaining_blocks[-1][1]

    # Take top 5 charts (or however many we found)
    top_5 = chart_slices[:5]
    
    for i, (start, end) in enumerate(top_5):
        # Add some padding if possible
        pad = 5
        s = max(0, start - pad)
        e = min(img.height, end + pad)
        
        cropped = img.crop((0, s, img.width, e))
        
        # Save
        slice_filename = f"{name}_slice_{i+1}{ext}"
        slice_path = os.path.join(output_dir, slice_filename)
        cropped.save(slice_path)
        print(f"Saved {slice_path}")

def main():
    input_dir = r'c:\Users\22638\Desktop\sungrow\prepare\AI模式欧洲典型电站分析20250717\2025-06-01_2025-06-30_5096060_empirical_result'
    output_dir = r'c:\Users\22638\Desktop\sungrow\prepare\AI模式欧洲典型电站分析20250717\slices'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    png_files = glob.glob(os.path.join(input_dir, '*.png'))
    
    if not png_files:
        print(f"No png files found in {input_dir}")
        return

    for png_file in png_files:
        slice_image(png_file, output_dir)

if __name__ == "__main__":
    main()
