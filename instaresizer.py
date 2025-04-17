import cv2
import numpy as np
import os
from PIL import Image


def resize_to_square(input_path, output_path):
    input_path = os.path.join(os.getcwd(), input_path)
    output_path = os.path.join(os.getcwd(), output_path)
    
    if not os.path.exists(input_path):
        print("Error: File does not exist.")
        return
    
    file_ext = os.path.splitext(input_path)[1].lower()
    
    if file_ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']:
        image = Image.open(input_path)
        icc_profile = image.info.get("icc_profile")
        w, h = image.size
        size = max(w, h)
        
        square_image = Image.new("RGB", (size, size), (0, 0, 0))
        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        square_image.paste(image, (x_offset, y_offset))
        
        save_params = {"format": image.format}
        if icc_profile:
            save_params["icc_profile"] = icc_profile
        
        square_image.save(output_path, **save_params)
        print(f"Image saved to {output_path}")

    elif file_ext in ['.mp4', '.avi', '.mov', '.mkv', '.wmv']:
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print("Error: Could not open video file.")
            return
        
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        size = max(width, height)
        
        out = cv2.VideoWriter(output_path, fourcc, fps, (size, size))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            square_frame = np.zeros((size, size, 3), dtype=np.uint8)
            
            x_offset = (size - width) // 2
            y_offset = (size - height) // 2
            
            square_frame[y_offset:y_offset + height, x_offset:x_offset + width] = frame
            
            out.write(square_frame)
        
        cap.release()
        out.release()
        print(f"Video saved to {output_path}")

    else:
        print("Unsupported file format.")

def main():
    input_filename = input("Enter the input file name (in the same folder): ")
    output_filename = "output_" + input_filename
    resize_to_square(input_filename, output_filename)

if __name__ == "__main__":
    main()
