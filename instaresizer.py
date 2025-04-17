import cv2
import numpy as np
import os
from PIL import Image, ImageFilter, ImageEnhance
import argparse


def resize_and_crop(img, target_size):
    """
    Resize with aspect ratio and center crop to target_size.
    """
    
    img_ratio = img.width / img.height
    scale = target_size / img.height if img_ratio > 1 else target_size / img.width

    new_w = int(img.width * scale)
    new_h = int(img.height * scale)
    img = img.resize((new_w, new_h), Image.LANCZOS)

    left = (new_w - target_size) // 2
    top = (new_h - target_size) // 2

    return img.crop((left, top, left + target_size, top + target_size))


def resize_to_square(input_path, output_path, blur=False):
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

        if blur:
            bg = resize_and_crop(image, size)
            bg = bg.filter(ImageFilter.GaussianBlur(50))
            bg = ImageEnhance.Brightness(bg).enhance(0.3)
        else:
            bg = Image.new("RGB", (size, size), (0, 0, 0))

        x_offset = (size - w) // 2
        y_offset = (size - h) // 2
        bg.paste(image, (x_offset, y_offset))

        save_params = {"format": image.format}
        if icc_profile:
            save_params["icc_profile"] = icc_profile

        bg.save(output_path, **save_params)
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

        if blur:
            ret, first_frame = cap.read()
            if not ret:
                print("Error: Could not read the first frame.")
                cap.release()
                return

            # Create background once using first frame
            h_ratio = size / first_frame.shape[0]
            w_ratio = size / first_frame.shape[1]
            scale = max(h_ratio, w_ratio)
            resized = cv2.resize(first_frame, (int(first_frame.shape[1] * scale), int(first_frame.shape[0] * scale)))

            start_x = (resized.shape[1] - size) // 2
            start_y = (resized.shape[0] - size) // 2
            cropped_bg = resized[start_y:start_y + size, start_x:start_x + size]

            blurred_bg = cv2.GaussianBlur(cropped_bg, (99, 99), 0)
            dark_bg = (blurred_bg * 0.3).astype(np.uint8)

            # Reset to first frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if blur:
                frame_overlay = dark_bg.copy()
            else:
                frame_overlay = np.zeros((size, size, 3), dtype=np.uint8)
            
            x_offset = (size - width) // 2
            y_offset = (size - height) // 2
            frame_overlay[y_offset:y_offset + height, x_offset:x_offset + width] = frame
            out.write(frame_overlay)

        cap.release()
        out.release()
        print(f"Video saved to {output_path}")
        
    else:
        print("Unsupported file format.")


def main():
    parser = argparse.ArgumentParser(description="Resize images or videos to square format with optional blur background.")
    parser.add_argument("input", help="Input file path")
    parser.add_argument("output", help="Output file path")
    parser.add_argument("--blur", action="store_true", help="Apply blurred background instead of black")

    args = parser.parse_args()
    resize_to_square(args.input, args.output, blur=args.blur)


if __name__ == "__main__":
    main()
