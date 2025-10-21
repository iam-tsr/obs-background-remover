import cv2
import mediapipe as mp
import numpy as np
import time
import sys
import argparse
import pyvirtualcam
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# Initialize MediaPipe Image Segmenter
BaseOptions = mp.tasks.BaseOptions
ImageSegmenter = mp.tasks.vision.ImageSegmenter
ImageSegmenterOptions = mp.tasks.vision.ImageSegmenterOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Global variables
segmentation_result = None
last_timestamp_ms = 0


def result_callback(result, output_image: mp.Image, timestamp_ms: int):
    """Callback function to receive segmentation results asynchronously."""
    global segmentation_result
    segmentation_result = result


def blur_background(frame, mask, blur_strength=35):
    """Apply blur to background using the segmentation mask."""
    # Ensure blur_strength is odd
    if blur_strength % 2 == 0:
        blur_strength += 1
    
    # Resize mask to match frame dimensions if needed
    if mask.shape[:2] != frame.shape[:2]:
        mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    
    # Normalize mask to [0, 1] range
    mask_normalized = mask.astype(np.float64)
    
    # Expand mask to 3 channels for blending
    mask_3channel = np.stack([mask_normalized] * 3, axis=-1)
    
    # Create blurred version of the entire frame
    blurred_frame = cv2.GaussianBlur(frame, (blur_strength, blur_strength), 0)
    
    # Blend original frame (foreground) with blurred frame (background)
    output = (frame * mask_3channel + blurred_frame * (1 - mask_3channel)).astype(np.uint8)
    
    return output

def get_model_path():
    # When running as Flatpak
    flatpak_path = '/app/share/background-blur/selfie_segmenter_landscape.tflite'
    if os.path.exists(flatpak_path):
        return flatpak_path
    
    # When running locally (development)
    local_path = os.path.join(os.path.dirname(__file__), 'selfie_segmenter_landscape.tflite')
    if os.path.exists(local_path):
        return local_path
    
    raise FileNotFoundError("Could not find selfie_segmenter_landscape.tflite")

def main():
    global segmentation_result
    model_path = get_model_path()
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Virtual Camera with Background Blur')
    parser.add_argument('--input', type=int, default=1, help='Input camera device number (default: 1)')
    parser.add_argument('--model', type=str, default=model_path, help='Path to segmentation model')
    parser.add_argument('--blur', type=int, default=55, help='Blur strength (5-101, default: 55)')
    parser.add_argument('--width', type=int, default=640, help='Output width (default: 640)')
    parser.add_argument('--height', type=int, default=480, help='Output height (default: 480)')
    parser.add_argument('--fps', type=int, default=144, help='Output FPS (default: 144)')
    parser.add_argument('--preview', action='store_true', help='Show preview window')
    args = parser.parse_args()
    
    # Model path
    model_path = args.model
    blur_strength = args.blur
    
    # Ensure blur_strength is odd
    if blur_strength % 2 == 0:
        blur_strength += 1
    
    # Create ImageSegmenter options for live stream mode
    options = ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        output_category_mask=False,
        output_confidence_masks=True,
        result_callback=result_callback
    )
    
    # Open input webcam
    cap = cv2.VideoCapture(args.input)
    if not cap.isOpened():
        print(f"Error: Could not open camera device {args.input}")
        return 1
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)
    
    # Get actual dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"Virtual Camera with Background Blur")
    print(f"Input camera: /dev/video{args.input}")
    print(f"Resolution: {width}x{height} @ {args.fps} FPS")
    print(f"Blur strength: {blur_strength}")
    print(f"Press Ctrl+C to stop")
    
    frame_count = 0
    
    # FPS calculation variables
    fps = 0
    frame_counter = 0
    start_time = time.time()
    
    try:
        # Create virtual camera
        with pyvirtualcam.Camera(width=width, height=height, fps=args.fps) as cam:
            print(f'Virtual camera created: {cam.device}')
            print(f'You can now use this camera in OBS or other applications!')
            
            # Create the image segmenter
            with ImageSegmenter.create_from_options(options) as segmenter:
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        print("Error: Failed to capture frame")
                        break
                    
                    # Convert BGR to RGB
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Create MediaPipe Image
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                    
                    # Calculate timestamp in milliseconds
                    frame_count += 1
                    timestamp_ms = int(time.time() * 1000)
                    
                    # Perform segmentation asynchronously
                    segmenter.segment_async(mp_image, timestamp_ms)
                    
                    # Calculate FPS
                    frame_counter += 1
                    if frame_counter >= 30:
                        end_time = time.time()
                        fps = frame_counter / (end_time - start_time)
                        frame_counter = 0
                        start_time = time.time()
                    
                    # Process and display result if available
                    if segmentation_result is not None:
                        # Get the confidence mask for the person category
                        if len(segmentation_result.confidence_masks) > 0:
                            mask = segmentation_result.confidence_masks[0].numpy_view()
                            
                            # Apply background blur
                            output_frame = blur_background(rgb_frame, mask, blur_strength)
                        else:
                            output_frame = rgb_frame
                    else:
                        output_frame = rgb_frame
                    
                    # Send frame to virtual camera
                    cam.send(output_frame)
                    
                    # Show preview if requested
                    if args.preview:
                        # Convert back to BGR for OpenCV display
                        preview_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
                        cv2.putText(preview_frame, f"FPS: {fps:.1f}", 
                                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(preview_frame, f"Virtual Camera: {cam.device}", 
                                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        cv2.imshow('Virtual Camera Preview', preview_frame)
                        
                        # Handle keyboard input
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            break
                    else:
                        # Small delay to prevent CPU overuse
                        cam.sleep_until_next_frame()
                        
    except KeyboardInterrupt:
        print("\nStopping virtual camera...")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Clean up
        cap.release()
        if args.preview:
            cv2.destroyAllWindows()
        print("Virtual camera stopped.")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
