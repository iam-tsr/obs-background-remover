# advanced_background_processor.py
import cv2
import numpy as np
import time
import sys
import argparse
import os
from typing import Optional, Tuple
import torch

# Optional libraries
MEDIAPIPE_AVAILABLE = False
REMBG_AVAILABLE = False
TRANSFORMERS_AVAILABLE = False

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except Exception:
    pass

try:
    from rembg import remove, new_session
    REMBG_AVAILABLE = True
except Exception:
    pass

class BackgroundProcessor:
    """
    Advanced background processing with multiple models and techniques.
    """

    def __init__(self, model_type="mediapipe", quality="medium"):
        self.model_type = model_type
        self.quality = quality
        self.segmenter = None
        self.session = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.bg_subtractor = None
        self._init_model()

    def _init_model(self):
        if self.model_type == "mediapipe" and MEDIAPIPE_AVAILABLE:
            self._init_mediapipe()
        elif self.model_type == "rembg" and REMBG_AVAILABLE:
            self._init_rembg()
        else:
            self._init_opencv_fallback()

    def _init_mediapipe(self):
        # MediaPipe Selfie Segmentation is optimized for real-time person segmentation
        # model_selection=1 uses the landscape model, generally better for full-body/person use-cases
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmenter = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)

    def _init_rembg(self):
        # RMBG v1.4 provides high-quality foreground extraction with strong edge preservation
        # Choose model based on desired quality-speed tradeoff
        model_name = "u2net"
        if self.quality == "high":
            model_name = "isnet-general-use"
        elif self.quality == "fast":
            model_name = "u2netp"
        self.session = new_session(model_name)

    def _init_opencv_fallback(self):
        # Classical background subtraction as a fast fallback when DL models unavailable
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            detectShadows=True, history=500, varThreshold=50
        )

    def _segment_mediapipe(self, frame_bgr: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self.segmenter.process(rgb)
        mask = results.segmentation_mask.astype(np.float32)
        return mask

    def _segment_rembg(self, frame_bgr: np.ndarray) -> np.ndarray:
        # Encode frame to PNG bytes
        _, buffer = cv2.imencode('.png', frame_bgr)
        input_bytes = buffer.tobytes()
        output_bytes = remove(input_bytes, session=self.session)
        # Decode back with alpha
        arr = np.frombuffer(output_bytes, np.uint8)
        out = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        if out is None:
            # Fall back to all-ones mask if decode failed
            h, w = frame_bgr.shape[:2]
            return np.ones((h, w), dtype=np.float32)
        if out.shape[2] == 4:
            alpha = out[:, :, 3].astype(np.float32) / 255.0
            return alpha
        # Fallback if no alpha returned
        gray = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        mask = (gray > 10).astype(np.float32)
        return mask

    def _segment_opencv(self, frame_bgr: np.ndarray) -> np.ndarray:
        fg = self.bg_subtractor.apply(frame_bgr)
        return (fg.astype(np.float32) / 255.0)

    def get_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        try:
            if self.model_type == "mediapipe" and self.segmenter is not None:
                return self._segment_mediapipe(frame_bgr)
            if self.model_type == "rembg" and self.session is not None:
                return self._segment_rembg(frame_bgr)
            return self._segment_opencv(frame_bgr)
        except Exception as e:
            # If segmentation errors, return foreground=1.0 mask to avoid black frames
            h, w = frame_bgr.shape[:2]
            return np.ones((h, w), dtype=np.float32)

class AdvancedMaskOps:
    @staticmethod
    def bilateral(mask: np.ndarray, d=9, sigma_color=75, sigma_space=75) -> np.ndarray:
        m8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        sm = cv2.bilateralFilter(m8, d, sigma_color, sigma_space)
        return sm.astype(np.float32) / 255.0

    @staticmethod
    def morph(mask: np.ndarray, ksize=3, iterations=1) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        m8 = (np.clip(mask, 0, 1) * 255).astype(np.uint8)
        # Close small holes then open to remove noise
        m8 = cv2.morphologyEx(m8, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        m8 = cv2.morphologyEx(m8, cv2.MORPH_OPEN, kernel, iterations=iterations)
        return m8.astype(np.float32) / 255.0

    @staticmethod
    def feather(mask: np.ndarray, radius=5) -> np.ndarray:
        if radius <= 0:
            return np.clip(mask, 0, 1)
        k = radius * 2 + 1
        return cv2.GaussianBlur(mask, (k, k), 0)

class Effects:
    @staticmethod
    def blur_background(frame_bgr: np.ndarray, mask: np.ndarray, blur=55, feather=5) -> np.ndarray:
        if blur % 2 == 0:
            blur += 1
        mask = np.clip(mask, 0, 1)
        mask_f = AdvancedMaskOps.feather(mask, radius=feather)
        if mask_f.shape[:2] != frame_bgr.shape[:2]:
            mask_f = cv2.resize(mask_f, (frame_bgr.shape[1], frame_bgr.shape[0]))
        m3 = np.dstack([mask_f] * 3)
        blurred = cv2.GaussianBlur(frame_bgr, (blur, blur), 0)
        out = (frame_bgr.astype(np.float32) * m3 + blurred.astype(np.float32) * (1 - m3)).astype(np.uint8)
        return out

    @staticmethod
    def replace_background(frame_bgr: np.ndarray, mask: np.ndarray, bg_bgr: Optional[np.ndarray], color=(0, 255, 0)) -> np.ndarray:
        mask = np.clip(mask, 0, 1)
        if bg_bgr is None:
            bg_bgr = np.full_like(frame_bgr, color, dtype=np.uint8)
        else:
            if bg_bgr.shape[:2] != frame_bgr.shape[:2]:
                bg_bgr = cv2.resize(bg_bgr, (frame_bgr.shape[1], frame_bgr.shape[0]))
        m3 = np.dstack([mask] * 3)
        out = (frame_bgr.astype(np.float32) * m3 + bg_bgr.astype(np.float32) * (1 - m3)).astype(np.uint8)
        return out

def find_available_cameras(max_idx=5):
    cams = []
    for i in range(max_idx):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                cams.append(i)
            cap.release()
    return cams

def open_camera(index, w, h, fps):
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_V4L2, cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, fps)
            ret, _ = cap.read()
            if ret:
                return cap
            cap.release()
    # Fallback to any working camera
    avail = find_available_cameras()
    if avail:
        cap = cv2.VideoCapture(avail[0], cv2.CAP_DSHOW if sys.platform.startswith("win") else cv2.CAP_ANY)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            cap.set(cv2.CAP_PROP_FPS, fps)
            return cap
    raise RuntimeError("No working cameras found")

def main():
    parser = argparse.ArgumentParser(description="Advanced Background Processor")
    parser.add_argument("--input", type=int, default=0, help="Camera index (default: 0)")
    parser.add_argument("--model", choices=["mediapipe", "rembg", "opencv"], default="mediapipe", help="Segmentation model")
    parser.add_argument("--quality", choices=["fast", "medium", "high"], default="medium", help="Quality setting")
    parser.add_argument("--effect", choices=["blur", "replace", "green"], default="blur", help="Background effect")
    parser.add_argument("--background", type=str, default=None, help="Background image path for replace effect")
    parser.add_argument("--blur", type=int, default=55, help="Blur kernel (odd number)")
    parser.add_argument("--feather", type=int, default=5, help="Edge feather radius")
    parser.add_argument("--width", type=int, default=1280, help="Width")
    parser.add_argument("--height", type=int, default=720, help="Height")
    parser.add_argument("--fps", type=int, default=30, help="FPS")
    parser.add_argument("--save", action="store_true", help="Save output to advanced_output.avi")
    args = parser.parse_args()

    # Validate availability of chosen model
    if args.model == "mediapipe" and not MEDIAPIPE_AVAILABLE:
        print("MediaPipe not installed; falling back to OpenCV.")
        args.model = "opencv"
    if args.model == "rembg" and not REMBG_AVAILABLE:
        print("rembg not installed; falling back to MediaPipe or OpenCV.")
        args.model = "mediapipe" if MEDIAPIPE_AVAILABLE else "opencv"

    # Initialize processor and camera
    processor = BackgroundProcessor(args.model, args.quality)
    cap = open_camera(args.input, args.width, args.height, args.fps)

    # Load background if needed
    bg_img = None
    if args.effect == "replace" and args.background and os.path.exists(args.background):
        bg_img = cv2.imread(args.background)

    # Prepare writer
    writer = None
    if args.save:
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter("advanced_output.avi", fourcc, args.fps, (args.width, args.height))

    # UI info
    print("Advanced Background Processor")
    print(f"Model={args.model}  Quality={args.quality}  Effect={args.effect}")
    print("Controls: Q=Quit  S=Toggle effect  B=Cycle blur  E=Cycle effect  F=Toggle feather")

    # Runtime toggles
    effect_enabled = True
    effects_cycle = ["blur", "green", "replace"]
    blur_options = [15, 35, 55, 75, 95]
    blur_idx = min(2, max(0, blur_options.index(args.blur) if args.blur in blur_options else 2))
    feather_on = True

    # FPS tracking
    fps_count, fps_time, fps_val = 0, time.time(), 0.0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for selfie view
            frame = cv2.flip(frame, 1)

            out_frame = frame
            if effect_enabled:
                # Segmentation
                mask = processor.get_mask(frame)

                # Post-process mask
                mask = AdvancedMaskOps.bilateral(mask, d=7, sigma_color=60, sigma_space=60)
                mask = AdvancedMaskOps.morph(mask, ksize=3, iterations=1)
                if feather_on:
                    mask = AdvancedMaskOps.feather(mask, radius=args.feather)

                # Effects
                if args.effect == "blur":
                    out_frame = Effects.blur_background(frame, mask, blur=blur_options[blur_idx], feather=args.feather)
                elif args.effect == "green":
                    out_frame = Effects.replace_background(frame, mask, bg_bgr=None, color=(0, 255, 0))
                elif args.effect == "replace":
                    out_frame = Effects.replace_background(frame, mask, bg_bgr=bg_img)

            # FPS overlay
            fps_count += 1
            if fps_count >= 30:
                now = time.time()
                fps_val = fps_count / (now - fps_time)
                fps_time, fps_count = now, 0

            cv2.putText(out_frame, f"FPS: {fps_val:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            cv2.putText(out_frame, f"Model={args.model} Effect={args.effect} Blur={blur_options[blur_idx]} Feather={'ON' if feather_on else 'OFF'}",
                        (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)

            cv2.imshow("Advanced Background Processor", out_frame)
            if writer is not None:
                writer.write(out_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                effect_enabled = not effect_enabled
            elif key == ord('b'):
                blur_idx = (blur_idx + 1) % len(blur_options)
            elif key == ord('e'):
                # Cycle effect
                cur_i = effects_cycle.index(args.effect)
                args.effect = effects_cycle[(cur_i + 1) % len(effects_cycle)]
            elif key == ord('f'):
                feather_on = not feather_on

    finally:
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    sys.exit(main())
