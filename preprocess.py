#!/usr/bin/env python3
"""
Standalone ROS2 node that:
  1. Subscribes to an RGB image topic.
  2. Captures a single frame.
  3. Applies the full pre-processing pipeline
     (white balance → gamma correction → CLAHE).
  4. Saves the original and enhanced images to:
        pre_processing/original/<frame_id>.png
        pre_processing/enhanced/<frame_id>.png

     Both files share the same <frame_id> so you can always match a pair:
         original/frame_20260306_153042_000001.png
         enhanced/frame_20260306_153042_000001.png

Usage:
    python capture_and_preprocess.py [--topic /camera/color/image_raw]
                                     [--output_dir pre_processing]
                                     [--gamma 0.8]
                                     [--frames 1]          # how many frames to capture
"""

import argparse
import os
import time
from datetime import datetime

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge, CvBridgeError
from rclpy.node import Node
from sensor_msgs.msg import Image


# ---------------------------------------------------------------------------
# Pure-function pre-processing helpers (no ROS dependency)
# ---------------------------------------------------------------------------

def apply_white_balance(img: np.ndarray) -> np.ndarray:
    """Neutralise colour cast using the Gray World algorithm (opencv-contrib)."""
    wb = cv2.xphoto.createGrayworldWB()
    wb.setSaturationThreshold(0.4)
    return wb.balanceWhite(img)


def apply_gamma_correction(img: np.ndarray, gamma: float = 1.2) -> np.ndarray:
    """Brighten / darken mid-tones via a pre-computed Look-Up Table."""
    inv_gamma = 1.0 / gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(img, table)


def apply_clahe(img: np.ndarray) -> np.ndarray:
    """Adaptive local-contrast enhancement on the L channel (LAB colour space)."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)
    enhanced_lab = cv2.merge((l_enhanced, a_ch, b_ch))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2RGB)


def preprocess_rgb(frame: np.ndarray, gamma: float = 0.8) -> np.ndarray:
    """Full pipeline: white balance → gamma correction → CLAHE."""
    step1 = apply_white_balance(frame)
    step2 = apply_gamma_correction(step1, gamma=gamma)
    step3 = apply_clahe(step2)
    return step3


# ---------------------------------------------------------------------------
# ROS2 node
# ---------------------------------------------------------------------------

class CaptureAndPreprocessNode(Node):
    """
    Subscribes to an RGB topic, grabs *frames_to_capture* frames,
    processes each one and saves the original / enhanced pair.
    """

    def __init__(
        self,
        rgb_topic: str,
        output_dir: str,
        frames_to_capture: int,
        gamma: float,
    ) -> None:
        super().__init__("capture_and_preprocess_node")

        self.bridge = CvBridge()
        self.output_dir = output_dir
        self.frames_to_capture = frames_to_capture
        self.gamma = gamma
        self.captured = 0
        self.frame_counter = 0          # increments on every received message

        # Prepare output directories
        self.original_dir = os.path.join(output_dir, "original")
        self.enhanced_dir = os.path.join(output_dir, "enhanced")
        os.makedirs(self.original_dir, exist_ok=True)
        os.makedirs(self.enhanced_dir, exist_ok=True)

        self.get_logger().info(f"Subscribing to: {rgb_topic}")
        self.get_logger().info(
            f"Will capture {frames_to_capture} frame(s)."
        )
        self.get_logger().info(f"Originals  → {self.original_dir}")
        self.get_logger().info(f"Enhanced   → {self.enhanced_dir}")

        self.subscription = self.create_subscription(
            Image,
            rgb_topic,
            self._rgb_callback,
            10,
        )

    # ------------------------------------------------------------------

    def _rgb_callback(self, msg: Image) -> None:
        """Receive a frame, optionally pre-process it, then save the pair."""
        self.frame_counter += 1

        # Already collected enough frames – ignore further messages
        if self.captured >= self.frames_to_capture:
            return

        # Convert ROS Image → numpy (RGB)
        try:
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
            img = np.ascontiguousarray(img, dtype=np.uint8)
        except CvBridgeError as exc:
            self.get_logger().error(f"CvBridge conversion failed: {exc}")
            return

        # Build a shared base filename so original ↔ enhanced pairs are obvious
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        frame_id = f"frame_{timestamp}_{self.frame_counter:06d}"

        # ---- Save original (RGB → BGR for OpenCV imwrite) -----------------
        original_path = os.path.join(self.original_dir, f"{frame_id}.png")
        cv2.imwrite(original_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        self.get_logger().info(f"[{frame_id}] Original saved  → {original_path}")

        # ---- Apply pre-processing -----------------------------------------
        t0 = time.perf_counter()
        try:
            enhanced = preprocess_rgb(img, gamma=self.gamma)
        except Exception as exc:
            self.get_logger().error(f"Pre-processing failed: {exc}")
            self.captured += 1
            self._check_done()
            return
        elapsed_ms = (time.perf_counter() - t0) * 1000
        self.get_logger().info(
            f"[{frame_id}] Pre-processing done in {elapsed_ms:.1f} ms"
        )

        # ---- Save enhanced -------------------------------------------------
        enhanced_path = os.path.join(self.enhanced_dir, f"{frame_id}.png")
        cv2.imwrite(enhanced_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
        self.get_logger().info(f"[{frame_id}] Enhanced  saved  → {enhanced_path}")

        self.captured += 1
        self._check_done()

    def _check_done(self) -> None:
        """Shutdown cleanly once the requested number of frames is captured."""
        if self.captured >= self.frames_to_capture:
            self.get_logger().info(
                f"All {self.frames_to_capture} frame(s) captured. Shutting down."
            )
            # Raise a KeyboardInterrupt-equivalent to break rclpy.spin()
            raise SystemExit


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args=None):
    parser = argparse.ArgumentParser(
        description="Capture a frame from a ROS2 topic, apply pre-processing, "
                    "and save the original / enhanced pair."
    )
    parser.add_argument(
        "--topic",
        type=str,
        default="/camera/camera/color/image_raw",
        help="RGB image topic to subscribe to",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pre_processing",
        help="Root directory for output images (default: pre_processing/)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.8,
        help="Gamma value used in gamma correction (default: 0.8)",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=1,
        help="Number of frames to capture before shutting down (default: 1)",
    )

    parsed, ros_args = parser.parse_known_args()

    rclpy.init(args=ros_args if ros_args else args)

    node = CaptureAndPreprocessNode(
        rgb_topic=parsed.topic,
        output_dir=parsed.output_dir,
        frames_to_capture=parsed.frames,
        gamma=parsed.gamma,
    )

    try:
        rclpy.spin(node)
    except SystemExit:
        # Normal exit triggered by _check_done()
        pass
    except KeyboardInterrupt:
        node.get_logger().info("Interrupted by user.")
    finally:
        try:
            node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    main()
