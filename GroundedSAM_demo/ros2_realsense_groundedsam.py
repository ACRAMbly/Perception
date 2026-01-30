#!/usr/bin/env python3
"""
Simple ROS2 Jazzy test node for GroundedSAM on RealSense RGB images.
- Subscribes to an RGB topic, runs GroundedSAM per frame, overlays masks, and displays a window.
- No pose estimation; this is only to validate segmentation from ROS images.
"""

import argparse
import os
import cv2
import numpy as np
import distinctipy
import torch
import sys
import pathlib

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from typing import Optional

# Ensure repo root is on sys.path so `GroundedSAM_demo` is importable
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from GroundedSAM_demo.grounded_sam import GroundedSAM


class GroundedSAMROS2Camera(Node):
    def __init__(self,
                 checkpoints_dir: str,
                 grounded_dino_config_dir: str,
                 grounded_dino_use_vitb: bool,
                 box_threshold: float,
                 text_threshold: float,
                 use_yolo_sam: bool,
                 sam_vit_model: str,
                 mask_threshold: Optional[float],
                 prompt_text: str,
                 background: str,
                 segmentor_width_size: Optional[int],
                 rgb_topic: str,
                 publish_vis: bool):
        super().__init__('groundedsam_ros2_camera')

        self.bridge = CvBridge()
        self.prompt_text = prompt_text
        self.background = background
        self.publish_vis = publish_vis

        # Colors for visualization
        self.colors = distinctipy.get_colors(50)

        # Avoid YOLO-SAM mask_threshold=None bug by using -1.0 to keep all masks
        effective_mask_threshold = mask_threshold
        if use_yolo_sam and effective_mask_threshold is None:
            effective_mask_threshold = -1.0

        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.get_logger().info('Loading GroundedSAM...')
        self.grounded_sam = GroundedSAM.load_grounded_sam_model(
            checkpoints_dir=checkpoints_dir,
            grounded_dino_config_dir=grounded_dino_config_dir,
            grounded_dino_use_vitb=grounded_dino_use_vitb,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            use_yolo_sam=use_yolo_sam,
            sam_vit_model=sam_vit_model,
            mask_threshold=effective_mask_threshold,
            prompt_text=prompt_text,
            segmentor_width_size=segmentor_width_size,
            device=device,
        )
        self.get_logger().info(f'GroundedSAM loaded on device={device}. Prompt="{prompt_text}"')

        # Subscriber
        self.rgb_sub = self.create_subscription(Image, rgb_topic, self.rgb_callback, 10)
        self.get_logger().info(f'Subscribed to RGB: {rgb_topic}')

        # Optional publisher for visualization
        if publish_vis:
            self.vis_pub = self.create_publisher(Image, '/groundedsam/visualization', 10)
        else:
            self.vis_pub = None

        cv2.namedWindow('GroundedSAM ROS2', cv2.WINDOW_NORMAL)

        self._frame_idx = 0
        self._last_thresholds = (box_threshold, text_threshold)
        self._debug_dir = None

    def _apply_background(self, frame_rgb: np.ndarray) -> np.ndarray:
        if self.background == 'rgb':
            return frame_rgb.copy()
        if self.background == 'gray':
            return cv2.cvtColor(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
        if self.background == 'black':
            return np.zeros_like(frame_rgb, dtype=np.uint8)
        if self.background == 'white':
            return np.ones_like(frame_rgb, dtype=np.uint8) * 255
        return frame_rgb.copy()

    def _overlay_masks(self, frame: np.ndarray, masks: np.ndarray) -> np.ndarray:
        alpha = 0.33
        for mask_idx, mask in enumerate(masks):
            r = int(255 * self.colors[mask_idx][0])
            g = int(255 * self.colors[mask_idx][1])
            b = int(255 * self.colors[mask_idx][2])
            frame[mask, 0] = alpha * r + (1 - alpha) * frame[mask, 0]
            frame[mask, 1] = alpha * g + (1 - alpha) * frame[mask, 1]
            frame[mask, 2] = alpha * b + (1 - alpha) * frame[mask, 2]
        return frame

    def _draw_boxes(self, frame: np.ndarray, boxes_tensor) -> np.ndarray:
        try:
            boxes = boxes_tensor.detach().cpu().numpy().astype(float)
        except Exception:
            return frame
        for (x1, y1, x2, y2) in boxes:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
        return frame

    def _retry_relaxed(self, frame_rgb: np.ndarray):
        # Temporarily relax thresholds once if nothing is detected
        try:
            gd = self.grounded_sam.ground_dino
            old_box, old_text = gd.box_threshold, gd.text_threshold
            self.get_logger().info(f'No detections. Retrying with box=0.05 text=0.05')
            gd.box_threshold, gd.text_threshold = 0.05, 0.05
            detections = self.grounded_sam.generate_masks(frame_rgb)
            gd.box_threshold, gd.text_threshold = old_box, old_text
            return detections
        except Exception as e:
            self.get_logger().warn(f'Retry failed: {e}')
            return None

    def rgb_callback(self, msg: Image):
        try:
            frame_rgb = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            frame_rgb = np.ascontiguousarray(frame_rgb)
        except CvBridgeError as e:
            self.get_logger().error(f'RGB conversion failed: {e}')
            return

        self._frame_idx += 1
        detections = self.grounded_sam.generate_masks(frame_rgb)
        if detections is None:
            # Also peek at raw GroundingDINO output for debugging
            try:
                boxes_dbg, confs_dbg, phrases_dbg = self.grounded_sam.ground_dino.predict(frame_rgb, self.prompt_text)
                self.get_logger().info(f'Frame {self._frame_idx}: raw DINO boxes={boxes_dbg.shape[0]} (pre-SAM)')
            except Exception:
                boxes_dbg = None
            detections = self._retry_relaxed(frame_rgb)

        masks = None
        if detections is not None:
            try:
                num_boxes = int(detections["boxes"].shape[0])
                num_masks = int(detections["masks"].shape[0])
            except Exception:
                num_boxes = -1
                num_masks = -1
            self.get_logger().info(f'Frame {self._frame_idx}: boxes={num_boxes}, masks={num_masks}')

            if num_masks > 0:
                # Sort masks by box x/y to have stable coloring
                bboxes = detections["boxes"].cpu().numpy()
                masks_t = detections["masks"].squeeze(1).cpu().numpy()
                order = np.lexsort([bboxes[:, 1], bboxes[:, 0]])
                masks = masks_t[order]

        vis = self._apply_background(frame_rgb)
        if masks is not None and masks.shape[0] > 0:
            vis = self._overlay_masks(vis, masks.astype(bool))
        else:
            # Visual hint when nothing is detected; also draw raw DINO boxes if any
            try:
                boxes_dbg, confs_dbg, phrases_dbg = self.grounded_sam.ground_dino.predict(frame_rgb, self.prompt_text)
                if boxes_dbg is not None and boxes_dbg.shape[0] > 0:
                    vis = self._draw_boxes(vis, boxes_dbg)
            except Exception:
                pass
            cv2.putText(vis, 'No detections', (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2, cv2.LINE_AA)

        # Show
        try:
            cv2.imshow('GroundedSAM ROS2', cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)
        except Exception:
            pass

        # Optional publish
        if self.vis_pub is not None:
            try:
                vis_msg = self.bridge.cv2_to_imgmsg(vis, encoding='rgb8')
                self.vis_pub.publish(vis_msg)
            except CvBridgeError as e:
                self.get_logger().error(f'Failed to publish visualization: {e}')


def main():
    parser = argparse.ArgumentParser(description='ROS2 RealSense GroundedSAM test (segmentation only).')

    # Model params
    parser.add_argument('--checkpoints_dir', type=str, default='GroundedSAM_demo/checkpoints',
                        help='Path to model checkpoints root.')
    parser.add_argument('--grounded_dino_config_dir', type=str, default='GroundedSAM_demo/cfg/gdino',
                        help='Path to GroundingDINO config directory.')
    parser.add_argument('--grounded_dino_use_vitb', action='store_true', default=False,
                        help='Use Swin-B variant for GroundingDINO.')
    parser.add_argument('--box_threshold', type=float, default=0.1, help='GroundingDINO box threshold.')
    parser.add_argument('--text_threshold', type=float, default=0.1, help='GroundingDINO text threshold.')
    parser.add_argument('--use_yolo_sam', action='store_true', default=False,
                        help='Use YOLO-SAM (requires checkpoint under checkpoints/yolosam).')
    parser.add_argument('--sam_vit_model', type=str, default='vit_t',
                        help='SAM variant: vit_t/vit_b/vit_l/vit_h or YOLO-SAM .pt file name.')
    parser.add_argument('--mask_threshold', type=float, default=0.01,
                        help='Mask confidence threshold; for YOLO-SAM we keep all by using -1.0 internally if None.')
    parser.add_argument('--prompt_text', type=str, default='objects', help='Detection prompt text.')
    parser.add_argument('--segmentor_width', type=int, default=640, help='Resize width for SAM (keeps aspect).')

    # ROS params
    parser.add_argument('--rgb_topic', type=str, default='/tracy_camera/camera/camera/color/image_raw',
                        help='RGB image topic name.')
    parser.add_argument('--publish_vis', action='store_true', default=True,
                        help='Publish visualization to /groundedsam/visualization.')

    # Visualization
    parser.add_argument('--background', type=str, default='gray', choices=['rgb','gray','black','white'],
                        help='Background for overlay.')
    parser.add_argument('--debug_dir', type=str, default='', help='Optional dir to save RGB frames for debugging.')

    args = parser.parse_args()

    rclpy.init()
    node = None
    try:
        node = GroundedSAMROS2Camera(
            checkpoints_dir=args.checkpoints_dir,
            grounded_dino_config_dir=args.grounded_dino_config_dir,
            grounded_dino_use_vitb=args.grounded_dino_use_vitb,
            box_threshold=args.box_threshold,
            text_threshold=args.text_threshold,
            use_yolo_sam=args.use_yolo_sam,
            sam_vit_model=args.sam_vit_model,
            mask_threshold=args.mask_threshold,
            prompt_text=args.prompt_text,
            background=args.background,
            segmentor_width_size=args.segmentor_width,
            rgb_topic=args.rgb_topic,
            publish_vis=args.publish_vis,
        )
        node.get_logger().info('Started GroundedSAM ROS2 camera test. Press Ctrl+C to exit.')
        rclpy.spin(node)
        # set debug dir if provided
        if args.debug_dir:
            try:
                os.makedirs(args.debug_dir, exist_ok=True)
                node._debug_dir = args.debug_dir
            except Exception:
                pass
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
        import traceback; traceback.print_exc()
    finally:
        try:
            if node is not None and rclpy.ok():
                node.destroy_node()
        except Exception:
            pass
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == '__main__':
    main()
