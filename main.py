#!/usr/bin/env python3
"""
ROS2 Jazzy node for FoundationPose with RealSense camera integration.
Subscribes to RGB, depth, and camera info topics to perform object detection and pose estimation.
"""

import argparse
import cv2
import distinctipy
import os
import numpy as np
import torch
import trimesh
import logging
from typing import Optional
import nvdiffrast.torch as dr

# ROS2 imports
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
from std_msgs.msg import Header
from cv_bridge import CvBridge, CvBridgeError
from tf2_ros import TransformBroadcaster

# Your existing imports
from GroundedSAM_demo.grounded_sam import GroundedSAM
from estimater import FoundationPose
from Utils import draw_xyz_axis, draw_posed_3d_box

# Constants
BOP_DEPTH_SCALE = 1.0
ZFAR = np.inf


class FoundationPoseROS2Node(Node):
    """
    ROS2 node for real-time object detection and 6D pose estimation using
    GroundedSAM and FoundationPose with RealSense camera.
    """
    
    def __init__(self,
                 checkpoints_dir: str,
                 grounded_dino_config_dir: str,
                 grounded_dino_use_vitb: bool,
                 box_threshold: float,
                 text_threshold: float,
                 use_yolo_sam: bool,
                 sam_vit_model: str,
                 mask_threshold: float,
                 prompt_text: str,
                 background: str,
                 mesh_path: str,
                 mesh_obj_id: int,
                 debug_dir: str,
                 debug: int,
                 rgb_topic: str = '/camera/color/image_raw',
                 depth_topic: str = '/camera/depth/image_rect_raw',
                 camera_info_topic: str = '/camera/color/camera_info',
                 publish_pose: bool = True,
                 publish_tf: bool = True):
        
        super().__init__('foundationpose_node')
        
        # Initialize CvBridge
        self.bridge = CvBridge()
        
        # Configuration
        self.prompt_text = prompt_text
        self.background = background
        self.debug = debug
        self.debug_dir = debug_dir
        self.publish_pose = publish_pose
        self.publish_tf = publish_tf
        self.mesh_obj_id = mesh_obj_id
        
        # State variables
        self.cam_K = None
        self.color_image = None
        self.depth_image = None
        self.camera_frame_id = "camera_color_optical_frame"
        self.latest_pose = None
        # L1 bounding-box consistency: remembers the box from the previous frame
        # so the tracker always stays on the spatially closest candidate.
        self.last_bbox: Optional[torch.Tensor] = None
        
        # Synchronization
        self.color_received = False
        self.depth_received = False
        self.camera_info_received = False
        
        # Colors for visualization
        self.colors = distinctipy.get_colors(50)
        
        # Frame counter
        self.frame_count = 0
        
        self.get_logger().info("Initializing FoundationPose models...")
        
        # Load GroundedSAM model
        self.grounded_sam = self._load_grounded_sam(
            checkpoints_dir=checkpoints_dir,
            grounded_dino_config_dir=grounded_dino_config_dir,
            grounded_dino_use_vitb=grounded_dino_use_vitb,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            use_yolo_sam=use_yolo_sam,
            sam_vit_model=sam_vit_model,
            mask_threshold=mask_threshold,
            prompt_text=prompt_text
        )
        
        # Load FoundationPose estimator
        self.est, self.mesh = self._load_foundation_pose(
            mesh_path=mesh_path,
            mesh_obj_id=mesh_obj_id,
            debug_dir=debug_dir,
            debug=debug
        )
        
        self.get_logger().info("Models loaded successfully!")
        
        # ROS2 Subscriptions
        self.rgb_sub = self.create_subscription(
            Image,
            rgb_topic,
            self.rgb_callback,
            10
        )
        
        self.depth_sub = self.create_subscription(
            Image,
            depth_topic,
            self.depth_callback,
            10
        )
        
        self.camera_info_sub = self.create_subscription(
            CameraInfo,
            camera_info_topic,
            self.camera_info_callback,
            10
        )
        
        # ROS2 Publishers
        if self.publish_pose:
            self.pose_pub = self.create_publisher(
                PoseStamped,
                '/foundationpose/object_pose',
                10
            )
        
        if self.publish_tf:
            self.tf_broadcaster = TransformBroadcaster(self)
        
        # Visualization publisher (optional)
        self.vis_pub = self.create_publisher(
            Image,
            '/foundationpose/visualization',
            10
        )
        
        # Create debug directories
        if self.debug >= 2:
            os.makedirs(f'{self.debug_dir}/masks', exist_ok=True)
            os.makedirs(f'{self.debug_dir}/depth_images', exist_ok=True)
            os.makedirs(f'{self.debug_dir}/pose_vis', exist_ok=True)
        
        # Create OpenCV window for visualization
        cv2.namedWindow("FoundationPose ROS2", cv2.WINDOW_NORMAL)
        
        self.get_logger().info(f"FoundationPose node initialized!")
        self.get_logger().info(f"Subscribed to:")
        self.get_logger().info(f"  RGB: {rgb_topic}")
        self.get_logger().info(f"  Depth: {depth_topic}")
        self.get_logger().info(f"  Camera Info: {camera_info_topic}")
        self.get_logger().info(f"Prompt text: '{self.prompt_text}'")
    
    def _load_grounded_sam(self, checkpoints_dir, grounded_dino_config_dir,
                          grounded_dino_use_vitb, box_threshold, text_threshold,
                          use_yolo_sam, sam_vit_model, mask_threshold, prompt_text):
        """Load GroundedSAM model."""
        grounded_sam = GroundedSAM.load_grounded_sam_model(
            checkpoints_dir=checkpoints_dir,
            grounded_dino_config_dir=grounded_dino_config_dir,
            grounded_dino_use_vitb=grounded_dino_use_vitb,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            use_yolo_sam=use_yolo_sam,
            sam_vit_model=sam_vit_model,
            mask_threshold=mask_threshold,
            prompt_text=prompt_text,
            segmentor_width_size=None,
            device=None
        )
        return grounded_sam
    
    def _load_foundation_pose(self, mesh_path, mesh_obj_id, debug_dir, debug):
        """Load FoundationPose estimator and mesh."""
        # Initialize with placeholder mesh
        glctx = dr.RasterizeCudaContext()
        mesh_tmp = trimesh.primitives.Box(extents=np.ones((3)), transform=np.eye(4)).to_mesh()
        
        est = FoundationPose(
            model_pts=mesh_tmp.vertices.copy(),
            model_normals=mesh_tmp.vertex_normals.copy(),
            symmetry_tfs=None,
            mesh=mesh_tmp,
            scorer=None,
            refiner=None,
            glctx=glctx,
            debug_dir=debug_dir,
            debug=debug
        )
        
        # Load actual mesh if provided
        loaded_mesh = None
        if mesh_path is not None:
            loaded_mesh = self._load_mesh_file(mesh_path, mesh_obj_id, est)
        
        return est, loaded_mesh
    
    def _load_mesh_file(self, mesh_path, mesh_obj_id, est):
        """Load mesh file and reset estimator."""
        candidate_path = mesh_path
        
        if os.path.isdir(mesh_path):
            if mesh_obj_id is not None:
                patterns = [
                    f"obj_{mesh_obj_id:06d}.ply",
                    f"obj_{mesh_obj_id:06d}.obj",
                    f"{mesh_obj_id}.ply",
                    f"{mesh_obj_id}.obj",
                ]
                for p in patterns:
                    test_path = os.path.join(mesh_path, p)
                    if os.path.exists(test_path):
                        candidate_path = test_path
                        break
            
            if os.path.isdir(candidate_path):
                for fname in os.listdir(candidate_path):
                    if fname.lower().endswith(('.ply', '.obj')):
                        candidate_path = os.path.join(candidate_path, fname)
                        break
        
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(f"Mesh file not found: {candidate_path}")
        
        try:
            loaded_mesh = trimesh.load(candidate_path)
            loaded_mesh.vertices *= 1e-3  # Convert mm to meters
            
            symmetry_tfs = np.array([np.eye(4)])
            est.reset_object(
                model_pts=loaded_mesh.vertices.copy(),
                model_normals=loaded_mesh.vertex_normals.copy(),
                symmetry_tfs=symmetry_tfs,
                mesh=loaded_mesh
            )
            
            self.get_logger().info(f"Loaded mesh: {candidate_path} ({len(loaded_mesh.vertices)} vertices)")
            return loaded_mesh
            
        except Exception as e:
            raise RuntimeError(f"Failed to load mesh at {candidate_path}: {e}")
    
    def camera_info_callback(self, msg: CameraInfo):
        """Extract camera intrinsics from CameraInfo message."""
        try:
            K = np.array(msg.k, dtype=np.float32).reshape(3, 3)
            self.cam_K = K
            self.camera_frame_id = msg.header.frame_id
            
            if not self.camera_info_received:
                self.get_logger().info(f"Camera intrinsics received:\n{K}")
                self.camera_info_received = True
                
        except Exception as e:
            self.get_logger().error(f"Failed to parse CameraInfo: {e}")
    
    def rgb_callback(self, msg: Image):
        """Process RGB image from camera."""
        try:
            self.color_image = None
            # Convert to RGB numpy array
            img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
            img = np.asarray(img, dtype=np.uint8)
            img = np.ascontiguousarray(img)
            self.color_image = img
            self.color_received = True
            
        except CvBridgeError as e:
            self.get_logger().error(f"RGB conversion failed: {e}")
    
    def depth_callback(self, msg: Image):
        """Process depth image from camera and trigger processing."""
        try:
            self.depth_image = None
            # Convert depth to float32 in meters
            depth = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            
            if depth.dtype == np.uint16:
                # RealSense typically publishes depth in mm as uint16
                depth_m = depth.astype(np.float32) * 1e-3
            else:
                depth_m = depth.astype(np.float32)
            
            # Sanitize depth values
            depth_m[depth_m < 0.001] = 0
            depth_m[depth_m > ZFAR] = 0
            
            self.depth_image = depth_m
            self.depth_received = True
            
            # Process frame when all data is available
            self.process_frame()
            
        except CvBridgeError as e:
            self.get_logger().error(f"Depth conversion failed: {e}")
    
    def process_frame(self):
        """Main processing function - runs detection and pose estimation."""
        # Check if all required data is available
        if not (self.color_received and self.depth_received and self.camera_info_received):
            return
        
        if self.color_image is None or self.depth_image is None or self.cam_K is None:
            return
        
        self.frame_count += 1
        rgb_frame = self.color_image.copy()
        depth = self.depth_image.copy()
        #depth = cv2.resize(depth, (640, 480), interpolation=cv2.INTER_NEAREST)
        
        # Run object detection with L1 bounding-box consistency.
        # _get_consistent_mask() queries GroundedSAM with lowered thresholds to
        # obtain many candidate boxes, then picks the one whose position is
        # closest (L1 distance) to the box selected in the previous frame.
        consistent_mask, consistent_score = self._get_consistent_mask(rgb_frame)
        masks = None
        masks_scores = None

        if consistent_mask is not None:
            # Morphological cleanup on the L1-selected mask
            kernel = np.ones((3, 3), np.uint8)
            m_u8 = (consistent_mask.astype(np.uint8) * 255)
            m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_OPEN, kernel)
            m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, kernel)
            cleaned_mask = m_u8.astype(bool)

            masks = np.array([cleaned_mask])
            masks_scores = np.array([consistent_score])

            self.get_logger().info(
                f"Frame {self.frame_count}: L1-consistent detection "
                f"(score={consistent_score:.3f})"
            )

            # Debug: save the selected mask
            if self.debug >= 2:
                mask_img = (cleaned_mask * 255).astype(np.uint8)
                cv2.imwrite(
                    f'{self.debug_dir}/masks/frame_{self.frame_count:06d}_mask.png',
                    mask_img
                )
        
        # Create visualization frame
        frame = self._create_visualization_background(rgb_frame, masks, masks_scores)
        
        # Pose estimation
        pose = None
        if masks is not None and masks.shape[0] > 0 and self.mesh is not None:
            most_confident_mask = masks[0]
            
            # Debug: Save depth image
            if self.debug >= 2:
                depth_vis = (depth * 1000).astype(np.uint16)
                cv2.imwrite(
                    f'{self.debug_dir}/depth_images/frame_{self.frame_count:06d}_depth.png',
                    depth_vis
                )
            
            try:
                #print("image size: {}, depth size: {} and mask shape: {}".format(rgb_frame.shape, depth.shape, most_confident_mask.shape))
                #print("image dtype: {}, depth dtype: {} and mask dtype: {}".format(rgb_frame.dtype, depth.dtype, most_confident_mask.dtype))
                #depth = depth[:, :640]  # TODO which side to crop?

                pose = self._estimate_pose(
                    mask=most_confident_mask,
                    depth=depth,
                    rgb=rgb_frame
                )
                
                if pose is not None:
                    self.latest_pose = pose
                    self.get_logger().info(f"Pose estimated successfully")
                    
                    # Publish pose as ROS message
                    if self.publish_pose:
                        self._publish_pose(pose)
                    
                    # Publish TF
                    if self.publish_tf:
                        self._publish_tf(pose)
                        
            except Exception as e:
                self.get_logger().error(f"Pose estimation failed: {e}")
        
        # Visualize pose on frame
        if pose is not None and self.mesh is not None:
            frame = self._visualize_pose(frame, pose)
        
        # Display frame
        cv2.imshow("FoundationPose ROS2", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        cv2.waitKey(1)
        
        # Publish visualization
        try:
            vis_msg = self.bridge.cv2_to_imgmsg(frame, encoding='rgb8')
            self.vis_pub.publish(vis_msg)
        except CvBridgeError as e:
            self.get_logger().error(f"Failed to publish visualization: {e}")
    
    def _create_visualization_background(self, rgb_frame, masks, masks_scores):
        """Create visualization with colored mask overlays."""
        if self.background == "rgb":
            frame = rgb_frame.copy()
        elif self.background == "gray":
            frame = cv2.cvtColor(
                cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY),
                cv2.COLOR_GRAY2RGB
            )
        elif self.background == "black":
            frame = np.zeros_like(rgb_frame, dtype=np.uint8)
        elif self.background == "white":
            frame = np.ones_like(rgb_frame, dtype=np.uint8) * 255
        else:
            frame = rgb_frame.copy()
        
        if masks is not None:
            alpha = 0.3
            for mask_idx, mask in enumerate(masks):
                r = int(255 * self.colors[mask_idx][0])
                g = int(255 * self.colors[mask_idx][1])
                b = int(255 * self.colors[mask_idx][2])
                frame[mask, 0] = alpha * r + (1 - alpha) * frame[mask, 0]
                frame[mask, 1] = alpha * g + (1 - alpha) * frame[mask, 1]
                frame[mask, 2] = alpha * b + (1 - alpha) * frame[mask, 2]
        
        return frame
        
    # ------------------------------------------------------------------
    # L1 bounding-box consistency
    # ------------------------------------------------------------------

    def _get_consistent_mask(self, rgb_frame: np.ndarray):
        """Return the mask for the temporally consistent detection.

        GroundingDINO is queried with *lowered* thresholds (0.1 / 0.1) so that
        it returns many candidate boxes even when the object is partially
        occluded.  The winning box is then chosen by:

        * **Frame 0** (or after a full detection loss): the box with the
          highest SAM mask confidence – i.e. the default behaviour.
        * **Frame N**: the box whose L1 distance to the *previous* frame's
          winning box is smallest, preventing the tracker from jumping to a
          visually similar but spatially distant object.

        Returns:
            (mask_bool_array, mask_score) – both are ``None`` when no
            detections are found at all.
        """
        # Use the thresholds that were configured at startup.
        detections = self.grounded_sam.generate_masks(rgb_frame)

        if detections is None or detections["masks"].shape[0] == 0:
            # No detections at all – reset memory so the next hit starts fresh.
            self.last_bbox = None
            return None, None

        boxes = detections["boxes"]                              # [N, 4] xyxy
        masks_arr = detections["masks"].squeeze(1).cpu().numpy()  # [N, H, W]
        masks_scores = detections["masks_scores"].cpu().numpy()   # [N]

        # ---- Frame 0: highest-confidence box ----
        if self.last_bbox is None:
            best_idx = int(np.argmax(masks_scores))
            self.last_bbox = boxes[best_idx].detach()
            return masks_arr[best_idx], float(masks_scores[best_idx])

        # ---- Frame N: minimum L1 distance to previous box ----
        last = self.last_bbox.to(boxes.device)
        l1_distances = torch.abs(boxes - last).sum(dim=1)        # [N]
        best_idx = int(torch.argmin(l1_distances).item())
        self.last_bbox = boxes[best_idx].detach()
        return masks_arr[best_idx], float(masks_scores[best_idx])

    def _estimate_pose(self, mask, depth, rgb):
        torch.cuda.set_device(0)
        self.est.to_device('cuda:0')
        self.est.glctx = dr.RasterizeCudaContext(0)

        # Enforce consistency
        # if rgb.shape != depth.shape:
        #     depth = cv2.resize(
        #         depth
        #         (rgb.shape[1], rgb.shape[0]),
        #         interpolation=cv2.INTER_NEAREST
        #     )
        print(f"depth shape: {depth.shape}, rgb shape: {rgb.shape}, mask shape: {mask.shape}, K shape: {self.cam_K.shape}")
        pose = self.est.register(
            K=self.cam_K.astype(np.float64),
            rgb=rgb,
            depth=depth.astype(np.float64),
            ob_mask=mask,
            ob_id=self.mesh_obj_id,
            iteration=5
        )
        print("Estimated pose:\n", pose)
        return pose

    def _visualize_pose(self, color, pose):
        """Visualize pose with 3D axis and bounding box."""
        try:
            pose_np = pose.detach().cpu().numpy() if hasattr(pose, 'detach') else np.asarray(pose)
            
            # Calculate bounding box
            to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
            bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
            center_pose = pose_np @ np.linalg.inv(to_origin)
            
            # Draw 3D axis
            axis_scale = float(np.max(extents)) if np.max(extents) > 0 else 0.1
            vis = draw_xyz_axis(
                color,
                ob_in_cam=center_pose,
                scale=axis_scale,
                K=self.cam_K,
                thickness=3,
                transparency=0,
                is_input_rgb=True
            )
            
            # Draw 3D bounding box
            vis = draw_posed_3d_box(
                self.cam_K,
                img=vis,
                ob_in_cam=center_pose,
                bbox=bbox,
                line_color=(0, 255, 0),
                linewidth=2
            )
            
            # Save debug visualization
            if self.debug >= 2:
                cv2.imwrite(
                    f'{self.debug_dir}/pose_vis/frame_{self.frame_count:06d}.png',
                    vis[..., ::-1]
                )
            
            return vis
            
        except Exception as e:
            self.get_logger().error(f"Visualization error: {e}")
            return color
    
    def _publish_pose(self, pose):
        """Publish pose as PoseStamped message."""
        try:
            pose_np = pose.detach().cpu().numpy() if hasattr(pose, 'detach') else np.asarray(pose)
            
            pose_msg = PoseStamped()
            pose_msg.header = Header()
            pose_msg.header.stamp = self.get_clock().now().to_msg()
            pose_msg.header.frame_id = self.camera_frame_id
            
            # Translation
            pose_msg.pose.position.x = float(pose_np[0, 3])
            pose_msg.pose.position.y = float(pose_np[1, 3])
            pose_msg.pose.position.z = float(pose_np[2, 3])
            
            # Rotation (convert rotation matrix to quaternion)
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_matrix(pose_np[:3, :3])
            quat = rot.as_quat()  # Returns [x, y, z, w]
            
            pose_msg.pose.orientation.x = float(quat[0])
            pose_msg.pose.orientation.y = float(quat[1])
            pose_msg.pose.orientation.z = float(quat[2])
            pose_msg.pose.orientation.w = float(quat[3])
            
            self.pose_pub.publish(pose_msg)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish pose: {e}")
    
    def _publish_tf(self, pose):
        """Publish pose as TF transform."""
        try:
            pose_np = pose.detach().cpu().numpy() if hasattr(pose, 'detach') else np.asarray(pose)
            
            t = TransformStamped()
            t.header.stamp = self.get_clock().now().to_msg()
            t.header.frame_id = self.camera_frame_id
            t.child_frame_id = f'object_{self.mesh_obj_id}' if self.mesh_obj_id else 'detected_object'
            
            # Translation
            t.transform.translation.x = float(pose_np[0, 3])
            t.transform.translation.y = float(pose_np[1, 3])
            t.transform.translation.z = float(pose_np[2, 3])
            
            # Rotation
            from scipy.spatial.transform import Rotation
            rot = Rotation.from_matrix(pose_np[:3, :3])
            quat = rot.as_quat()
            
            t.transform.rotation.x = float(quat[0])
            t.transform.rotation.y = float(quat[1])
            t.transform.rotation.z = float(quat[2])
            t.transform.rotation.w = float(quat[3])
            
            self.tf_broadcaster.sendTransform(t)
            
        except Exception as e:
            self.get_logger().error(f"Failed to publish TF: {e}")


def main(args=None):
    """Main entry point for ROS2 node."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="ROS2 FoundationPose with RealSense")
    
    # Model parameters
    parser.add_argument("--checkpoints_dir", type=str, 
                       default="GroundedSAM_demo/checkpoints",
                       help="Path to model checkpoints")
    parser.add_argument("--grounded_dino_config_dir", type=str,
                       default="GroundedSAM_demo/cfg/gdino",
                       help="Path to GroundedDino config")
    parser.add_argument("--grounded_dino_use_vitb", action="store_true",
                       default=False,
                       help="Use ViT-B backbone for GroundingDino")
    parser.add_argument("--box_threshold", type=float, default=0.3,
                       help="Box confidence threshold")
    parser.add_argument("--text_threshold", type=float, default=0.3,
                       help="Text matching threshold")
    parser.add_argument("--use_yolo_sam", action="store_true", default=True,
                       help="Use YOLO SAM implementation")
    parser.add_argument("--sam_vit_model", type=str, default="sam2.1_l.pt",
                       help="SAM model variant")
    parser.add_argument("--mask_threshold", type=float, default=0.01,
                       help="Mask confidence threshold")
    parser.add_argument("--prompt_text", type=str, default="objects",
                       help="Detection prompt text")
    parser.add_argument("--background", type=str, default="gray",
                       choices=["rgb", "gray", "black", "white"],
                       help="Visualization background")
    
    # Mesh parameters
    parser.add_argument("--mesh_path", type=str, required=True,
                       help="Path to object mesh file or directory")
    parser.add_argument("--mesh_obj_id", type=int, default=None,
                       help="Object ID for mesh resolution")
    
    # Debug parameters
    parser.add_argument("--debug", type=int, default=2,
                       help="Debug level (0-3)")
    parser.add_argument("--debug_dir", type=str, default="debug",
                       help="Debug output directory")
    
    # ROS2 topics
    parser.add_argument("--rgb_topic", type=str,
                       default="/camera/camera/color/image_raw",
                       help="RGB image topic")
    parser.add_argument("--depth_topic", type=str,
                       default="/camera/camera/aligned_depth_to_color/image_raw",
                       help="Depth image topic")
    parser.add_argument("--camera_info_topic", type=str,
                       default="/camera/camera/color/camera_info",
                       help="Camera info topic")
    
    # Publishing options
    parser.add_argument("--publish_pose", action="store_true", default=True,
                       help="Publish pose as PoseStamped")
    parser.add_argument("--publish_tf", action="store_true", default=True,
                       help="Publish pose as TF transform")
    
    parsed_args = parser.parse_args()
    
    # Initialize ROS2
    rclpy.init(args=args)
    # Sanity check: ensure ROS2 message types are loaded (not ROS1)
    try:
        if not hasattr(Image, '_TYPE_SUPPORT') or not hasattr(CameraInfo, '_TYPE_SUPPORT'):
            print("Error: sensor_msgs appear to be ROS 1 types without _TYPE_SUPPORT.\n"
                  "Ensure you have sourced your ROS 2 environment (e.g., /opt/ros/jazzy/setup.bash)\n"
                  "after any ROS 1 setup, and run with a Python that sees ROS 2 packages.")
            return
    except Exception:
        # If import resolution fails, surface a clear hint
        print("Error checking ROS message type support. Verify ROS 2 environment and PYTHONPATH.")
        return
    
    try:
        # Create and run node
        node = FoundationPoseROS2Node(
            checkpoints_dir=parsed_args.checkpoints_dir,
            grounded_dino_config_dir=parsed_args.grounded_dino_config_dir,
            grounded_dino_use_vitb=parsed_args.grounded_dino_use_vitb,
            box_threshold=parsed_args.box_threshold,
            text_threshold=parsed_args.text_threshold,
            use_yolo_sam=parsed_args.use_yolo_sam,
            sam_vit_model=parsed_args.sam_vit_model,
            mask_threshold=parsed_args.mask_threshold,
            prompt_text=parsed_args.prompt_text,
            background=parsed_args.background,
            mesh_path=parsed_args.mesh_path,
            mesh_obj_id=parsed_args.mesh_obj_id,
            debug_dir=parsed_args.debug_dir,
            debug=parsed_args.debug,
            rgb_topic=parsed_args.rgb_topic,
            depth_topic=parsed_args.depth_topic,
            camera_info_topic=parsed_args.camera_info_topic,
            publish_pose=parsed_args.publish_pose,
            publish_tf=parsed_args.publish_tf
        )
        
        node.get_logger().info("Node started! Press Ctrl+C to exit.")
        rclpy.spin(node)
        
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Guard node cleanup to avoid UnboundLocalError
        try:
            if 'node' in locals() and node is not None:
                if rclpy.ok():
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
#python ./main.py  --mesh_path demo_data/models --mesh_obj_id 2 --prompt_text "Objects" --background gray --rgb_topic /tracy_camera/camera/camera/color/image_raw --depth_topic /tracy_camera/camera/camera/depth/image_rect_raw --camera_info_topic /tracy_camera/camera/camera/color/camera_info --box_threshold 0.1 --text_threshold 0.1
