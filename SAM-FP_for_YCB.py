import argparse
import cv2
import distinctipy
import os
import json
import torch
import imageio.v2 as imageio
import copy
import logging

import numpy as np

from GroundedSAM_demo.grounded_sam import GroundedSAM

from typing_extensions import Union

from estimater import FoundationPose

import trimesh

from Utils import *

BOP_DEPTH_SCALE = 1.0
ZFAR = np.inf


def load_models(checkpoints_dir: str, grounded_dino_config_dir: str, grounded_dino_use_vitb: bool,
               box_threshold: float, text_threshold: float, use_yolo_sam: bool, 
               sam_vit_model: str, mask_threshold: float, prompt_text: str, debug_dir: str, debug: int):
    """Initialize and load GroundedSAM and FoundationPose models.
    
    Returns:
        tuple: (grounded_sam, est) - The initialized models
    """
    print("Initializing models.")
    
    # Initialize GroundedSAM
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
    
    # Initialize FoundationPose with placeholder mesh
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
    
    return grounded_sam, est


def prepare_rgb_source(device: Union[int, str], image_exts: set, video_exts: set):
    """Prepare RGB source based on device parameter.
    
    Args:
        device: Device specification (webcam index, file path, or directory path)
        image_exts: Set of supported image extensions
        video_exts: Set of supported video extensions
        
    Returns:
        tuple: (rgb_source_type, rgb_single_frame, rgb_sequence_files, camera, rgb_video_cap)
    """
    rgb_source_type = None  # one of ['webcam','video','single_image','image_sequence']
    rgb_single_frame = None
    rgb_sequence_files = []
    camera = None
    rgb_video_cap = None

    # Interpret 'device' argument: if path exists decide type, else attempt webcam/video integer index or filename
    if isinstance(device, str) and os.path.exists(device):
        if os.path.isdir(device):
            rgb_sequence_files = sorted([
                os.path.join(device, f) for f in os.listdir(device)
                if os.path.splitext(f.lower())[1] in image_exts
            ])
            if not rgb_sequence_files:
                raise FileNotFoundError(f"No RGB image files found in directory: {device}")
            rgb_source_type = 'image_sequence'
            print(f"Loaded RGB image sequence with {len(rgb_sequence_files)} frames from {device}")
        else:
            ext_rgb = os.path.splitext(device.lower())[1]
            if ext_rgb in image_exts:
                rgb_single_frame = get_color(device)
                if rgb_single_frame is None:
                    raise IOError(f"Failed to read RGB image: {device}")
                rgb_source_type = 'single_image'
                print(f"Loaded single RGB image {device}")
            elif ext_rgb in video_exts:
                rgb_video_cap = cv2.VideoCapture(device)
                if not rgb_video_cap.isOpened():
                    raise IOError(f"Failed to open RGB video: {device}")
                rgb_source_type = 'video'
                print(f"Opened RGB video {device}")
            else:
                # fallback try webcam cast
                try:
                    cam_index = int(device)
                    camera = cv2.VideoCapture(cam_index)
                    if not camera.isOpened():
                        raise IOError(f"Failed to open webcam index {cam_index}")
                    rgb_source_type = 'webcam'
                    print(f"Opened webcam at index {cam_index}")
                except ValueError:
                    raise ValueError(f"Unrecognized RGB device/path: {device}")
    else:
        # treat as webcam index
        try:
            cam_index = int(device)
            camera = cv2.VideoCapture(cam_index)
            if not camera.isOpened():
                raise IOError(f"Failed to open webcam index {cam_index}")
            rgb_source_type = 'webcam'
            print(f"Opened webcam at index {cam_index}")
        except ValueError:
            # string path that doesn't exist
            raise FileNotFoundError(f"Provided device path does not exist: {device}")

    print("Initialized models and prepared RGB source.")
    
    return rgb_source_type, rgb_single_frame, rgb_sequence_files, camera, rgb_video_cap


def prepare_depth_source(depth_path: str, image_exts: set, video_exts: set, npy_ext: str):
    """Prepare depth source based on depth_path parameter.
    
    Args:
        depth_path: Path to depth data (file or directory)
        image_exts: Set of supported image extensions
        video_exts: Set of supported video extensions
        npy_ext: Numpy file extension
        
    Returns:
        tuple: (depth_source_type, depth_single_frame, depth_sequence_files, depth_npy_array, depth_video_cap)
    """
    depth_source_type = None  # one of ['single_image','image_sequence','npy_single','npy_sequence','video']
    depth_single_frame = None
    depth_sequence_files = []
    depth_npy_array = None
    depth_video_cap = None

    if os.path.isdir(depth_path):
        # directory of image sequence
        depth_sequence_files = sorted([
            os.path.join(depth_path, f) for f in os.listdir(depth_path)
            if os.path.splitext(f.lower())[1] in image_exts
        ])
        if not depth_sequence_files:
            raise FileNotFoundError(f"No depth image files found in directory: {depth_path}")
        depth_source_type = 'image_sequence'
        print(f"Loaded depth image sequence with {len(depth_sequence_files)} frames from {depth_path}")
    else:
        ext = os.path.splitext(depth_path.lower())[1]
        if ext in image_exts:
            depth_single_frame = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth_single_frame is None:
                raise IOError(f"Failed to read depth image: {depth_path}")
            depth_source_type = 'single_image'
            print(f"Loaded single depth image {depth_path}")
        elif ext == npy_ext:
            depth_npy_array = np.load(depth_path)
            if depth_npy_array.ndim == 2:
                depth_source_type = 'npy_single'
                print(f"Loaded single depth npy array with shape {depth_npy_array.shape} from {depth_path}")
            elif depth_npy_array.ndim == 3:
                depth_source_type = 'npy_sequence'
                print(f"Loaded depth npy sequence with {depth_npy_array.shape[0]} frames from {depth_path}")
            else:
                raise ValueError(f"Unsupported depth npy shape {depth_npy_array.shape}; expected (H,W) or (N,H,W)")
        elif ext in video_exts:
            depth_video_cap = cv2.VideoCapture(depth_path)
            if not depth_video_cap.isOpened():
                raise IOError(f"Failed to open depth video: {depth_path}")
            depth_source_type = 'video'
            print(f"Opened depth video {depth_path}")
        else:
            raise ValueError(f"Unrecognized depth_path extension: {ext}. Supported: image {image_exts}, video {video_exts}, npy {npy_ext}, or directory of images.")
    
    return depth_source_type, depth_single_frame, depth_sequence_files, depth_npy_array, depth_video_cap


def load_mesh(mesh_path: str, mesh_obj_id: int, est):
    """Load mesh and reset the estimator object if mesh is provided.
    
    Args:
        mesh_path: Path to mesh file or directory
        mesh_obj_id: Object ID for mesh resolution
        est: FoundationPose estimator instance
        
    Returns:
        loaded_mesh: The loaded mesh object or None
    """
    loaded_mesh = None
    if mesh_path is not None:
        candidate_path = mesh_path
        if os.path.isdir(mesh_path):
            # If directory and obj_id provided, try standard naming patterns
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
            # If still directory, attempt to find first mesh file
            if os.path.isdir(candidate_path):
                for fname in os.listdir(candidate_path):
                    if fname.lower().endswith(('.ply', '.obj')):
                        candidate_path = os.path.join(candidate_path, fname)
                        break
        if not os.path.exists(candidate_path):
            raise FileNotFoundError(f"Mesh file not found (after resolution): {candidate_path}")
        try:
            # loaded_mesh = trimesh.load(candidate_path, process=False)
            loaded_mesh = trimesh.load(candidate_path)

            loaded_mesh.vertices *= 1e-3  # convert mm to meters

            # # If units likely in mm (heuristic: bounding box large), optionally scale (TODO: expose flag if needed)
            # if loaded_mesh.scale > 1000:  # crude heuristic; leave unchanged otherwise
            #     pass

            symmetry_tfs = np.array([np.eye(4)])
            est.reset_object(model_pts=loaded_mesh.vertices.copy(),
                             model_normals=loaded_mesh.vertex_normals.copy(),
                             symmetry_tfs=symmetry_tfs, mesh=loaded_mesh)
            print(f"Loaded mesh from {candidate_path} (verts={len(loaded_mesh.vertices)})")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load mesh at {candidate_path}: {e}")
    else:
        print("No mesh_path provided; using placeholder box mesh.")
    
    return loaded_mesh


def load_intrinsics(intrinsics_path: str):
    """Load camera intrinsics data and depth scale.
    
    Args:
        intrinsics_path: Path to intrinsics file
        
    Returns:
        tuple: (intrinsics_data, BOP_DEPTH_SCALE) - The intrinsics data and depth scale
    """
    intrinsics_data = None
    bop_depth_scale = BOP_DEPTH_SCALE  # Use global default
    
    if intrinsics_path:
        try:
            intrinsics_data, bop_depth_scale = load_K(intrinsics_path)
            print(f"Loaded intrinsics data {intrinsics_path}")
            
            if isinstance(intrinsics_data, dict):
                print(f"Loaded intrinsics dict with {len(intrinsics_data)} entries from {intrinsics_path}")
            elif isinstance(intrinsics_data, np.ndarray):
                print(f"Loaded intrinsics matrix from {intrinsics_path}:\n{intrinsics_data}")
        except Exception as e:
            print(f"Warning: failed to load intrinsics from {intrinsics_path}: {e}. Will use fallback.")
            intrinsics_data = None
    
    return intrinsics_data, bop_depth_scale


def get_pose(device: Union[int, str],
                              checkpoints_dir: str, grounded_dino_config_dir: str, grounded_dino_use_vitb: bool,
                              box_threshold: float,  text_threshold: float,
                              use_yolo_sam: bool, sam_vit_model: str, mask_threshold: float,
                              prompt_text: str,
                              background: str,
                              depth_path: str,
                              mesh_path: str,
                              mesh_obj_id: int,
                              intrinsics_path: str,
                              scene_gt_path: str = None):
    print("Starting process to get the grounded segmentation masks of video frames.")

    # File extension sets (must be defined before any usage in list comprehensions)
    image_exts = {'.png', '.jpg', '.jpeg', '.tiff', '.tif', '.bmp', '.exr'}
    video_exts = {'.mp4', '.avi', '.mkv', '.mov'}
    npy_ext = '.npy'

    # Initialize models
    grounded_sam, est = load_models(
        checkpoints_dir=checkpoints_dir,
        grounded_dino_config_dir=grounded_dino_config_dir,
        grounded_dino_use_vitb=grounded_dino_use_vitb,
        box_threshold=box_threshold,
        text_threshold=text_threshold,
        use_yolo_sam=use_yolo_sam,
        sam_vit_model=sam_vit_model,
        mask_threshold=mask_threshold,
        prompt_text=prompt_text,
        debug_dir=args.debug_dir if 'args' in globals() else 'debug',
        debug=args.debug if 'args' in globals() else 0
    )

    # -------------------- Prepare RGB source (mirrors depth handling) --------------------
    rgb_source_type, rgb_single_frame, rgb_sequence_files, camera, rgb_video_cap = prepare_rgb_source(
        device=device, image_exts=image_exts, video_exts=video_exts
    )


    # -------------------- Prepare depth source --------------------
    depth_source_type, depth_single_frame, depth_sequence_files, depth_npy_array, depth_video_cap = prepare_depth_source(
        depth_path=depth_path, image_exts=image_exts, video_exts=video_exts, npy_ext=npy_ext
    )

    # -------------------- Load mesh (if provided) --------------------
    loaded_mesh = load_mesh(mesh_path=mesh_path, mesh_obj_id=mesh_obj_id, est=est)


    # -------------------- Synchronization setup --------------------
    def _depth_len():
        if depth_source_type == 'image_sequence':
            return len(depth_sequence_files)
        if depth_source_type == 'npy_sequence':
            return depth_npy_array.shape[0]
        return None
    def _rgb_len():
        if rgb_source_type == 'image_sequence':
            return len(rgb_sequence_files)
        return None

    d_len = _depth_len()
    r_len = _rgb_len()
    if d_len is not None and r_len is not None:
        sync_len = min(d_len, r_len)
    else:
        sync_len = d_len if r_len is None else r_len  # if one is streaming, limit by finite one

    # Load intrinsics (can return a 3x3 np.ndarray or a dict per frame)
    intrinsics_data, bop_depth_scale = load_intrinsics(intrinsics_path=intrinsics_path)
    # Ensure the global depth scale used by get_depth is updated
    global BOP_DEPTH_SCALE
    BOP_DEPTH_SCALE = bop_depth_scale
    print(f"Using BOP depth scale: {BOP_DEPTH_SCALE}")

    # expected maximal number of different objects/needed color in an image
    num_max_objs = 50
    colors = distinctipy.get_colors(num_max_objs)

    print("Start getting the grounded segmentation masks for all frames.")

    cv2.namedWindow("GroundedSam", cv2.WINDOW_NORMAL)

    frame_idx = 0
    while True:
        if sync_len is not None and frame_idx >= sync_len:
            print("Reached end of synchronized sequences.")
            break

        # -------------------- Fetch RGB frame --------------------
        if rgb_source_type == 'webcam':
            ret, original_frame = camera.read()
            if not ret:
                print("Failed to grab frame from webcam.")
                continue
            # Convert BGR to RGB since cv2.VideoCapture returns BGR
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        elif rgb_source_type == 'video':
            ret, original_frame = rgb_video_cap.read()
            if not ret:
                rgb_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, original_frame = rgb_video_cap.read()
                if not ret:
                    print("Failed to read frame from RGB video; ending loop.")
                    break
            # Convert BGR to RGB since cv2.VideoCapture returns BGR
            original_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        elif rgb_source_type == 'single_image':
            original_frame = rgb_single_frame.copy()
        elif rgb_source_type == 'image_sequence':
            seq_path = rgb_sequence_files[frame_idx % len(rgb_sequence_files)] if sync_len is None else rgb_sequence_files[frame_idx]
            original_frame = get_color(seq_path)
            if original_frame is None:
                print(f"Failed to read RGB frame: {seq_path}")
                break
        else:
            print("Unknown RGB source type; aborting.")
            break

        # Ensure the frame is in the correct format for OpenCV operations
        original_frame = np.asarray(original_frame, dtype=np.uint8)
        original_frame = np.ascontiguousarray(original_frame)
        cv2.normalize(original_frame, original_frame, 0, 255, cv2.NORM_MINMAX)

        # -------------------- Fetch depth frame --------------------
        if depth_source_type == 'single_image':
            depth_frame = depth_single_frame
        elif depth_source_type == 'image_sequence':
            d_seq_path = depth_sequence_files[frame_idx % len(depth_sequence_files)] if sync_len is None else depth_sequence_files[frame_idx]
            # depth_frame = cv2.imread(d_seq_path, cv2.IMREAD_UNCHANGED)
            depth_frame = get_depth(depth_file=d_seq_path)
        elif depth_source_type == 'npy_single':
            depth_frame = depth_npy_array
        elif depth_source_type == 'npy_sequence':
            if sync_len is None:
                depth_frame = depth_npy_array[frame_idx % depth_npy_array.shape[0]]
            else:
                depth_frame = depth_npy_array[frame_idx]
        elif depth_source_type == 'video':
            ret_d, depth_frame = depth_video_cap.read()
            if not ret_d:
                depth_video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_d, depth_frame = depth_video_cap.read()
                if not ret_d:
                    print("Failed to read depth frame from video; skipping pose estimation for this frame.")
                    depth_frame = None
        else:
            depth_frame = None

        if depth_frame is not None:
            if depth_frame.ndim == 3 and depth_frame.shape[2] > 1:
                depth_frame = depth_frame[:, :, 0]
            depth = depth_frame.astype(np.float32)
        else:
            depth = None
        # depth_v = depth>=0.001
        # print('depth_v.sum() 2', depth_v.sum())
        # depth *= 0  # no depth for the moment TODO

        # -------------------- Detection --------------------
        rgb_frame = original_frame  # Already in RGB format
        detections = grounded_sam.generate_masks(rgb_frame)

        if detections:
            masks = detections["masks"].squeeze(1).cpu().numpy()
            masks_scores = detections["masks_scores"].cpu().numpy()
            sort_indicis = np.argsort(-masks_scores)
            masks = masks[sort_indicis]
            masks_scores = masks_scores[sort_indicis]

            # Light morphological cleanup to shrink over-segmented masks
            cleaned_masks = []
            kernel = np.ones((3,3), np.uint8)
            for m in masks:
                m_u8 = (m.astype(np.uint8) * 255)
                m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_OPEN, kernel)
                m_u8 = cv2.morphologyEx(m_u8, cv2.MORPH_CLOSE, kernel)
                cleaned_masks.append(m_u8.astype(bool))
            masks = np.array(cleaned_masks)

            # Save the most confident mask to debug folder
            if args.debug>=2 and masks.shape[0] > 0 and 'args' in globals() and hasattr(args, 'debug_dir'):
                masks_dir = f'{args.debug_dir}/masks'
                os.makedirs(masks_dir, exist_ok=True)
                most_confident_mask = masks[0]  # First mask after sorting by confidence
                # Convert boolean mask to 255/0 for saving as image
                mask_img = (most_confident_mask * 255).astype(np.uint8)
                cv2.imwrite(f'{masks_dir}/frame_{frame_idx:06d}_mask.png', mask_img)
                print(f"Saved most confident mask for frame {frame_idx} (confidence: {masks_scores[0]:.3f})")

            if background == "rgb":
                frame = rgb_frame.copy()
            elif background == "gray":
                frame = cv2.cvtColor(cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            elif background == "black":
                frame = np.zeros_like(rgb_frame, dtype=np.uint8)
            elif background == "white":
                frame = np.ones_like(rgb_frame, dtype=np.uint8)
            else:
                raise NotImplementedError("Value for 'background' must be one of ['rgb', 'gray', 'black', 'white']")

            alpha = 0.3
            for mask_idx, mask in enumerate(masks):
                r = int(255 * colors[mask_idx][0])
                g = int(255 * colors[mask_idx][1])
                b = int(255 * colors[mask_idx][2])
                frame[mask, 0] = alpha * r + (1 - alpha) * frame[mask, 0]
                frame[mask, 1] = alpha * g + (1 - alpha) * frame[mask, 1]
                frame[mask, 2] = alpha * b + (1 - alpha) * frame[mask, 2]
        else:
            masks = None


        # -------------------- Pose estimation --------------------
        if masks is not None and depth is not None and loaded_mesh is not None:

            most_confident_mask = masks[0]
            
            # Save the depth image exactly as passed to the estimator
            if args.debug >= 2 and 'args' in globals() and hasattr(args, 'debug_dir'):
                depth_dir = f'{args.debug_dir}/depth_images'
                os.makedirs(depth_dir, exist_ok=True)
                # Save depth as numpy array (preserves exact values)
                # np.save(f'{depth_dir}/frame_{frame_idx:06d}_depth.npy', depth)
                # Also save as visualization (16-bit PNG for better precision)
                depth_vis = (depth * 1000).astype(np.uint16)  # Convert to mm and save as 16-bit
                cv2.imwrite(f'{depth_dir}/frame_{frame_idx:06d}_depth.png', depth_vis)
                print(f"Saved depth image for frame {frame_idx} (shape: {depth.shape}, min: {depth.min():.3f}, max: {depth.max():.3f})")
            
            out = run_pose_estimation_worker(most_confident_mask, est, intrinsics_data, ob_id=mesh_obj_id, device=0, depth_image=depth, rgb_image=rgb_frame)
            
            # -------------------- Ground Truth Comparison --------------------
            if scene_gt_path is not None and out is not None:
                # Extract actual frame ID from filename
                actual_frame_id = frame_idx  # Default fallback
                if rgb_source_type == 'image_sequence':
                    # Extract frame ID from RGB filename
                    rgb_filename = os.path.basename(seq_path)
                    actual_frame_id = int(os.path.splitext(rgb_filename)[0])
                elif rgb_source_type == 'single_image':
                    # For single image, try to extract from filename
                    rgb_filename = os.path.basename(device) if isinstance(device, str) else f"{frame_idx:06d}"
                    try:
                        actual_frame_id = int(os.path.splitext(rgb_filename)[0])
                    except ValueError:
                        actual_frame_id = 0  # Use 0 for single images with non-numeric names
                
                # Get ground truth pose for current frame
                gt_pose = get_gt_pose(scene_gt_path, actual_frame_id, mesh_obj_id)
                # out = gt_pose
                
                if gt_pose is not None:
                    # Convert estimated pose to numpy if needed
                    try:
                        est_pose = out.detach().cpu().numpy() if hasattr(out, 'detach') else (out.cpu().numpy() if hasattr(out, 'cpu') else np.asarray(out))
                    except Exception:
                        est_pose = np.asarray(out)
                    
                    # Compare poses
                    comparison = compare_poses(gt_pose, est_pose)
                    
                    print(f"Frame {actual_frame_id:06d} (loop {frame_idx}) - Object {mesh_obj_id}:")
                    print(f"  Translation error: {comparison['translation_error']:.4f}")
                    print(f"  Rotation error: {comparison['rotation_error_degrees']:.2f}°")
                    print(f"  Poses similar: {comparison['poses_similar']}")
                    if comparison['poses_similar']==False:
                        print(f'the ground truth pose is \n {gt_pose}')

                    
                    
                    # Optionally save comparison results
                    if args.debug and hasattr(args, 'debug_dir'):
                        os.makedirs(f'{args.debug_dir}/pose_comparison', exist_ok=True)
                        np.save(f'{args.debug_dir}/pose_comparison/frame_{actual_frame_id:06d}_gt.npy', gt_pose)
                        np.save(f'{args.debug_dir}/pose_comparison/frame_{actual_frame_id:06d}_est.npy', est_pose)
                        with open(f'{args.debug_dir}/pose_comparison/frame_{actual_frame_id:06d}_comparison.json', 'w') as f:
                            # Convert numpy types to regular python types for JSON serialization
                            json_comparison = {k: float(v) if isinstance(v, np.floating) else bool(v) if isinstance(v, np.bool_) else v 
                                            for k, v in comparison.items() if k not in ['gt_pose', 'est_pose']}
                            json_comparison['gt_pose'] = str(comparison['gt_pose'].tolist())
                            json_comparison['est_pose'] = str(comparison['est_pose'].tolist())
                            json.dump(json_comparison, f, indent=2)
                else:
                    print(f"Frame {actual_frame_id:06d} (loop {frame_idx}) - No ground truth pose found for object {mesh_obj_id}")
        else:
            out = None

        # -------------------- Display --------------------
        if masks is not None:
            disp_frame = frame
        else:
            disp_frame = rgb_frame
        # -------------------- Visualization overlays (using embedded function) --------------------
        if out is not None and loaded_mesh is not None:
            try:
                pose_np = out.detach().cpu().numpy() if hasattr(out, 'detach') else (out.cpu().numpy() if hasattr(out, 'cpu') else np.asarray(out))
            except Exception:
                pose_np = np.asarray(out)
            try:
                # Use the new visualization util; keep window management here, so disable internal window
                vis_img = visualize_pose_result(
                    color=disp_frame,
                    pose=pose_np,
                    K=intrinsics_data,
                    mesh=loaded_mesh,
                    debug_dir=args.debug_dir if 'args' in globals() else None,
                    frame_id=f"{frame_idx:06d}",
                    show_window=False,
                )
                disp_frame = vis_img
            except Exception as e:
                print(f"Visualization error: {e}")

        cv2.imshow("GroundedSam", cv2.cvtColor(disp_frame, cv2.COLOR_RGB2BGR))

        frame_idx += 1
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            print("Escape pressed, closing.")
            break

    # clean up
    if camera is not None:
        camera.release()
    if rgb_video_cap is not None:
        rgb_video_cap.release()
    if depth_video_cap is not None:
        depth_video_cap.release()
        cv2.destroyAllWindows()

        print("Finished processing the frames.")
        print("End")

def visualize_pose_result(color, pose, K, mesh, debug_dir=None, frame_id=None, show_window=True):
    try:
        # Calculate bounding box and transformation like in run_demo.py
        to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
        bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2,3)
        center_pose = pose@np.linalg.inv(to_origin)
        
        # Axis length based on mesh size to avoid oversized axes
        axis_scale = float(np.max(extents)) if np.max(extents) > 0 else 0.1
        vis = draw_xyz_axis(color, ob_in_cam=center_pose, scale=axis_scale, K=K, thickness=3, transparency=0, is_input_rgb=True)
        # Also draw the 3D bounding box
        vis = draw_posed_3d_box(K, img=vis, ob_in_cam=center_pose, bbox=bbox, line_color=(0,255,0), linewidth=2)

        if debug_dir and frame_id:
            os.makedirs(f'{debug_dir}/pose_vis', exist_ok=True)
            cv2.imwrite(f'{debug_dir}/pose_vis/{frame_id}.png', vis[...,::-1])
        
        return vis
        
    except Exception as e:
        print(f"Error in visualize_pose_result: {e}")
        import traceback
        traceback.print_exc()
        return color

def run_pose_estimation_worker(mask, est:FoundationPose, internal_cam_parameters, ob_id=None, device:int=0, depth_image=None, rgb_image=None):
    result = NestDict()
    torch.cuda.set_device(device)
    est.to_device(f'cuda:{device}')
    est.glctx = dr.RasterizeCudaContext(device)

    # est.gt_pose = reader.get_gt_pose(i_frame, ob_id)
    # print("mask sum:", np.sum(mask))
    # print("depth min/max:", np.min(depth_image), np.max(depth_image))
    # print("depth sum:", np.sum(depth_image))
    # print("Camera intrinsics:", internal_cam_parameters)
    # mymask = (mask * 255).astype(np.uint8)
    # depth_vis = np.multiply(depth_image, 1000)  # Convert to mm and save as 16-bit

    print(f"types \n depth image: {depth_image.dtype}, rgb image: {rgb_image.dtype}, mask: {mask.dtype}'")
    print(f'dims: \n depth: {depth_image.shape}\n rgb: {rgb_image.shape}\n mask: {mask.shape}')
    print(f"K is : {internal_cam_parameters}")
    pose = est.register(K=internal_cam_parameters, rgb=rgb_image, depth=depth_image, ob_mask=mask, ob_id=ob_id, iteration=5)
    logging.info(f"pose:\n{pose}")

    result = pose

    return result


def load_K(path="/home/student/Desktop/perception/FoundationPose/demo_data/YCB_Video/test/000048/scene_camera.json"):


    K_table = {}
    with open(path,'r') as ff:
        info = json.load(ff)
    for k in info:
        K_table[f'{int(k):06d}'] = np.array(info[k]['cam_K']).reshape(3,3)
        bop_depth_scale = info[k]['depth_scale']

    K = list(K_table.values())[0]
    return K, bop_depth_scale

def get_depth(filled=False, depth_file=None):
    if filled:
        depth = cv2.imread(depth_file,-1)/1e3
    else:
        depth = cv2.imread(depth_file,-1)*1e-3*BOP_DEPTH_SCALE
    # if self.resize!=1:
    #     depth = cv2.resize(depth, fx=self.resize, fy=self.resize, dsize=None, interpolation=cv2.INTER_NEAREST)
    depth[depth<0.001] = 0
    depth[depth>ZFAR] = 0
    return depth

def get_color(path, resize=1):
    color = imageio.imread(path)
    if len(color.shape)==2:
        color = np.tile(color[...,None], (1,1,3))  # Gray to RGB
    elif len(color.shape)==3 and color.shape[2]==4:
        color = color[...,:3]  # RGBA to RGB (remove alpha channel)
    if resize!=1:
        color = cv2.resize(color, fx=resize, fy=resize, dsize=None, interpolation=cv2.INTER_NEAREST)
    # Ensure compatibility with OpenCV operations
    color = np.asarray(color, dtype=np.uint8)
    color = np.ascontiguousarray(color)
    return color

def get_gt_pose(scene_gt_path, frame_id, ob_id, mask=None, use_my_correction=False):
    """Load ground truth pose for a specific frame and object ID.
    
    Args:
        scene_gt_path: Path to the scene_gt.json file
        frame_id: Frame identifier (integer or string)
        ob_id: Object ID to get pose for
        mask: Optional mask for multi-instance disambiguation
        use_my_correction: Whether to apply dataset-specific corrections
        
    Returns:
        4x4 pose matrix or None if not found
    """
    if not os.path.exists(scene_gt_path):
        print(f"Ground truth file not found: {scene_gt_path}")
        return None
    
    try:
        with open(scene_gt_path, 'r') as ff:
            scene_gt = json.load(ff)
    except Exception as e:
        print(f"Error loading scene_gt.json: {e}")
        return None
    
    ob_in_cam = np.eye(4)
    best_iou = -np.inf
    
    # Convert frame_id to string for lookup
    frame_key = str(int(frame_id))
    
    if frame_key not in scene_gt:
        print(f"Frame {frame_key} not found in scene_gt")
        return None
    
    for i_k, k in enumerate(scene_gt[frame_key]):
        if k['obj_id'] == ob_id:
            cur = np.eye(4)
            cur[:3,:3] = np.array(k['cam_R_m2c']).reshape(3,3)
            cur[:3,3] = np.array(k['cam_t_m2c']) / 1e3  # Convert mm to m
            
            if mask is not None:
                # For multi-instance disambiguation (requires mask_visib directory)
                scene_dir = os.path.dirname(scene_gt_path)
                mask_file = f'{scene_dir}/mask_visib/{frame_id:06d}_{i_k:06d}.png'
                if os.path.exists(mask_file):
                    gt_mask = cv2.imread(mask_file, -1).astype(bool)
                    intersect = (gt_mask * mask).astype(bool)
                    union = (gt_mask + mask).astype(bool)
                    if union.sum() > 0:
                        iou = float(intersect.sum()) / union.sum()
                        if iou > best_iou:
                            best_iou = iou
                            ob_in_cam = cur
                else:
                    print(f"Mask file not found: {mask_file}")
                    ob_in_cam = cur
                    break
            else:
                ob_in_cam = cur
                break
    
    return ob_in_cam


def compare_poses(gt_pose, est_pose, translation_threshold=0.05, rotation_threshold_degrees=5.0):
    """Compare two 4x4 pose matrices and return similarity metrics.
    
    Args:
        pose1: First 4x4 pose matrix
        pose2: Second 4x4 pose matrix  
        translation_threshold: Threshold for translation difference in meters
        rotation_threshold_degrees: Threshold for rotation difference in degrees
        
    Returns:
        dict with comparison results
    """
    if gt_pose is None or est_pose is None:
        return {
            'translation_error': float('inf'),
            'rotation_error_degrees': float('inf'),
            'translation_match': False,
            'rotation_match': False,
            'poses_similar': False
        }
    
    # Translation error (Euclidean distance)
    translation_error = np.linalg.norm(gt_pose[:3, 3] - est_pose[:3, 3])
    
    # Rotation error using rotation matrix difference
    R1 = gt_pose[:3, :3]
    R2 = est_pose[:3, :3]
    
    # Compute rotation error using trace of R1^T * R2
    R_diff = R1.T @ R2
    trace_val = np.trace(R_diff)
    # Clamp trace to valid range for arccos
    trace_val = np.clip(trace_val, -1.0, 3.0)
    rotation_error_rad = np.arccos((trace_val - 1) / 2)
    rotation_error_degrees = np.degrees(rotation_error_rad)
    
    # Check if poses are similar within thresholds
    translation_match = translation_error < translation_threshold
    rotation_match = rotation_error_degrees < rotation_threshold_degrees
    poses_similar = translation_match and rotation_match
    
    return {
        'translation_error': translation_error,
        'rotation_error_degrees': rotation_error_degrees,
        'translation_match': translation_match,
        'rotation_match': rotation_match,
        'poses_similar': poses_similar,
        'gt_pose': gt_pose,
        'est_pose': est_pose
    }


    # (Removed stray call to self.make_id_strs which was invalid in this context)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Applies 'GroundedSam' to all video frames.")

    # G_SAM args
    parser.add_argument("-i", "--device", dest="device", type=str, default=0,
                        help="Device/Video file for the video camera.")
    
    parser.add_argument("--depth_path", type=str, help="Path to the depth data (single image, directory of images, .npy, or video).")
    parser.add_argument("--intrinsics_path", type=str, default="/home/student/Desktop/perception/FoundationPose/demo_data/YCB_Video/test/000048/scene_camera.json", help="Path to camera intrinsics file (scene_camera.json, JSON, YAML, or txt).")
    parser.add_argument("--mesh_path", type=str, default=None, help="Path to mesh file (.ply/.obj) or directory containing mesh files.")
    parser.add_argument("--mesh_obj_id", type=int, default=None, help="Object id used to resolve mesh filename inside mesh directory (e.g., obj_000001.ply).")
    parser.add_argument("--scene_gt_path", type=str, default='demo_data/YCB_Video/test/000048/scene_gt.json', help="Path to the scene_gt.json file for ground truth pose comparison.")

    parser.add_argument("-e", "--checkpoints_dir", dest="checkpoints_dir", type=str, default="GroundedSAM_demo/checkpoints",
                    help="Path of the root dir containing the checkpoints.")
    parser.add_argument("-d", "--grounded_dino_config_dir", dest="grounded_dino_config_dir", type=str, default="GroundedSAM_demo/cfg/gdino",
                        help="Path of the dir containing the configuration files for 'GroundedDino'.")
    parser.add_argument("-u", "--grounded_dino_use_vitb", dest="grounded_dino_use_vitb", action="store_true", default=False,
                        help="Use the vit_b backbone for GroundingDino.")

    parser.add_argument("-m", "--box_threshold", dest="box_threshold", type=float, default=0.1,
                        help="The minimum confidence score of a bounding box to be used as prompt.")
    parser.add_argument("-n", "--text_threshold", dest="text_threshold", type=float, default=0.1,
                        help="The minimum confidence score that the bounding box class matches the phrase.")

    parser.add_argument("-y", "--use_yolo_sam", dest="use_yolo_sam", action="store_true", default=False,
                        help="Use Yolo implementation of SAM.")
    parser.add_argument("-s", "--sam_vit_model", dest="sam_vit_model", type=str, default="vit_b",
                        help="Which SAM model/backbone size to use.")
    parser.add_argument("-q", "--mask_threshold", dest="mask_threshold", type=float, default=0.01,
                        help="The minimum confidence score of a segmentation masks.")

    parser.add_argument("-p", "--prompt_text", dest="prompt_text", type=str, default="objects",
                        help="Prompt for the bounding box search.")

    parser.add_argument("-b", "--background", dest="background", choices=["rgb", "gray", "black", "white"],
                        default="gray",
                        help="Choose which background image for the renderings to use: 'rgb': scene image, 'gray': grayscale scene image, 'black'/'white': black/white background.")
    
    # F_Pose args
    code_dir = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument('--debug', type=int, default=2)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')

    #TODO: add foundation pose args
    #args = parser.parse_args()


    argsv_list = ["-i", "demo_data/YCB_Video/test/000048/rgb", "--depth_path", "demo_data/YCB_Video/test/000048/depth",
                  "-m", "0.3", "-n", "0.3", "-y", "-s", "sam2.1_l.pt", "-q", "0.01", "-p", "Objects", "-b", "gray", "--mesh_path",
                  "demo_data/YCB_Video/models", "--mesh_obj_id", "14", "--scene_gt_path", "demo_data/YCB_Video/test/000048/scene_gt.json"]
    args = parser.parse_args(argsv_list)
    

    get_pose(device=args.device,
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
                              depth_path=args.depth_path,
                              mesh_path=args.mesh_path,
                              mesh_obj_id=args.mesh_obj_id,
                              intrinsics_path=args.intrinsics_path,
                              scene_gt_path=args.scene_gt_path)
    


    # Example usage commands:
    # python3 -m integration -i "GroundedSAM_demo/demo.mp4" -m 0.3 -n 0.3 -y -s "mobile_sam.pt" -q 0.01 -p "red cup" -b "gray"
    # python3 -m integration -i "demo_data/YCB_Video/test/000048/rgb" --depth_path "demo_data/YCB_Video/test/000048/depth" -m 0.3 -n 0.3 -y -s "mobile_sam.pt" -q 0.01 -p "red cup" -b "gray" --mesh_path "demo_data/YCB_Video/models" --mesh_obj_id 14
    # With ground truth comparison:
    # python3 -m integration -i "demo_data/YCB_Video/test/000048/rgb" --depth_path "demo_data/YCB_Video/test/000048/depth" -m 0.3 -n 0.3 -y -s "mobile_sam.pt" -q 0.01 -p "red cup" -b "gray" --mesh_path "demo_data/YCB_Video/models" --mesh_obj_id 14 --scene_gt_path "demo_data/YCB_Video/test/000048/scene_gt.json"