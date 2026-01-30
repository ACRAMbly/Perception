# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


from Utils import *
from datareader import *
import itertools
from learning.training.predict_score import *
from learning.training.predict_pose_refine import *
import yaml
import mycpp
import tempfile


class FoundationPose:
  """
  Main FoundationPose class for 6D object pose estimation and tracking.
  
  This class implements the FoundationPose algorithm which performs robust 6D object pose
  estimation from RGB-D images. It combines deep learning-based scoring and refinement
  with geometric pose hypotheses to achieve accurate pose estimation.
  """
  
  def __init__(self, model_pts, model_normals, symmetry_tfs=None, mesh=None, scorer:ScorePredictor=None, refiner:PoseRefinePredictor=None, glctx=None, debug=0, debug_dir='/home/bowen/debug/novel_pose_debug/'):
    """
    Initialize the FoundationPose estimator.
    
    Args:
        model_pts: 3D model points (vertices) of the target object
        model_normals: Normal vectors corresponding to model points
        symmetry_tfs: Transformation matrices representing object symmetries (optional)
        mesh: 3D mesh object (trimesh format) of the target object
        scorer: Pre-trained neural network for pose scoring (optional, creates default if None)
        refiner: Pre-trained neural network for pose refinement (optional, creates default if None)
        glctx: OpenGL/CUDA rendering context for visualization (optional)
        debug: Debug level (0=off, 1=basic, 2=detailed with visualizations)
        debug_dir: Directory path for saving debug outputs and visualizations
    """
    self.gt_pose = None
    self.ignore_normal_flip = True
    self.debug = debug
    self.debug_dir = debug_dir
    os.makedirs(debug_dir, exist_ok=True)

    self.reset_object(model_pts, model_normals, symmetry_tfs=symmetry_tfs, mesh=mesh)
    self.make_rotation_grid(min_n_views=40, inplane_step=60)

    self.glctx = glctx

    if scorer is not None:
      self.scorer = scorer
    else:
      self.scorer = ScorePredictor()

    if refiner is not None:
      self.refiner = refiner
    else:
      self.refiner = PoseRefinePredictor()

    self.pose_last = None   # Used for tracking; per the centered mesh


  def reset_object(self, model_pts, model_normals, symmetry_tfs=None, mesh=None):
    """
    Reset and initialize the target object model for pose estimation.
    
    This function processes the input 3D model to prepare it for pose estimation:
    - Centers the mesh at origin for numerical stability
    - Computes object diameter and voxel size for sampling
    - Creates point cloud representation with normals
    - Sets up mesh tensors for GPU rendering
    - Initializes symmetry transformations
    
    Args:
        model_pts: 3D model points (vertices) of the target object
        model_normals: Normal vectors corresponding to model points  
        symmetry_tfs: Transformation matrices representing object symmetries (optional)
        mesh: 3D mesh object (trimesh format) of the target object
    """
    max_xyz = mesh.vertices.max(axis=0)
    min_xyz = mesh.vertices.min(axis=0)
    self.model_center = (min_xyz+max_xyz)/2
    if mesh is not None:
      self.mesh_ori = mesh.copy()
      mesh = mesh.copy()
      mesh.vertices = mesh.vertices - self.model_center.reshape(1,3)

    model_pts = mesh.vertices
    self.diameter = compute_mesh_diameter(model_pts=mesh.vertices, n_sample=10000)
    self.vox_size = max(self.diameter/20.0, 0.003)
    # logging.info(f'self.diameter:{self.diameter}, vox_size:{self.vox_size}')
    self.dist_bin = self.vox_size/2
    self.angle_bin = 20  # Deg
    pcd = toOpen3dCloud(model_pts, normals=model_normals)
    pcd = pcd.voxel_down_sample(self.vox_size)
    self.max_xyz = np.asarray(pcd.points).max(axis=0)
    self.min_xyz = np.asarray(pcd.points).min(axis=0)
    self.pts = torch.tensor(np.asarray(pcd.points), dtype=torch.float32, device='cuda')
    self.normals = F.normalize(torch.tensor(np.asarray(pcd.normals), dtype=torch.float32, device='cuda'), dim=-1)
    # logging.info(f'self.pts:{self.pts.shape}')
    self.mesh_path = None
    self.mesh = mesh
    if self.mesh is not None:
      # Export mesh to a writable location. Some containerized environments mount /tmp read-only
      # causing a PermissionError when trimesh tries to write auxiliary material files.
      primary_dir = tempfile.gettempdir()
      try:
        self.mesh_path = os.path.join(primary_dir, f'{uuid.uuid4()}.obj')
        self.mesh.export(self.mesh_path)
      except PermissionError:
        # Fallback to a directory under debug_dir which we already created with write perms.
        fallback_dir = os.path.join(self.debug_dir, 'mesh_cache')
        os.makedirs(fallback_dir, exist_ok=True)
        self.mesh_path = os.path.join(fallback_dir, f'{uuid.uuid4()}.obj')
        self.mesh.export(self.mesh_path)
    self.mesh_tensors = make_mesh_tensors(self.mesh)

    if symmetry_tfs is None:
      self.symmetry_tfs = torch.eye(4).float().cuda()[None]
    else:
      self.symmetry_tfs = torch.as_tensor(symmetry_tfs, device='cuda', dtype=torch.float)

    logging.info("reset done")



  def get_tf_to_centered_mesh(self):
    """
    Get transformation matrix to convert from centered mesh coordinates to original coordinates.
    
    The algorithm works with mesh centered at origin for numerical stability. This function
    returns the transformation matrix that converts poses from the centered coordinate system
    back to the original mesh coordinate system.
    
    Returns:
        torch.Tensor: 4x4 transformation matrix that translates by negative model center
    """
    tf_to_center = torch.eye(4, dtype=torch.float, device='cuda')
    tf_to_center[:3,3] = -torch.as_tensor(self.model_center, device='cuda', dtype=torch.float)
    return tf_to_center


  def to_device(self, s='cuda:0'):
    """
    Move all tensor attributes and models to specified device (GPU/CPU).
    
    This function handles device migration for all class attributes including:
    - Tensor attributes (points, normals, rotation grids, etc.)
    - Neural network models (scorer and refiner)
    - Mesh tensors for rendering
    - OpenGL/CUDA context
    
    Args:
        s: Device string (e.g., 'cuda:0', 'cpu')
    """
    for k in self.__dict__:
      self.__dict__[k] = self.__dict__[k]
      if torch.is_tensor(self.__dict__[k]) or isinstance(self.__dict__[k], nn.Module):
        # logging.info(f"Moving {k} to device {s}")
        self.__dict__[k] = self.__dict__[k].to(s)
    for k in self.mesh_tensors:
      # logging.info(f"Moving {k} to device {s}")
      self.mesh_tensors[k] = self.mesh_tensors[k].to(s)
    if self.refiner is not None:
      self.refiner.model.to(s)
    if self.scorer is not None:
      self.scorer.model.to(s)
    if self.glctx is not None:
      self.glctx = dr.RasterizeCudaContext(s)



  def make_rotation_grid(self, min_n_views=40, inplane_step=60):
    """
    Generate a grid of rotation hypotheses for pose estimation.
    
    This function creates a comprehensive set of rotation hypotheses by:
    1. Sampling viewpoints uniformly on a sphere using icosphere sampling
    2. For each viewpoint, generating multiple in-plane rotations
    3. Clustering similar poses to reduce redundancy while respecting object symmetries
    
    The rotation grid is used during pose registration to generate initial pose hypotheses
    that are then refined using neural networks.
    
    Args:
        min_n_views: Minimum number of viewpoints to sample on sphere
        inplane_step: Step size in degrees for in-plane rotations (360/inplane_step rotations per view)
    """
    cam_in_obs = sample_views_icosphere(n_views=min_n_views)
    # logging.info(f'cam_in_obs:{cam_in_obs.shape}')
    rot_grid = []
    for i in range(len(cam_in_obs)):
      for inplane_rot in np.deg2rad(np.arange(0, 360, inplane_step)):
        cam_in_ob = cam_in_obs[i]
        R_inplane = euler_matrix(0,0,inplane_rot)
        cam_in_ob = cam_in_ob@R_inplane
        ob_in_cam = np.linalg.inv(cam_in_ob)
        rot_grid.append(ob_in_cam)

    rot_grid = np.asarray(rot_grid)
    # logging.info(f"rot_grid:{rot_grid.shape}")
    rot_grid = mycpp.cluster_poses(30, 99999, rot_grid, self.symmetry_tfs.data.cpu().numpy())
    rot_grid = np.asarray(rot_grid)
    # logging.info(f"after cluster, rot_grid:{rot_grid.shape}")
    self.rot_grid = torch.as_tensor(rot_grid, device='cuda', dtype=torch.float)
    logging.info(f"self.rot_grid: {self.rot_grid.shape}")


  def generate_random_pose_hypo(self, K, rgb, depth, mask, scene_pts=None):

    """
    @scene_pts: torch tensor (N,3)
    
    Generate pose hypotheses by combining rotation grid with estimated translation.
    
    This function creates initial pose hypotheses for pose estimation by:
    1. Taking all rotations from the pre-computed rotation grid
    2. Estimating a single translation using depth and mask information
    3. Combining each rotation with the estimated translation
    
    The resulting pose hypotheses serve as starting points for neural network refinement.
    
    Args:
        K: 3x3 camera intrinsic matrix
        rgb: RGB image (H, W, 3)
        depth: Depth image (H, W) in meters
        mask: Binary object mask (H, W) 
        scene_pts: Optional scene point cloud (not currently used)
        
    Returns:
        torch.Tensor: Array of 4x4 pose matrices (N, 4, 4) where N is number of rotations
    """
    ob_in_cams = self.rot_grid.clone()
    center = self.guess_translation(depth=depth, mask=mask, K=K)
    ob_in_cams[:,:3,3] = torch.tensor(center, device='cuda', dtype=torch.float).reshape(1,3)
    return ob_in_cams


  def guess_translation(self, depth, mask, K):
    """
    Estimate object translation (3D position) from depth image and object mask.
    
    This function estimates the 3D position of the object center by:
    1. Finding the 2D centroid of the object mask
    2. Computing the median depth value within the valid masked region
    3. Back-projecting the 2D centroid to 3D using camera intrinsics and median depth
    
    This provides a reasonable initial guess for object translation that works well
    even with partial occlusions and noisy depth data.
    
    Args:
        depth: Depth image (H, W) in meters
        mask: Binary object mask (H, W)
        K: 3x3 camera intrinsic matrix
        
    Returns:
        numpy.ndarray: 3D translation vector (3,) representing object center in camera coordinates
    """
    vs,us = np.where(mask>0)
    if len(us)==0:
      # logging.info(f'mask is all zero')
      return np.zeros((3))
    uc = (us.min()+us.max())/2.0
    vc = (vs.min()+vs.max())/2.0
    valid = mask.astype(bool) & (depth>=0.001)
    if not valid.any():
      # logging.info(f"valid is empty")
      return np.zeros((3))

    zc = np.median(depth[valid])
    center = (np.linalg.inv(K)@np.asarray([uc,vc,1]).reshape(3,1))*zc

    if self.debug>=2:
      pcd = toOpen3dCloud(center.reshape(1,3))
      o3d.io.write_point_cloud(f'{self.debug_dir}/init_center.ply', pcd)

    return center.reshape(3)


  def register(self, K, rgb, depth, ob_mask, ob_id=None, glctx=None, iteration=5):
    """
    Copmute pose from given pts to self.pcd
    @pts: (N,3) np array, downsampled scene points
    
    Register (estimate initial pose) of the object in the scene.
    
    This is the main function for initial pose estimation. It performs:
    1. Depth preprocessing (erosion, bilateral filtering)
    2. Pose hypothesis generation from rotation grid and translation estimation
    3. Neural network pose refinement using the PoseRefinePredictor
    4. Neural network pose scoring using the ScorePredictor
    5. Selection of best pose based on scores
    
    The function returns the pose with the highest confidence score after refinement.
    
    Args:
        K: 3x3 camera intrinsic matrix
        rgb: RGB image (H, W, 3) 
        depth: Depth image (H, W) in meters
        ob_mask: Binary object mask (H, W)
        ob_id: Object identifier for debugging (optional)
        glctx: OpenGL/CUDA rendering context (optional)
        iteration: Number of refinement iterations to perform
        
    Returns:
        numpy.ndarray: Best 4x4 pose matrix in original object coordinates
    """
    set_seed(0)
    # logging.info('Welcome')

    if self.glctx is None:
      if glctx is None:
        self.glctx = dr.RasterizeCudaContext()
        # self.glctx = dr.RasterizeGLContext()
      else:
        self.glctx = glctx

    depth = erode_depth(depth, radius=2, device='cuda')
    # depth_v = depth>=0.001  
    # print('depth_v.sum() 3', depth_v.sum())
    # print("depth min/max:", np.min(depth), np.max(depth))

    depth = bilateral_filter_depth(depth, radius=2, device='cuda')

    if self.debug>=2:
      xyz_map = depth2xyzmap(depth, K)
      valid = xyz_map[...,2]>=0.001
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_raw.ply',pcd)
      cv2.imwrite(f'{self.debug_dir}/ob_mask.png', (ob_mask*255.0).clip(0,255))

    normal_map = None
    # print('d\n',depth)
    # print('ob_mask\n',ob_mask)
    valid = (depth>=0.001) & (ob_mask>0)
    # mask_v = ob_mask>0
    # depth_v = depth>=0.001
    # print('mask_v.sum()', mask_v.sum())
    # print('depth_v.sum()', depth_v.sum())
    # print('valid.sum()', valid.sum())
    if valid.sum()<4:
      logging.info(f'valid too small, return')
      pose = np.eye(4)
      pose[:3,3] = self.guess_translation(depth=depth, mask=ob_mask, K=K)
      return pose

    if self.debug>=2:
      imageio.imwrite(f'{self.debug_dir}/color.png', rgb)
      cv2.imwrite(f'{self.debug_dir}/depth.png', (depth*1000).astype(np.uint16))
      valid = xyz_map[...,2]>=0.001
      pcd = toOpen3dCloud(xyz_map[valid], rgb[valid])
      o3d.io.write_point_cloud(f'{self.debug_dir}/scene_complete.ply',pcd)

    self.H, self.W = depth.shape[:2]
    self.K = K
    self.ob_id = ob_id
    self.ob_mask = ob_mask

    poses = self.generate_random_pose_hypo(K=K, rgb=rgb, depth=depth, mask=ob_mask, scene_pts=None)
    poses = poses.data.cpu().numpy()
    # logging.info(f'poses:{poses.shape}')
    center = self.guess_translation(depth=depth, mask=ob_mask, K=K)

    poses = torch.as_tensor(poses, device='cuda', dtype=torch.float)
    poses[:,:3,3] = torch.as_tensor(center.reshape(1,3), device='cuda')

    add_errs = self.compute_add_err_to_gt_pose(poses)
    # logging.info(f"after viewpoint, add_errs min:{add_errs.min()}")

    xyz_map = depth2xyzmap(depth, K)
    poses, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, xyz_map=xyz_map, glctx=self.glctx, mesh_diameter=self.diameter, iteration=iteration, get_vis=self.debug>=2)
    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_refiner.png', vis)

    scores, vis = self.scorer.predict(mesh=self.mesh, rgb=rgb, depth=depth, K=K, ob_in_cams=poses.data.cpu().numpy(), normal_map=normal_map, mesh_tensors=self.mesh_tensors, glctx=self.glctx, mesh_diameter=self.diameter, get_vis=self.debug>=2)
    if vis is not None:
      imageio.imwrite(f'{self.debug_dir}/vis_score.png', vis)

    add_errs = self.compute_add_err_to_gt_pose(poses)
    # logging.info(f"final, add_errs min:{add_errs.min()}")

    ids = torch.as_tensor(scores).argsort(descending=True)
    # logging.info(f'sort ids:{ids}')
    scores = scores[ids]
    poses = poses[ids]

    # logging.info(f'sorted scores:{scores}')

    best_pose = poses[0]@self.get_tf_to_centered_mesh()
    self.pose_last = poses[0]
    self.best_id = ids[0]

    self.poses = poses
    self.scores = scores

    return best_pose.data.cpu().numpy()


  def compute_add_err_to_gt_pose(self, poses):
    """
    @poses: wrt. the centered mesh
    Compute ADD (Average Distance of Model Points) error with respect to ground truth pose.
    
    This function would normally compute the ADD error metric between predicted poses
    and ground truth pose by:
    1. Transforming model points using predicted and ground truth poses
    2. Computing average distance between corresponding transformed points
    3. Returning error values for pose evaluation
    
    Currently returns dummy values (-1) as ground truth poses are not available.
    Used primarily for debugging and evaluation when ground truth is known.
    
    Args:
        poses: Array of predicted 4x4 pose matrices with respect to centered mesh
        
    Returns:
        torch.Tensor: ADD error values for each pose (currently dummy -1 values)
    """
    return -torch.ones(len(poses), device='cuda', dtype=torch.float)


  def track_one(self, rgb, depth, K, iteration, extra={}):
    """
    Track object pose in subsequent frames after initial registration.
    
    This function performs pose tracking for video sequences by:
    1. Using the previous frame's pose as initialization (warm start)
    2. Preprocessing depth image (erosion, bilateral filtering)
    3. Running pose refinement for specified iterations
    4. Updating the tracked pose for next frame
    
    Tracking is more efficient than registration as it uses temporal coherence
    and doesn't need to search through the entire rotation grid.
    
    Args:
        rgb: RGB image (H, W, 3)
        depth: Depth image (H, W) in meters  
        K: 3x3 camera intrinsic matrix
        iteration: Number of refinement iterations to perform
        extra: Dictionary for additional outputs (e.g., visualization)
        
    Returns:
        numpy.ndarray: Tracked 4x4 pose matrix in original object coordinates
        
    Raises:
        RuntimeError: If called before initial pose registration
    """
    if self.pose_last is None:
      # logging.info("Please init pose by register first")
      raise RuntimeError
    logging.info("Welcome")

    depth = torch.as_tensor(depth, device='cuda', dtype=torch.float)
    depth = erode_depth(depth, radius=2, device='cuda')
    depth = bilateral_filter_depth(depth, radius=2, device='cuda')
    logging.info("depth processing done")

    xyz_map = depth2xyzmap_batch(depth[None], torch.as_tensor(K, dtype=torch.float, device='cuda')[None], zfar=np.inf)[0]

    pose, vis = self.refiner.predict(mesh=self.mesh, mesh_tensors=self.mesh_tensors, rgb=rgb, depth=depth, K=K, ob_in_cams=self.pose_last.reshape(1,4,4).data.cpu().numpy(), normal_map=None, xyz_map=xyz_map, mesh_diameter=self.diameter, glctx=self.glctx, iteration=iteration, get_vis=self.debug>=2)
    logging.info("pose done")
    if self.debug>=2:
      extra['vis'] = vis
    self.pose_last = pose
    return (pose@self.get_tf_to_centered_mesh()).data.cpu().numpy().reshape(4,4)


