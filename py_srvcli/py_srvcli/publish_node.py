# Publish node for multi-object pose estimation
# ros2 topic echo --once  /cmd_vel > cmd_vel.yaml                using this command to see the message once
# ros2 run py_srvcli pose_publisher --yaml_file /path/to/your/ycbv_res.yml --rate 10.0
#ros2 topic echo /object_poses       #Listen to published poses
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion
from std_msgs.msg import Header
import yaml
import numpy as np
import os
import argparse
from scipy.spatial.transform import Rotation


class PosePublisher(Node):
    def __init__(self, yaml_file_path, publish_rate=10.0):
        super().__init__('pose_publisher')
        
        # Create publisher for poses
        self.pose_publisher = self.create_publisher(PoseStamped, '/object_poses', 10)
        
        # Set publishing rate
        self.timer = self.create_timer(1.0/publish_rate, self.publish_poses)
        
        # Load YAML data
        self.yaml_file_path = yaml_file_path
        self.pose_data = self.load_poses_from_yaml()
        
        # Iterator for cycling through poses
        self.current_video_idx = 0
        self.current_frame_idx = 0
        self.current_object_idx = 0
        
        # Extract all poses into a flat list for easy iteration
        self.all_poses = self.extract_all_poses()
        self.pose_index = 0
        
        self.get_logger().info(f'Pose Publisher initialized. Loaded {len(self.all_poses)} poses from {yaml_file_path}')
        self.get_logger().info(f'Publishing poses at {publish_rate} Hz on topic: /object_poses')

    def load_poses_from_yaml(self):
        """Load pose data from YAML file"""
        try:
            with open(self.yaml_file_path, 'r') as f:
                data = yaml.safe_load(f)
            return data
        except FileNotFoundError:
            self.get_logger().error(f'YAML file not found: {self.yaml_file_path}')
            return None
        except Exception as e:
            self.get_logger().error(f'Error loading YAML file: {str(e)}')
            return None

    def extract_all_poses(self):
        """Extract all poses from the YAML data structure"""
        poses = []
        
        if self.pose_data is None:
            return poses
        
        # Check if data is a simple 4x4 matrix (list of 4 lists with 4 elements each)
        if isinstance(self.pose_data, list) and len(self.pose_data) == 4:
            if all(isinstance(row, list) and len(row) == 4 for row in self.pose_data):
                # Single 4x4 pose matrix
                pose_matrix = np.array(self.pose_data)
                poses.append({
                    'video_id': 0,
                    'frame_id': 0,
                    'object_id': 0,
                    'pose_matrix': pose_matrix
                })
                return poses
        
        # Check if data is a list of 4x4 matrices
        if isinstance(self.pose_data, list) and len(self.pose_data) > 0:
            if isinstance(self.pose_data[0], list) and len(self.pose_data[0]) == 4:
                # Already handled above as single matrix
                pass
            else:
                # List of pose matrices
                for idx, pose_matrix in enumerate(self.pose_data):
                    if isinstance(pose_matrix, list) and len(pose_matrix) == 4:
                        pose_matrix = np.array(pose_matrix)
                        poses.append({
                            'video_id': 0,
                            'frame_id': idx,
                            'object_id': 0,
                            'pose_matrix': pose_matrix
                        })
                return poses
        
        # Check if data is a dictionary (nested structure: video -> frame -> object)
        if isinstance(self.pose_data, dict):
            for video_id, video_data in self.pose_data.items():
                if isinstance(video_data, dict):
                    for frame_id, frame_data in video_data.items():
                        if isinstance(frame_data, dict):
                            for object_id, pose_matrix in frame_data.items():
                                if isinstance(pose_matrix, list):
                                    pose_matrix = np.array(pose_matrix)
                                poses.append({
                                    'video_id': video_id,
                                    'frame_id': frame_id,
                                    'object_id': object_id,
                                    'pose_matrix': pose_matrix
                                })
        
        return poses

    def matrix_to_pose_msg(self, pose_matrix):
        """Convert 4x4 transformation matrix to ROS2 Pose message"""
        if isinstance(pose_matrix, list):
            pose_matrix = np.array(pose_matrix)
        
        # Extract translation
        translation = pose_matrix[:3, 3]
        
        # Extract rotation matrix and convert to quaternion
        rotation_matrix = pose_matrix[:3, :3]
        rotation = Rotation.from_matrix(rotation_matrix)
        quaternion = rotation.as_quat()  # [x, y, z, w]
        
        # Create Pose message
        pose = Pose()
        pose.position = Point(x=float(translation[0]), 
                             y=float(translation[1]), 
                             z=float(translation[2]))
        pose.orientation = Quaternion(x=float(quaternion[0]),
                                    y=float(quaternion[1]),
                                    z=float(quaternion[2]),
                                    w=float(quaternion[3]))
        
        return pose

    def publish_poses(self):
        """Publish poses from the YAML data"""
        if not self.all_poses:
            self.get_logger().warn('No poses to publish')
            return
        
        # Get current pose data
        current_pose_data = self.all_poses[self.pose_index]
        
        # Create PoseStamped message
        pose_msg = PoseStamped()
        pose_msg.header = Header()
        pose_msg.header.stamp = self.get_clock().now().to_msg()
        pose_msg.header.frame_id = 'camera_frame'  # Adjust frame_id as needed
        
        # Convert transformation matrix to Pose
        try:
            pose_msg.pose = self.matrix_to_pose_msg(current_pose_data['pose_matrix'])
            
            # Publish the pose
            self.pose_publisher.publish(pose_msg)
            
            # Log information
            self.get_logger().info(f'Published pose for object {current_pose_data["object_id"]} '
                                 f'from video {current_pose_data["video_id"]} '
                                 f'frame {current_pose_data["frame_id"]} '
                                 f'({self.pose_index + 1}/{len(self.all_poses)})')
            
            # Move to next pose
            self.pose_index = (self.pose_index + 1) % len(self.all_poses)
            
        except Exception as e:
            self.get_logger().error(f'Error publishing pose: {str(e)}')

    def publish_single_pose(self, video_id=None, frame_id=None, object_id=None):
        """Publish a specific pose by identifiers"""
        for pose_data in self.all_poses:
            if ((video_id is None or pose_data['video_id'] == video_id) and
                (frame_id is None or pose_data['frame_id'] == frame_id) and
                (object_id is None or pose_data['object_id'] == object_id)):
                
                pose_msg = PoseStamped()
                pose_msg.header = Header()
                pose_msg.header.stamp = self.get_clock().now().to_msg()
                pose_msg.header.frame_id = 'camera_frame'
                
                try:
                    pose_msg.pose = self.matrix_to_pose_msg(pose_data['pose_matrix'])
                    self.pose_publisher.publish(pose_msg)
                    
                    self.get_logger().info(f'Published specific pose for object {object_id} '
                                         f'from video {video_id} frame {frame_id}')
                    return True
                except Exception as e:
                    self.get_logger().error(f'Error publishing specific pose: {str(e)}')
                    return False
        
        self.get_logger().warn(f'Pose not found for video:{video_id}, frame:{frame_id}, object:{object_id}')
        return False


def main(args=None):
    """Main function to start the pose publisher node"""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='ROS2 Pose Publisher for FoundationPose YAML files')
    parser.add_argument('--yaml_file', type=str, 
                       default='/home/student/Desktop/perception/FoundationPose/debug/ycbv_res.yml',
                       help='Path to YAML file containing poses')
    parser.add_argument('--rate', type=float, default=1.0,
                       help='Publishing rate in Hz (default: 1.0)')
    
    # Parse known args to avoid conflicts with ROS2
    parsed_args, unknown = parser.parse_known_args()
    
    # Initialize ROS2
    rclpy.init(args=args)
    
    # Check if YAML file exists
    if not os.path.exists(parsed_args.yaml_file):
        print(f"Error: YAML file not found: {parsed_args.yaml_file}")
        print("Please run the FoundationPose estimation first to generate the YAML file")
        return
    
    # Create and run the pose publisher node
    try:
        pose_publisher = PosePublisher(parsed_args.yaml_file, parsed_args.rate)
        rclpy.spin(pose_publisher)
    except KeyboardInterrupt:
        print("\nShutting down pose publisher...")
    finally:
        # Clean up
        if 'pose_publisher' in locals():
            pose_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
