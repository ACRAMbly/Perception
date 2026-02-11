import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge
import numpy as np
import cv2
import os
from datetime import datetime
from example_interfaces.srv import Trigger
import threading




class MultiObjectPoseNode(Node):
    def __init__(self):
        super().__init__('multi_object_pose_node')
        # Subscriptions (raw messages only; images converted in timer)
        self.create_subscription(CameraInfo, '/camera/camera/color/camera_info', self.camera_info_callback, 10)
        self.create_subscription(Image, '/camera/camera/color/image_raw', self._color_msg_cb, 10)
        self.create_subscription(Image, '/camera/camera/depth/image_rect_raw', self._depth_msg_cb, 10)

        # Service: Save the latest color and depth images from topics
        self.srv = self.create_service(Trigger, 'save_camera_frames', self.get_frame_and_save_callback)

        # For image conversion
        self.bridge = CvBridge()
        self.color_image = None  # Latest processed cv image (color)
        self.depth_image = None  # Latest processed cv image (depth)
        self.cam_K = None

        # Raw message buffers (updated by subscriptions)
        self._latest_color_msg = None
        self._latest_depth_msg = None
        self._last_color_stamp = None
        self._last_depth_stamp = None
        self._rlock = threading.RLock()

        # Timer to update frames every 100 ms
        # self.create_timer(0.1, self.timer_callback)

    def process_latest_images(self):
        # Convert new raw messages (if any) to cv images at a controlled rate.
        with self._rlock:
            if self._latest_color_msg is not None:
                stamp = (self._latest_color_msg.header.stamp.sec, self._latest_color_msg.header.stamp.nanosec)
                if stamp != self._last_color_stamp:
                    try:
                        self.color_image = self.bridge.imgmsg_to_cv2(self._latest_color_msg, "bgr8")
                        self._last_color_stamp = stamp
                    except Exception as e:
                        self.get_logger().warn(f"Color conversion failed: {e}")
            if self._latest_depth_msg is not None:
                stamp = (self._latest_depth_msg.header.stamp.sec, self._latest_depth_msg.header.stamp.nanosec)
                if stamp != self._last_depth_stamp:
                    try:
                        self.depth_image = self.bridge.imgmsg_to_cv2(self._latest_depth_msg, "8UC1") / 1e3
                        self._last_depth_stamp = stamp
                    except Exception as e:
                        self.get_logger().warn(f"Depth conversion failed: {e}")

    def _color_msg_cb(self, msg):
        with self._rlock:
            self._latest_color_msg = msg

    def _depth_msg_cb(self, msg):
        with self._rlock:
            self._latest_depth_msg = msg

    def get_frame_and_save_callback(self, request, response):
        # Call timer callback to ensure latest images are processed
        self.process_latest_images()
        #TODO: try and see if this works
        #TODO: trying reading image from the client side

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        color_filename = f"color/frame_{timestamp}.png"
        depth_filename = f"depth/frame_{timestamp}.png"

        if self.color_image is not None:
            self._ensure_dirs()
            self.save_image(self.color_image, color_filename)
        else:
            self.get_logger().warn("No color image available to save.")
            response.success = False
            response.message = f"Failed to save color to {color_filename}."
            return response
        if self.depth_image is not None:
            self._ensure_dirs()
            self.save_image(self.depth_image, depth_filename)
        else:
            self.get_logger().warn("No depth image available to save.")
            response.success = False
            response.message = f"Failed to save depth to {depth_filename}."
            return response

        response.success = True
        response.message = f"Saved color to {color_filename}, depth to {depth_filename}"
        self.color_image = None
        self.depth_image = None
        # place holder to call the foundation pose script
        # Execute Calling the foundation pose script
        # os.system("bash /home/student/Desktop/perception/FoundationPose/rosbag_testing/calling_pipeline.sh")



        return response


    def camera_info_callback(self, msg):
        if self.cam_K is None:
            self.cam_K = np.array(msg.k).reshape((3, 3))
            self.get_logger().info(f"Camera Intrinsics: {self.cam_K}")


    def save_image(self, image, filename):
        cv2.imwrite(filename, image)
        self.get_logger().info(f"Saved image: {filename}")

    def _ensure_dirs(self):
        for d in ("color", "depth"):
            if not os.path.exists(d):
                try:
                    os.makedirs(d, exist_ok=True)
                except Exception as e:
                    self.get_logger().error(f"Failed creating directory {d}: {e}")

def main(args=None):
    try:
        rclpy.init(args=args)
        node = MultiObjectPoseNode()
        rclpy.spin(node)
    except Exception as e:
        node.get_logger().error(f"Exception in main: {e}")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()