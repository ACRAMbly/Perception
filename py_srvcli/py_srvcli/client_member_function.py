import rclpy
from rclpy.node import Node
from example_interfaces.srv import Trigger


class SaveCameraFramesClient(Node):
    def __init__(self):
        super().__init__('save_camera_frames_client')
        self.cli = self.create_client(Trigger, 'save_camera_frames')
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('service not available, waiting again...')
        self.req = Trigger.Request()

    def send_request(self):
        return self.cli.call_async(self.req)


def main():
    rclpy.init()
    client = SaveCameraFramesClient()
    future = client.send_request()
    rclpy.spin_until_future_complete(client, future)
    response = future.result()
    if response is not None:
        client.get_logger().info(f"Service call success: {response.success}, message: {response.message}")
    else:
        client.get_logger().error("Service call failed or no response received.")
    client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()