import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lanefollowing')
        qos_profile = QoSProfile(depth=10)
        
        self.img_sub = self.create_subscription(
            Image,
            '/testcar/camera_sensor/image_raw',
            self.img_sub_callback,
            qos_profile
        )
        
        self.cvbridge = CvBridge()

    def img_sub_callback(self, msg):
        frame = self.cvbridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = self.cvbridge.imgmsg_to_cv2(msg, "mono8")

        roi_top_left = (5, 290)
        roi_bottom_right = (635, 320)

        min_slope = 0.05
        max_slope = np.inf

        x_less_than_320 = -np.inf
        x_greater_than_320 = np.inf

        roi = frame[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Define the region of interest mask
        mask = np.zeros_like(edges)
        roi_vertices = np.array([[(roi_top_left[0], 0), roi_top_left, roi_bottom_right, (roi_bottom_right[0], 0)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)

        # Apply the mask to the edges
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi / 180, threshold=30, minLineLength=50, maxLineGap=20)

        blue_dot_x_coordinates = []

        # Draw the detected lines on the frame
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1)

                if min_slope < abs(slope) < max_slope:
                    cv2.line(frame, (roi_top_left[0] + x1, roi_top_left[1] + y1), (roi_top_left[0] + x2, roi_top_left[1] + y2), (0, 0, 255), 2)

                    center_x = int((x1 + x2) / 2)
                    center_y = int((roi_top_left[1] + roi_bottom_right[1]) / 2)
                    cv2.circle(frame, (roi_top_left[0] + center_x, center_y), radius=5, color=(255, 0, 0), thickness=-1)

                    # Store x-coordinate of the blue dot
                    blue_dot_x_coordinates.append(roi_top_left[0] + center_x)

        cv2.rectangle(frame, roi_top_left, roi_bottom_right, (0, 255, 0), thickness=2)

        center_x = (roi_top_left[0] + roi_bottom_right[0]) // 2
        cv2.line(frame, (center_x, roi_top_left[1]), (center_x, roi_bottom_right[1]), (203, 192, 255), thickness=2)

        if blue_dot_x_coordinates:
            x_less_than_320 = max((x for x in blue_dot_x_coordinates if x < 320), default=x_less_than_320)

        print("Maximum x-coordinate less than 320:", x_less_than_320)

        if x_less_than_320 <50:
            command = "Go left"
        elif x_less_than_320 > 160:
            command = "Go right"
        elif 50 <= x_less_than_320 <= 160:
            command = "Straight"
        else:
            command = "NOT"

        print("Command:", command)
        self.get_logger().info(f"Command: {command}")

        cv2.imshow('Video', frame)
        cv2.imshow('Video2', edges)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    lane_detection_node = LaneDetectionNode()
    rclpy.spin(lane_detection_node)
    lane_detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()