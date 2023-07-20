import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from std_msgs.msg import String
import time

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

        # Create a publisher for the movement commands
        self.movement_pub = self.create_publisher(
            String,
            '/movement_cmd',
            qos_profile
        )
        
        self.cvbridge = CvBridge()
        self.lane_detection = None

    def img_sub_callback(self, msg):
        frame = self.cvbridge.imgmsg_to_cv2(msg, 'bgr8')

        if self.lane_detection is None:
            self.lane_detection = LaneDetection(frame)
        else:
            self.lane_detection.frame = frame

        self.lane_detection.detect_lanes()
        
        # Publish the movement command
        command = self.lane_detection.get_movement_command()
        msg = String()
        msg.data = command
        self.movement_pub.publish(msg)

class LaneDetection:
    def __init__(self, frame):
        self.frame = frame
        self.stop_detection = False
        self.interrupt_time = None
        self.allow_direction_detection = True
        self.allow_stop_line_detection = True

    def detect_lanes(self):
        gray = cv2.cvtColor(self.frame, cv2.COLOR_BGR2GRAY)

        roi_top_left = (5, 50)
        roi_bottom_right = (635, 130)

        min_slope = -np.inf
        max_slope = np.inf

        x_less_than_320 = -np.inf
        x_greater_than_320 = np.inf

        roi = gray[roi_top_left[1]:roi_bottom_right[1], roi_top_left[0]:roi_bottom_right[0]]
        blurred = cv2.GaussianBlur(roi, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)

        # Define the region of interest mask
        mask = np.zeros_like(edges)
        roi_vertices = np.array([[(roi_top_left[0], 0), roi_top_left, roi_bottom_right, (roi_bottom_right[0], 0)]], dtype=np.int32)
        cv2.fillPoly(mask, roi_vertices, 255)

        # Apply the mask to the edges
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(masked_edges, rho=1, theta=np.pi / 180, threshold=20, minLineLength=20, maxLineGap=50)

        blue_dot_x_coordinates = []
        line_angles = []  # Store line angles for horizontal line detection

        # Draw the detected lines on the frame
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1)

                if min_slope < abs(slope) < max_slope:
                    cv2.line(self.frame, (roi_top_left[0] + x1, roi_top_left[1] + y1), (roi_top_left[0] + x2, roi_top_left[1] + y2), (0, 0, 255), 2)

                    center_x = int((x1 + x2) / 2)
                    center_y = int((roi_top_left[1] + roi_bottom_right[1]) / 2)
                    cv2.circle(self.frame, (roi_top_left[0] + center_x, center_y), radius=5, color=(255, 0, 0), thickness=-1)

                    # Store x-coordinate of the blue dot
                    blue_dot_x_coordinates.append(roi_top_left[0] + center_x)

                    # Calculate line angle
                    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                    line_angles.append(angle)

        cv2.rectangle(self.frame, roi_top_left, roi_bottom_right, (0, 255, 0), thickness=2)

        center_x = (roi_top_left[0] + roi_bottom_right[0]) // 2
        cv2.line(self.frame, (center_x, roi_top_left[1]), (center_x, roi_bottom_right[1]), (203, 192, 255), thickness=2)

        if self.allow_direction_detection:
            if blue_dot_x_coordinates:
                x_less_than_320 = max((x for x in blue_dot_x_coordinates if x < 320), default=x_less_than_320)

            print("Maximum x-coordinate less than 320:", x_less_than_320)

            if x_less_than_320 < 50:
                self.movement_command = "Go left"
            elif x_less_than_320 > 160:
                self.movement_command = "Go right"
            elif 50 <= x_less_than_320 <= 160:
                self.movement_command = "Straight"
            else:
                self.movement_command = "NOT"

            print("Command:", self.movement_command)

        if self.allow_stop_line_detection:
            # Check for horizontal lines
            if any(315 < roi_top_left[0] + center_x < 325 and (abs(angle) > 160 or abs(angle) < 20) for center_x, angle in zip(blue_dot_x_coordinates, line_angles)):
                self.stop_detection = True
                self.allow_direction_detection = False
                self.allow_stop_line_detection = False
                self.movement_command = "Stop"
                print("Command:", self.movement_command)

        if self.stop_detection:
            if self.interrupt_time is None:
                self.interrupt_time = time.time()
            elapsed_time = time.time() - self.interrupt_time

            if elapsed_time > 4 and elapsed_time <= 8:  # Allow direction detection after 4 seconds
                self.allow_direction_detection = True
            elif elapsed_time > 8:  # Allow stop line detection after 8 seconds
                self.allow_stop_line_detection = True

            if elapsed_time > 8:  # Stop detection for 4 seconds
                self.stop_detection = False
                self.interrupt_time = None
                
        cv2.imshow('Video', self.frame)
        cv2.waitKey(1)
        
    def get_movement_command(self):
        return self.movement_command

def main(args=None):
    rclpy.init(args=args)
    lane_detection_node = LaneDetectionNode()
    rclpy.spin(lane_detection_node)
    lane_detection_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
