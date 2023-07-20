#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import String


class RobotMotion(Node):
    def __init__(self):
        super().__init__('robot_motion')

        # Create a subscriber for cmd_vel topic
        self.cmd_vel_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_vel_callback,
            10
        )
        
        self.movement_sub = self.create_subscription(
            String,
            '/movement_cmd',
            self.movement_cmd_callback,
            10
        )
        
        self.cmd_vel_pub = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

    def cmd_vel_callback(self, msg):
        # Here you can write the code to apply the cmd_vel commands to your robot in Gazebo
        # For example, you might publish these velocities to the robot's motor controllers.
        self.cmd_vel_pub.publish(msg)
        linear_x = msg.linear.x
        angular_z = msg.angular.z

        # Print the received velocities (optional, for debugging)
        self.get_logger().info(f'Received cmd_vel: Linear X: {linear_x}, Angular Z: {angular_z}')

    def movement_cmd_callback(self, msg):
        movement_command = msg.data
        
        if movement_command == "Go left":
            # Set appropriate linear and angular velocities for left turn
            linear_x = 0.2
            angular_z = 1.0
        elif movement_command == "Go right":
            # Set appropriate linear and angular velocities for right turn
            linear_x = 0.2
            angular_z = -1.0
        elif movement_command == "Straight":
            # Set appropriate linear and angular velocities for straight motion
            linear_x = 10.0
            angular_z = 0.0
        elif movement_command == "Stop":
            # Set linear and angular velocities to stop the robot
            linear_x = 0.0
            angular_z = 0.0
        else:
            # Handle other cases or default behavior
            linear_x = 0.0
            angular_z = 0.0

        # Publish the calculated velocities to cmd_vel topic
        twist_msg = Twist()
        twist_msg.linear.x = linear_x
        twist_msg.angular.z = angular_z
        self.cmd_vel_pub.publish(twist_msg)
    
def main(args=None):
    rclpy.init(args=args)
    robot_motion = RobotMotion()
    try:
        rclpy.spin(robot_motion)
    except KeyboardInterrupt:
        pass
    finally:
        robot_motion.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
