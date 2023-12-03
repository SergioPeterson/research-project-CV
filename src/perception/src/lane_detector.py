#!/usr/bin/env python

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo # For camera intrinsic parameters
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
import os
import time
import glob
import tf
from geometry_msgs.msg import Point, PointStamped
from std_msgs.msg import Header
from importlib import reload
import utils; reload(utils)
from utils import *
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
from importlib import reload
import utils2; reload(utils2)
from utils2 import *
from helpers import *
'''
Structure:
This will subsrcibe to the camera node and get the image as a ROS image and turn it into a CV2 image and then do some computation (right now it does nothing) and 
return the new image and publishs it into /detected_cup we can also publish the information about the image into /goal_point (we can rename it into anything) we can publish
the divaiation from the center and the left and right edges

This file also calibrates the camera


ToDo:
* Fix any errors we have in the notebook, make sure given the example image it works as functioned (i put some example images although i found them online not the real images)
* transfer the information from the notebook into this file, keep in mind that process_image will be called many times a second for each frame so make sure we don't have any
repeated code (D.R.Y.) and make it efficent
* you can test things out by returning the debuging image into the node /detected_cup and checking what it looks like in rviz


To work:
If working on Cory Lab
- run ros
 * source deval/setup.bash
 * catkin_make
 * roscore
- open a new tab and connect to a bot, and ssh into it
 * ssh fruitname@fruitname
 * password : fruitname2022
- open a new tab and calirate camera 
 * roslaunch turtlebot3_bringup turtlebot3_robot.launch --screen
- open a new tab and turn on the cam
 * roslaunch realsense2_camera rs_camera.launch mode:=Manual color_width:=424 color_height:=240 depth_width:=424 depth_height:=240 align_depth:=true depth_fps:=6 color_fps:=6
- open a new tab and set the target and goal(we will not need this in the final product but for now we need it since we are working with lab8 code, will fix later)
 * rosrun tf static_transform_publisher 0 0 0 0 0 0 base_footprint camera_link 100
- test code by running 
 * rosrun perception lane_detector.py


Possible erros:
- if its asking for an executable go to the rep where your py file is and write
 * chmod +x lane_detector.py
 -if its saying resence is not being detected, disconnect and recunnect

To visualize in rviz
- open a new tab in terminal and call rviz with 
 * rviz
-start by selecting a refrence frame can be anything and then open a new class  be at the hitting New at the bottom left and select 
* camera and the topic /raw_image for the video from the realsense itself
* image and the topic /detected_cup for the video after our image process
'''




class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)

        self.bridge = CvBridge()
        self.cv_color_image = None
        self.cv_depth_image = None

        self.color_image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.color_image_callback)
        self.depth_image_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_image_callback)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)

        self.tf_listener = tf.TransformListener()  # Create a TransformListener object

        # self.point_pub = rospy.Publisher("goal_point", Point, queue_size=10)
        self.image_pub = rospy.Publisher('detected_cup', Image, queue_size=10)

        rospy.spin()

    def camera_info_callback(self, msg):
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]

    def pixel_to_point(self, u, v, depth):
        X = (u - self.cx) * depth/self.fx
        Y = (v - self.cy) * depth/self.fy
        Z = depth
        return X, Y, Z

    def color_image_callback(self, msg):
        try:
            self.cv_color_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            if self.cv_depth_image is not None:
                self.process_images()

        except Exception as e:
            print("Error:", e)

    def depth_image_callback(self, msg):
        try:
            self.cv_depth_image = self.bridge.imgmsg_to_cv2(msg, "16UC1")

        except Exception as e:
            print("Error:", e)

    def process_images(self):
        try:
            # self.tf_listener.waitForTransform("/odom", "/camera_link", rospy.Time(), rospy.Duration(10.0))
            # point_odom = self.tf_listener.transformPoint("/odom", PointStamped(header=Header(stamp=rospy.Time(), frame_id="/camera_link")))
            # X_odom, Y_odom, Z_odom = point_odom.point.x, point_odom.point.y, point_odom.point.z
            # print("Real-world coordinates in odom frame: (X, Y, Z) = ({:.2f}m, {:.2f}m, {:.2f}m)".format(X_odom, Y_odom, Z_odom))
            # print("Publishing goal point: ", X_odom, Y_odom, Z_odom)
            # Publish the transformed point
            # self.point_pub.publish(Point(X_odom, Y_odom, Z_odom))

            # Overlay cup points on color image for visualization
            cup_img = self.cv_color_image.copy()

            # Convert to ROS Image message and publish
            ros_image = self.bridge.cv2_to_imgmsg(cup_img, "bgr8")
            self.image_pub.publish(ros_image)

        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("TF Error: " + e)
            return

        '''
        For detecting cup we don't need but imma keep for refrence
        # Convert the color image to HSV color space
        hsv = cv2.cvtColor(self.cv_color_image, cv2.COLOR_BGR2HSV)
        
        # To see the current HSV values in the center row of the image (where your cup should be), we will print out
        # the HSV mean of the HSV values of the center row. You should add at least +/- 10 to the current Hue value and
        # +/- 80 to the current Saturation and Value values to define your range.
        mean_center_row_hsv_val = np.mean(hsv[len(hsv)//2], axis=0)
        print("Current mean values at center row of image: ", mean_center_row_hsv_val)
        lower_hsv = np.array([70,85,65]) # TODO: Define lower HSV values for cup color
        upper_hsv = np.array([160,255,255]) # TODO: Define upper HSV values for cup color

        # TODO: Threshold the image to get only cup colors
        # HINT: Lookup cv2.inRange()
        mask = cv2.inRange(hsv,lower_hsv,upper_hsv)

        # TODO: Get the coordinates of the cup points on the mask
        # HINT: Lookup np.nonzero()
        y_coords, x_coords = np.nonzero(mask)

        # If there are no detected points, exit
        if len(x_coords) == 0 or len(y_coords) == 0:
            print("No points detected. Is your color filter wrong?")
            return

        # Calculate the center of the detected region by 
        center_x = int(np.mean(x_coords))
        center_y = int(np.mean(y_coords))

        # Fetch the depth value at the center
        depth = self.cv_depth_image[center_y, center_x]

        if self.fx and self.fy and self.cx and self.cy:
            camera_x, camera_y, camera_z = self.pixel_to_point(center_x, center_y, depth)
            camera_link_x, camera_link_y, camera_link_z = camera_z, -camera_x, -camera_y
            # Convert from mm to m
            camera_link_x /= 1000
            camera_link_y /= 1000
            camera_link_z /= 1000

            # Convert the (X, Y, Z) coordinates from camera frame to odom frame
            try:
                self.tf_listener.waitForTransform("/odom", "/camera_link", rospy.Time(), rospy.Duration(10.0))
                point_odom = self.tf_listener.transformPoint("/odom", PointStamped(header=Header(stamp=rospy.Time(), frame_id="/camera_link"), point=Point(camera_link_x, camera_link_y, camera_link_z)))
                X_odom, Y_odom, Z_odom = point_odom.point.x, point_odom.point.y, point_odom.point.z
                print("Real-world coordinates in odom frame: (X, Y, Z) = ({:.2f}m, {:.2f}m, {:.2f}m)".format(X_odom, Y_odom, Z_odom))

                if X_odom < 0.001 and X_odom > -0.001:
                    print("Erroneous goal point, not publishing - Is the cup too close to the camera?")
                else:
                    print("Publishing goal point: ", X_odom, Y_odom, Z_odom)
                    # Publish the transformed point
                    self.point_pub.publish(Point(X_odom, Y_odom, Z_odom))

                    # Overlay cup points on color image for visualization
                    cup_img = self.cv_color_image.copy()
                    cup_img[y_coords, x_coords] = [0, 0, 255]  # Highlight cup points in red
                    cv2.circle(cup_img, (center_x, center_y), 5, [0, 255, 0], -1)  # Draw green circle at center
                    
                    # # Undistort the image before processing for lane detection
                    # undist_img = h.undistort_image(cup_img, self.objpts, self.imgpts)
                    
                    # # Process image for lane detection
                    # lane_detection_result = self.lane_detector.process_image(undist_img)

                    # # Convert lane_detection_result to a ROS Image message and publish
                    # ros_lane_image = self.bridge.cv2_to_imgmsg(lane_detection_result, "bgr8")
                    # self.image_pub.publish(ros_lane_image)

                    # Convert to ROS Image message and publish
                    ros_image = self.bridge.cv2_to_imgmsg(cup_img, "bgr8")
                    self.image_pub.publish(ros_image)

            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                print("TF Error: " + e)
                return
        
        '''

if __name__ == '__main__':
    ObjectDetector()
