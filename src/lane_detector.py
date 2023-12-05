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
from classes import AdvancedLaneDetectorWithMemory,Starter
import rospkg



class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)

        self.bridge = CvBridge()
        self.cv_color_image = None
        self.cv_depth_image = None

        # Get the path to the ROS package
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('perception')
        self.starter = Starter(package_path)

        self.ld = self.starter.get_laneDectection() 
        print("Ready, Done setting up")


        self.color_image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.color_image_callback)
        self.depth_image_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_image_callback)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)

        self.tf_listener = tf.TransformListener()  # Create a TransformListener object

        self.point_pub = rospy.Publisher("goal_point", Point, queue_size=10)
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
            img = self.cv_color_image.copy()
            proc_img,lcr, rcr, lco = self.ld.process_image(img)
            ros_image = self.bridge.cv2_to_imgmsg(proc_img, "bgr8")
            self.image_pub.publish(ros_image)


            # Publishing the information we need
            # self.tf_listener.waitForTransform("/odom", "/camera_link", rospy.Time(), rospy.Duration(10.0))
            # point_odom = self.tf_listener.transformPoint("/odom", PointStamped(header=Header(stamp=rospy.Time(), frame_id="/camera_link")))
            X_odom, Y_odom, Z_odom = lcr, rcr, lco
            print("Real-world coordinates in odom frame: (X, Y, Z) = ({:.2f}m, {:.2f}m, {:.2f}m)".format(X_odom, Y_odom, Z_odom))
            print("Publishing goal point: ", X_odom, Y_odom, Z_odom)
            self.point_pub.publish(Point(X_odom, Y_odom, Z_odom))



        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("TF Error: " + e)
            return

if __name__ == '__main__':
    ObjectDetector()
