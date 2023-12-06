#!/usr/bin/env python

import rospy
import numpy as np
from sensor_msgs.msg import Image
from sensor_msgs.msg import CameraInfo
from cv_bridge import CvBridge
import tf
from geometry_msgs.msg import Point
from importlib import reload
import utils; reload(utils)
from utils import *
import numpy as np
from importlib import reload
import utils2; reload(utils2)
from utils2 import *
from helpers import *
from classes import Starter
import rospkg



class ObjectDetector:
    def __init__(self):
        rospy.init_node('object_detector', anonymous=True)

        # Get the path to the ROS package
        rospack = rospkg.RosPack()
        package_path = rospack.get_path('perception')
        self.starter = Starter(package_path)
        self.bridge = CvBridge()
        self.ld = self.starter.get_laneDectection() 
        print("Ready, Done setting up")


        self.color_image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.color_image_callback)
        self.depth_image_sub = rospy.Subscriber("/camera/aligned_depth_to_color/image_raw", Image, self.depth_image_callback)

        self.fx = None
        self.fy = None
        self.cx = None
        self.cy = None

        self.camera_info_sub = rospy.Subscriber("/camera/color/camera_info", CameraInfo, self.camera_info_callback)

        self.tf_listener = tf.TransformListener()

        self.point_pub = rospy.Publisher("goal_point", Point, queue_size=10)
        self.image_pub = rospy.Publisher('detected_cup', Image, queue_size=10)

        rospy.spin()

    def camera_info_callback(self, msg):
        self.fx = msg.K[0]
        self.fy = msg.K[4]
        self.cx = msg.K[2]
        self.cy = msg.K[5]


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

            print("Real-world coordinates in odom frame: (X, Y, Z) = ({:.2f}m, {:.2f}m, {:.2f}m)".format(lcr, rcr, lco))
            print("Publishing goal point: ", lcr, rcr, lco)
            self.point_pub.publish(Point(lcr, rcr, lco))



        except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
            print("TF Error: " + e)
            return

if __name__ == '__main__':
    ObjectDetector()
