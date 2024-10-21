#This is a script to view a bag file using OpenCV

import sys
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import rosbag
import cv2  # Import the cv2 module

class ImageViewer:
    def __init__(self, bag_file, width=640, height=480):
        self.bag_file = bag_file
        self.bridge = CvBridge()
        self.width = width
        self.height = height

        # Initialize the ROS node
        rospy.init_node('bag_image_viewer', anonymous=True)

        # Open the bag file
        self.bag = rosbag.Bag(self.bag_file)

        # Subscriber for image messages
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self.image_callback)

    def image_callback(self, data):
        # Convert the ROS image message to an OpenCV image
        cv_image = self.bridge.imgmsg_to_cv2(data, desired_encoding='bgr8')

        # Resize the image to the specified width and height
        cv_image_resized = cv2.resize(cv_image, (self.width, self.height))

        # Display the image using OpenCV
        cv2.imshow("Camera Image", cv_image_resized)
        cv2.waitKey(1)

    def play_bag(self):
        # Play through the bag file messages
        for topic, msg, t in self.bag.read_messages(topics=['/camera/image_raw']):
            self.image_callback(msg)

        # Close the bag file
        self.bag.close()

if __name__ == '__main__':
    bag_filename = input("Enter the filename of the .bag file to view: ")
    if not bag_filename.endswith('.bag'):
        print("Please provide a valid .bag file.")
        sys.exit(1)

    viewer = ImageViewer(bag_filename)
    try:
        viewer.play_bag()
    except rospy.ROSInterruptException:
        pass
    finally:
        cv2.destroyAllWindows()
