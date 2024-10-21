#This script is for recording a video with a single camera live feed

import rospy
from sensor_msgs.msg import Image
import rosbag

def image_callback(data):
    # Record the image to the bag file
    bag.write('/camera/image_raw', data)

def main():
    # Initialize the ROS node
    rospy.init_node('bag_recorder', anonymous=True)
    
    # Open a bag file for recording
    global bag
    bag = rosbag.Bag('my_camera_data.bag', 'w')

    # Subscribe to the camera image topic
    rospy.Subscriber('/camera/image_raw', Image, image_callback)

    try:
        # Keep the node running
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        # Close the bag file when shutting down
        bag.close()

if __name__ == '__main__':
    main()
