import rospy
import numpy as np
import open3d.visualization
import torch
import matplotlib.pyplot as plt
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from tools.realsense import RealSensePointCloudGenerator
import cv2
import open3d
from tools import data_logging
import os

from gpd_ros.msg import CloudIndexed,CloudSources
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2,PointField
from geometry_msgs.msg import Point
import sensor_msgs.point_cloud2 as pc2

rospy.init_node("human_in_loop_segmentation")
cloud_pub=rospy.Publisher("/cloud_stitched",PointCloud2,latch=True)

# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]

# Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)
def convertCloudFromOpen3dToRos(open3d_cloud, shift_to_origin=True,frame_id="odom"):
    # Set "header"
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    # Set "fields" and "cloud_data"
    points=np.asanyarray(open3d_cloud.points)
    if shift_to_origin:
        points-=np.min(points,axis=0)
    fields=FIELDS_XYZ
    cloud_data=points
    print(cloud_data)
    # create ros_cloud
    return pc2.create_cloud(header, fields, cloud_data)

segmented_pcd=open3d.io.read_point_cloud("assets/VariableFrictionHingeGrasping/human_in_the_loop_pc_segmentation20250312_210539/cropped_point_cloud.pcd")

cloud_source_msg=convertCloudFromOpen3dToRos(segmented_pcd)
cloud_pub.publish(cloud_source_msg)
print("published")
rospy.spin()