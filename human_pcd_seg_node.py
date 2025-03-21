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
from geometry_msgs.msg import TransformStamped,Transform
import sensor_msgs.point_cloud2 as pc2
import tf2_msgs.msg
import tf2_ros
from scipy.spatial.transform import Rotation

camera_frame="camera_depth_optical_frame"
fixed_frame="world"

def transform_msg_to_homogeneous(msg:Transform):
    rotation=Rotation.from_quat([msg.rotation.x,msg.rotation.y,msg.rotation.z,msg.rotation.w])
    mat=np.zeros((4,4))
    mat[3,3]=1
    mat[:,:3]=np.array([msg.translation.x,msg.translation.y,msg.translation.z])
    mat[:3,:3]=rotation.as_matrix()
    return mat

rospy.init_node("human_in_loop_segmentation")
cloud_pub=rospy.Publisher("/cloud_stitched",PointCloud2,latch=True)
full_cloud_pub=rospy.Publisher("/raw_cloud",PointCloud2,latch=True)
pub_tf = rospy.Publisher("/tf", tf2_msgs.msg.TFMessage, queue_size=1)
tfBuffer=tf2_ros.Buffer()
listener=tf2_ros.TransformListener(tfBuffer)
camera_to_world=tfBuffer.lookup_transform(camera_frame,fixed_frame,rospy.Time(0),rospy.Duration(10))
camera_to_world_g=transform_msg_to_homogeneous(camera_to_world.transform)
camera_to_world_R=camera_to_world_g[:3,:3]
plt.ion()

device = torch.device("cuda")
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
# turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

seed=3
background_distance=3

sam2_checkpoint = "checkpoints/sam2.1_hiera_tiny.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"

sam2 = build_sam2(model_cfg, sam2_checkpoint, device=device, apply_postprocessing=False)
predictor = SAM2ImagePredictor(sam2)

camera=RealSensePointCloudGenerator(background_distance)
got_image=False
print("press 'c' to capture image or 'q' to quit")
while not got_image:
    timestamp,color,depth=camera.wait_for_rgbd()
    color_array=np.asanyarray(color.get_data())
    cv2.imshow("Live",color_array)
    if cv2.waitKey(10)==ord("c"):
        got_image=True
        camera.stop_capture()
        print('Image captured')
    if cv2.waitKey(10)==ord('q'):
        camera.stop_capture()
        print("Quitting")
        exit()

cv2.destroyAllWindows()
image=Image.fromarray(color_array)
image = np.array(image.convert("RGB"))

predictor.set_image(image)

rng=np.random.default_rng(seed)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([rng.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)
    return mask_image

def show_points(coords, labels, ax, marker_size=375):
    # import pdb
    # pdb.set_trace()
    pos_points = coords[labels>=1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))

def add_mask(ax,mask,point_coords,input_labels,borders=True):
    masked_image=show_mask(mask,ax,borders=borders)
    if point_coords is not None:
        assert input_labels is not None
        show_points(point_coords, input_labels, plt.gca())
    return masked_image

fig=plt.figure(figsize=(20, 20))
ax=fig.gca()
ax.imshow(image)
ax.axis('off')

item_idx=1
print("Click on an object to prompt SAM2")
pt=plt.ginput(1)

masks, scores, logits = predictor.predict(
    point_coords=pt,
    point_labels=[item_idx],
    multimask_output=True,
)
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

masked_image=add_mask(ax, masks[0], point_coords=np.array(pt), input_labels=np.array([item_idx],dtype=np.int64), borders=True)

raw_depth=camera.depth_array_from_depth_frame(depth)

o3dpcd=camera.o3dpcd_from_color_and_depth_arrays(masked_image,raw_depth,False)

masked_depth=raw_depth.copy()
bool_mask=masks[0].astype(np.bool)
masked_depth[np.logical_not(bool_mask)]=2*background_distance
segmented_pcd=camera.o3dpcd_from_color_and_depth_arrays(masked_image,masked_depth,False)
all_points_in_camera=o3dpcd.point.positions.numpy()
segmented_points_in_camera=segmented_pcd.point.positions.numpy()

# The data structure of each point in ros PointCloud2: 16 bits = x + y + z + rgb
FIELDS_XYZ = [
    PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
    PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
    PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
]

def represent_pointcloud_in_frame(points,g_current2desired):
    R=g_current2desired[:3,:3]
    t=g_current2desired[:3,3]
    return points@R.T+t

# Convert the datatype of point cloud from Open3D to ROS PointCloud2 (XYZRGB only)
def convertCloudFromOpen3dToRos(points_in_camera, should_shift=True,world_base_frame=False, frame_id="object"):
    # Set "header"
    header = Header()
    header.stamp = rospy.Time.now()
    header.frame_id = frame_id

    # Set "fields" and "cloud_data"
    if should_shift:
        #create "object frame"
        t = TransformStamped()
        if world_base_frame:
            t.header.frame_id = fixed_frame
        else:
            t.header.frame_id = camera_frame
        t.header.stamp = rospy.Time.now()
        t.child_frame_id = frame_id
        if world_base_frame:
            points_in_world=represent_pointcloud_in_frame(points_in_camera,camera_to_world_g)
            shift=np.min(points_in_world,axis=0)
            points=points_in_world-shift
            t.transform.translation.x = shift[0]
            t.transform.translation.y = shift[1]
            t.transform.translation.z = shift[2]
        else:
            shift=np.min(points_in_camera,axis=0)
            points=points_in_camera-shift

            t.transform.translation.x = shift[0]
            t.transform.translation.y = shift[1]
            t.transform.translation.z = shift[2]

        t.transform.rotation.x = 0
        t.transform.rotation.y = 0
        t.transform.rotation.z = 0
        t.transform.rotation.w = 1

        tfm = tf2_msgs.msg.TFMessage([t])
        pub_tf.publish(tfm)
    else:
        shift=0
        points=points_in_camera

    fields=FIELDS_XYZ
    cloud_data=points

    # create ros_cloud
    return pc2.create_cloud(header, fields, cloud_data),shift

cloud_source_msg,shift=convertCloudFromOpen3dToRos(segmented_points_in_camera,should_shift=True, frame_id="debris")
cloud_pub.publish(cloud_source_msg)
full_cloud,_=convertCloudFromOpen3dToRos(all_points_in_camera,should_shift=False, frame_id=camera_frame)
full_cloud_pub.publish(full_cloud)

segmented_pcd_centroid_in_camera=np.mean(segmented_points_in_camera,axis=0)
t_in_camera= TransformStamped()
t_in_camera.header.frame_id=camera_frame
t_in_camera.header.stamp=rospy.Time.now()
t_in_camera.child_frame_id="segmented_pcd_centroid_in_camera"
t_in_camera.transform.translation.x=segmented_pcd_centroid_in_camera[0]
t_in_camera.transform.translation.y=segmented_pcd_centroid_in_camera[1]
t_in_camera.transform.translation.z=segmented_pcd_centroid_in_camera[2]
t_in_camera.transform.rotation.x = 0
t_in_camera.transform.rotation.y = 0
t_in_camera.transform.rotation.z = 0
t_in_camera.transform.rotation.w = 1

# segmented_pcd_centroid_in_world=camera_to_world_R@segmented_pcd_centroid_in_camera+camera_to_world_g[:3,3]
# t_in_world= TransformStamped()
# t_in_world.header.frame_id=fixed_frame
# t_in_world.header.stamp=rospy.Time.now()
# t_in_world.child_frame_id="segmented_pcd_centroid_in_world"
# t_in_world.transform.translation.x=segmented_pcd_centroid_in_world[0]
# t_in_world.transform.translation.y=segmented_pcd_centroid_in_world[1]
# t_in_world.transform.translation.z=segmented_pcd_centroid_in_world[2]
# t_in_world.transform.rotation.x = 0
# t_in_world.transform.rotation.y = 0
# t_in_world.transform.rotation.z = 0
# t_in_world.transform.rotation.w = 1

tfm = tf2_msgs.msg.TFMessage([t_in_camera])
pub_tf.publish(tfm)
rospy.spin()