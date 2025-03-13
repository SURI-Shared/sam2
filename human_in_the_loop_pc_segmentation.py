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

outfolder=data_logging.make_timestamped_folder("assets/VariableFrictionHingeGrasping","human_in_the_loop_pc_segmentation")
data_logging.write_README(outfolder,__file__,seed=seed,background_distance=background_distance,sam2_checkpoint=sam2_checkpoint,model_cfg=model_cfg)

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
image.save(os.path.join(outfolder,"color.png"))
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
np.save(os.path.join(outfolder,"raw_depth.npy"),raw_depth)

o3dpcd=camera.o3dpcd_from_color_and_depth_arrays(masked_image,raw_depth,False)

open3d.visualization.draw([o3dpcd])
open3d.t.io.write_point_cloud(os.path.join(outfolder,"segmented_pointcloud.pcd"),o3dpcd)

masked_depth=raw_depth.copy()
bool_mask=masks[0].astype(np.bool)
masked_depth[np.logical_not(bool_mask)]=2*background_distance
segmented_pcd=camera.o3dpcd_from_color_and_depth_arrays(masked_image,masked_depth,False)
open3d.t.io.write_point_cloud(os.path.join(outfolder,"cropped_point_cloud.pcd"),segmented_pcd)