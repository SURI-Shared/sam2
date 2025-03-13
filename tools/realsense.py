'''
Created on Sep 13, 2022

@author: ggutow
'''
import os
import shutil
import pickle

import pyrealsense2 as rs
import time
import numpy as np

import open3d as o3d
import cv2

from collections import namedtuple
Intrinsics=namedtuple("Intrinsics",("fx","fy","ppx","ppy","model","coeffs","height","width"))

def pointset_from_image(depth_image,intrinsic_mat,background_depth,dtype=np.float64):
    z=depth_image.copy()
    height,width=z.shape
    u=np.tile(np.arange(width)[np.newaxis,:],(height,1))
    v=np.tile(np.arange(height)[:,np.newaxis],(1,width))
    cx=intrinsic_mat[0,2]
    cy=intrinsic_mat[1,2]
    fx=intrinsic_mat[0,0]
    fy=intrinsic_mat[1,1]
    x=(u-cx)*z/fx
    y=(v-cy)*z/fy
    points=np.stack((x,y,z),2)
    keep=np.logical_and(z<=background_depth,z>0)
    return points[keep].astype(dtype)

def record_frames(dur):
    pipeline=rs.pipeline()
    align=rs.align(rs.stream.color)
    #device=rs.config().resolve(rs.pipeline_wrapper(pipeline)).get_device()
    profile=pipeline.start()
    start=time.perf_counter()
    dimages=[]
    cimages=[]
    try:
        while time.perf_counter()-start<dur:
            frames=pipeline.wait_for_frames()
            aligned_frames=align.process(frames)
            depthf=aligned_frames.get_depth_frame()
            colorf=aligned_frames.get_color_frame()
            if not depthf or not colorf:
                continue
            dimages.append(np.array(depthf.get_data()))
            cimages.append(np.array(colorf.get_data()))
    finally:
        pipeline.stop()
        print(len(dimages))
    return np.array(cimages),np.array(dimages),profile

def record_with_decimation(dur):
    pipeline=rs.pipeline()
    profile=pipeline.start()
    depth_scale=profile.get_device().first_depth_sensor().get_depth_scale()
    decimation=rs.decimation_filter()
    decimation.set_option(rs.option.filter_magnitude, 3)

    start=time.perf_counter()
    dimages=[]
    cimages=[]
    try:
        while time.perf_counter()-start<dur:
            frames=pipeline.wait_for_frames()
            depthf=frames.get_depth_frame()
            depthf=decimation.process(depthf)
            depth_intrinsics = rs.video_stream_profile(
            depthf.profile).get_intrinsics()
            colorf=frames.get_color_frame()
            color_intrinsics=rs.video_stream_profile(colorf.profile).get_intrinsics()
            if not depthf or not colorf:
                continue
            dimages.append(np.array(depthf.get_data()))
            cimages.append(np.array(colorf.get_data()))
    finally:
        pipeline.stop()
        print(len(dimages))
    return np.array(cimages),np.array(dimages),color_intrinsics,depth_intrinsics,depth_scale

def record_and_save_sequence(dur,folder,filename,decimation_factor,xml=None,camera_rotation=None,camera_translation=None):
    if camera_rotation is None:
        camera_rotation=np.eye(3)
    if camera_translation is None:
        camera_translation=np.zeros((3,))
    pipeline=rs.pipeline()
    profile=pipeline.start()
    depth_scale=profile.get_device().first_depth_sensor().get_depth_scale()
    if decimation_factor>1:
        decimation=rs.decimation_filter()
        decimation.set_option(rs.option.filter_magnitude, 3)
        get_depth=lambda frames : decimation.process(frames.get_depth_frame())
    else:
        get_depth=lambda frames: frames.get_depth_frame()

    start=time.perf_counter()
    dimages=[]
    cimages=[]
    image_times=[]
    try:
        while time.perf_counter()-start<dur:
            frames=pipeline.wait_for_frames()
            time_domain=frames.get_frame_timestamp_domain()
            image_times.append(frames.timestamp)
            depthf=get_depth(frames)
            depth_profile=rs.video_stream_profile(depthf.profile)
            depth_intrinsics = depth_profile.get_intrinsics()
            colorf=frames.get_color_frame()
            color_profile=rs.video_stream_profile(colorf.profile)
            color_intrinsics=color_profile.get_intrinsics()
            if not depthf or not colorf:
                continue
            dimages.append(np.array(depthf.get_data())*depth_scale)
            cimages.append(np.array(colorf.get_data()))
            d2cextrinsics=depth_profile.get_extrinsics_to(color_profile)
            cv2.imshow("Color Stream",cimages[-1][:,:,::-1])
            cv2.waitKey(1)
    finally:
        pipeline.stop()
        print(len(dimages))

    datadict=dict()
    datadict["image_times"]=image_times
    datadict["timestamp_domain"]=time_domain.name
    datadict["color"]=np.array(cimages)
    datadict["depth_scaled"]=np.array(dimages)
    datadict["cintrinsics"]=picklable_intrinsics(color_intrinsics)
    datadict["dintrinsics"]=picklable_intrinsics(depth_intrinsics)
    datadict["d2cextrinsic"]={"rotation":np.array(d2cextrinsics.rotation).reshape((3,3)),"translation":np.array(d2cextrinsics.translation)}
    datadict["camera_rotation"]=camera_rotation
    datadict["camera_translation"]=camera_translation
    if folder is not None:
        try:
            with open(os.path.join(folder,filename),"wb") as fh:
                pickle.dump(datadict,fh)
        except FileNotFoundError:
            print(str(folder)+" not found; datadict NOT saved")
    if xml is not None:
        datadict["xml"]=xml
        try:
            shutil.copy(xml,folder)
        except Exception:
            print("Shutil was unable to copy "+str(xml)+"to "+str(folder))
    cv2.destroyAllWindows()
    return datadict

def get_camera_profile():
    pipeline=rs.pipeline()
    config=rs.config()
    wrapper=rs.pipeline_wrapper(pipeline)
    profile=config.resolve(wrapper)
    return profile

def get_depth_scale(profile=None):
    if profile is None:
        pipeline=rs.pipeline()
        config=rs.config()
        wrapper=rs.pipeline_wrapper(pipeline)
        profile=config.resolve(wrapper)
    return profile.get_device().first_depth_sensor().get_depth_scale()

def get_depth_intrinsics(profile=None):
    if profile is None:
        pipeline=rs.pipeline()
        config=rs.config()
        wrapper=rs.pipeline_wrapper(pipeline)
        profile=config.resolve(wrapper)
    streams=profile.get_streams()
    for stream in streams:
        if stream.stream_type()==rs.stream.depth:
            return stream.as_video_stream_profile().get_intrinsics()
    print("No depth stream found")

def get_color_intrinsics(profile=None):
    if profile is None:
        pipeline=rs.pipeline()
        config=rs.config()
        wrapper=rs.pipeline_wrapper(pipeline)
        profile=config.resolve(wrapper)
    streams=profile.get_streams()
    for stream in streams:
        if stream.stream_type()==rs.stream.color:
            return stream.as_video_stream_profile().get_intrinsics()
    print("No color stream found")    

def get_color_intrinsic_matrix(profile=None):
    intrinsics=get_color_intrinsics(profile)
    return intrinsics_to_matrix(intrinsics)

def intrinsics_to_matrix(intrinsics):
    return np.array([[intrinsics.fx,0,intrinsics.ppx],[0,intrinsics.fy,intrinsics.ppy],[0,0,1]])

def picklable_intrinsics(intrinsics):
    return Intrinsics(intrinsics.fx,intrinsics.fy,intrinsics.ppx,intrinsics.ppy,intrinsics.model.__repr__(),intrinsics.coeffs,intrinsics.height,intrinsics.width)

def pointcloud_from_depth_image(depth_image,intrinsic_mat):
    height,width=depth_image.shape
    fx=intrinsic_mat[0,0]
    fy=intrinsic_mat[1,1]
    cy = intrinsic_mat[1,2]
    cx = intrinsic_mat[0,2]
    o3dintrinsics=o3d.camera.PinholeCameraIntrinsic(width, height, fx, fy, cx, cy)
    o3d_cloud = o3d.geometry.PointCloud.create_from_depth_image(o3d.geometry.Image(depth_image.copy()), o3dintrinsics,depth_scale=1)
    return o3d_cloud

def o3d_rgbd(color_image,depth_image,cuda=True):
    cimage=o3d.t.geometry.Image(o3d.core.Tensor(np.asarray(color_image.get_data())))
    dimage=o3d.t.geometry.Image(o3d.core.Tensor(np.asarray(depth_image.get_data())))
    if cuda:
        cimage=cimage.cuda()
        dimage=dimage.cuda()
    rgbd=o3d.t.geometry.RGBDImage(cimage,dimage)
    return rgbd

def o3d_rgbd_from_color_and_depth_arrays(color_array,depth_array,cuda=True):
    cimage=o3d.t.geometry.Image(o3d.core.Tensor(color_array))
    dimage=o3d.t.geometry.Image(o3d.core.Tensor(depth_array))
    if cuda:
        cimage=cimage.cuda()
        dimage=dimage.cuda()
    rgbd=o3d.t.geometry.RGBDImage(cimage,dimage)
    return rgbd

class RealSensePointCloudGenerator():
    def __init__(self,background_depth,decimation_factor=1) -> None:
        self.background_depth=background_depth
        self.pipeline=rs.pipeline()
        self.align=rs.align(rs.stream.depth)
        #device=rs.config().resolve(rs.pipeline_wrapper(pipeline)).get_device()
        if decimation_factor>1:
            decimation=rs.decimation_filter()
            decimation.set_option(rs.option.filter_magnitude, decimation_factor)
            self.process=lambda frames : self.align.process(rs.composite_frame(decimation.process(frames)))
        else:
            self.process=self.align.process
        self.start_capture()
    def start_capture(self,start_record=False):
        self.pipeline_profile=self.pipeline.start()
        frames=self.pipeline.wait_for_frames()
        processed=self.process(frames)
        depthf=processed.get_depth_frame()
        colorf=processed.get_color_frame()
        self.depth_profile=depthf.profile.as_video_stream_profile()
        self.color_profile=colorf.profile.as_video_stream_profile()
        self.dintrinsic_mat=intrinsics_to_matrix(self.depth_profile.get_intrinsics())
        self.depth_scale=self.pipeline_profile.get_device().first_depth_sensor().get_depth_scale()
    def stop_capture(self):
        self.pipeline.stop()
    def resume_record(self):
        print("Not Implemented")
    def pause_record(self):
        pass
    def get_metadata(self):
        self.rgbd_metadata = o3d.t.io.RGBDVideoMetadata()
        self.rgbd_metadata.depth_format=self.depth_profile.format().name.upper()
        self.rgbd_metadata.color_format=self.color_profile.format().name.upper()
        self.rgbd_metadata.depth_scale=1/self.depth_scale
        self.rgbd_metadata.fps=self.depth_profile.fps()
        self.rgbd_metadata.height=self.depth_profile.height()
        self.rgbd_metadata.width=self.depth_profile.width()
        self.rgbd_metadata.intrinsics=o3d.camera.PinholeCameraIntrinsic(self.depth_profile.width(),self.depth_profile.height(),self.dintrinsic_mat)
        self.rgbd_metadata.serial_number=self.pipeline_profile.get_device().get_info(rs.camera_info.serial_number)
        return self.rgbd_metadata
    def capture_frame(self,wait=True,align_depth_to_color=True):
        '''
        ignores values of arguments and always waits and always aligns color to depth
        '''
        _,colorf,depthf=self.wait_for_rgbd()
        return o3d_rgbd(colorf,depthf,True)
    def wait_for_observation(self):
        frames=self.pipeline.wait_for_frames()
        timestamp=time.perf_counter()
        depthf=self.process(frames).get_depth_frame()
        return timestamp,self.pointset_from_depth(depthf)
    def wait_for_rgbd(self):
        frames=self.pipeline.wait_for_frames()
        timestamp=time.perf_counter()
        aligned_frames=self.process(frames)
        depthf=aligned_frames.get_depth_frame()
        colorf=aligned_frames.get_color_frame()
        return timestamp, colorf, depthf
    def depth_array_from_depth_frame(self,depthf):
        return (np.asanyarray(depthf.get_data())*self.depth_scale).astype(np.float32)
    def pointset_from_depth(self,depthf):
        return pointset_from_image(self.depth_array_from_depth_frame(depthf),self.dintrinsic_mat,self.background_depth,dtype=np.float32)
    def pointset_from_depth_array(self,depth_array):
        return pointset_from_image(depth_array,self.dintrinsic_mat,self.background_depth,dtype=np.float32)
    def o3dpcd_from_color_and_depth_arrays(self,color_array,depth_array,cuda=True):
        #assumes depth_array already had depth_scale applied
        rgbd=o3d_rgbd_from_color_and_depth_arrays(color_array,depth_array,cuda)
        return o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.core.Tensor(self.dintrinsic_mat),depth_scale=1,depth_max=self.background_depth)
    def o3dpcd_from_rgb_and_depth(self,color_image,depth_image):
        rgbd=o3d_rgbd(color_image,depth_image,True)
        return self.o3dpcd_from_o3drgbd(rgbd)
    def o3dpcd_from_o3drgbd(self,rgbd):
        return o3d.t.geometry.PointCloud.create_from_rgbd_image(rgbd, o3d.core.Tensor(self.dintrinsic_mat),depth_scale=1/self.depth_scale,depth_max=self.background_depth)
