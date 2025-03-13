import os
import pickle
import time
from functools import wraps
import timeit
import shutil

def make_timestamped_folder(path_prefix,folder_leading_name=""):
    timestr=time.strftime("%Y%m%d_%H%M%S")
    outfolder=os.path.join(path_prefix,folder_leading_name+timestr)
    os.makedirs(outfolder,exist_ok=True)
    return outfolder
def write_README(outfolder,script_to_copy,**kwargs):
    script_name=os.path.basename(script_to_copy)
    shutil.copy(script_to_copy,os.path.join(outfolder,script_name))
    with open(os.path.join(outfolder,"README"),"w") as fh:
        for key,val in kwargs.items():
            fh.write(f"{key}: {val}\n")