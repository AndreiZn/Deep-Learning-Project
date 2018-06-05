import sys
import os
import glob
from os import listdir
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from time import time
import json
from pprint import pprint
import shutil

#!/usr/bin/env python
"""
Pack multiple images of different sizes into one image.

Based on S W's recipe:
http://code.activestate.com/recipes/442299/
Licensed under the PSF License
"""
import argparse
import glob
from PIL import ImageChops

try: import timing # Optional, http://stackoverflow.com/a/1557906/724176
except: None

class PackNode(object):
    """
    Creates an area which can recursively pack other areas of smaller sizes into itself.
    """
    def __init__(self, area):
        #if tuple contains two elements, assume they are width and height, and origin is (0,0)
        if len(area) == 2:
            area = (0,0,area[0],area[1])
        self.area = area

    def __repr__(self):
        return "<%s %s>" % (self.__class__.__name__, str(self.area))

    def get_width(self):
        return self.area[2] - self.area[0]
    width = property(fget=get_width)

    def get_height(self):
        return self.area[3] - self.area[1]
    height = property(fget=get_height)

    def insert(self, area):
        if hasattr(self, 'child'):
            a = self.child[0].insert(area)
            if a is None: 
                return self.child[1].insert(area)
            return a

        area = PackNode(area)
        if area.width <= self.width and area.height <= self.height:
            self.child = [None,None]
            self.child[0] = PackNode((self.area[0]+area.width, self.area[1], self.area[2], self.area[1] + area.height))
            self.child[1] = PackNode((self.area[0], self.area[1]+area.height, self.area[2], self.area[3]))
            return PackNode((self.area[0], self.area[1], self.area[0]+area.width, self.area[1]+area.height))
        
def get_cropped_frames(image_array, bboxes):
    cropped_frames = []
    info_array = []
    for index, im in enumerate(image_array):
        bbs_im = bboxes[index]
        for bb in bbs_im:
            cropped_frames.append(image_array[index].crop(bb))
            info_array.append([index, bb])
    return cropped_frames, info_array

def pack_cropped_frames(cropped_frames, info_array):
    
    packed_images = []
    size = 720, 480
    format = 'RGB'
    sort = True
    
    # sort cropped_frames:
    if sort:
        cropped_frames = [(i, cropped_frames[i]) for i in range(len(cropped_frames))]
        cropped_frames = sorted(cropped_frames, key = lambda x: x[1].size[1], reverse=True)
        # get new_indices and sorted array of cropped_frames
        new_ind = [cropped_frames[i][0] for i in range(len(cropped_frames))]
        cropped_frames = [cropped_frames[i][1] for i in range(len(cropped_frames))]
        # modify info_array:
        info_array = [info_array[i] for i in new_ind]
    
    print ("Create tree")
    tree = PackNode(size)
    image = Image.new(format, size)
    
    #insert each image into the PackNode area
    for i, img in enumerate(cropped_frames):
        #print (img.size)
        uv = tree.insert(img.size)
        if uv is None: 
            #print('#', i, ', with size', img.size, "can't be packed; new packed_image will be created")
            image = image.crop(box=image.getbbox())
            packed_images.append(image)
            #print ("Create tree")
            tree = PackNode(size)
            image = Image.new(format, size)
            #print (img.size)
            uv = tree.insert(img.size)
        image.paste(img, uv.area)
        info_array[i].append(len(packed_images)) # ID of a packed_image
        info_array[i].append(uv.area) # position of cropped frame in a packed_image
    
    image = image.crop(box=image.getbbox())
    packed_images.append(image) 
    
    return packed_images, info_array


def openpose_image(image, image_name, json_output='./knapsack/json_output/', output_directory=None, net_resolution='"-1x368"'):
    
    pn_w, pn_h = -1, 384
    w, h = image.size[0], image.size[1]
    nr0, nr1 = int((pn_w*(w/720)//16+1)*16), int((pn_h*(h/480)//16+1)*16)
    net_resolution = str(nr0) + "x" + str(nr1)
    
    temp_directory='./knapsack/openpose_1_image/'
    image.save(temp_directory + image_name)
    
    if output_directory is not None:
        cmd = './build/examples/openpose/openpose.bin --no_display --net_resolution %s --image_dir ./knapsack/openpose_1_image/ --write_images %s --write_json %s'%(net_resolution, output_directory, json_output)
        os.system(cmd)
    else:
        cmd = './build/examples/openpose/openpose.bin --no_display --net_resolution %s --image_dir ./knapsack/openpose_1_image/ --write_json %s'%(net_resolution, json_output)
        os.system(cmd)
    
    os.remove(temp_directory + image_name)
    
# apply openpose to frames with specified bounding boxes
def apply_openpose_to_frames_with_bbs(frames, bbs, net_resolution):
    # crop frames from large frames according to bounding boxes
    cropped_frames, info_array = get_cropped_frames(image_array=frames, bboxes=bbs)
    # pack cropped frames into images "packed_images"
    packed_images, info_array = pack_cropped_frames(cropped_frames, info_array)
    # remove files from directories:
    shutil.rmtree('./knapsack/rendered_images/')
    os.makedirs('./knapsack/rendered_images/')
    shutil.rmtree('./knapsack/json_output/')
    os.makedirs('./knapsack/json_output/')
    # apply openpose to packed_images
    for index, p_im in enumerate(packed_images):
        img_name = (7-len(str(index)))*'0'+str(index)+'.png'
        openpose_image(p_im, img_name, output_directory='./knapsack/rendered_images/', net_resolution=net_resolution)
        json_f_name = './knapsack/json_output/'+ (7-len(str(index)))*'0'+str(index) + '_keypoints.json'
        with open(json_f_name) as f:
            data = json.load(f)
    return cropped_frames, packed_images, info_array

def get_openpose_coords(frames, bbs, json_output_folder='knapsack/json_output/', net_resolution='"384x160"'):
    
    cropped_frames, packed_images, info_array = apply_openpose_to_frames_with_bbs(frames, bbs, net_resolution)
    for num_p_im, f_name in enumerate(sorted(listdir(json_output_folder))):
        with open(json_output_folder+f_name) as f:
            data = json.load(f)
        for person in data['people']:
            for label in person:
                joint_coord = person[label]
                if label == 'pose_keypoints' and joint_coord!=[]:
                    
                    rng = range(len(joint_coord)//3)
                    x_coord = [joint_coord[i*3] for i in rng if joint_coord[i*3+2]>0.6]
                    if x_coord == []: break
                    x_av = np.average(x_coord) 
                    y_coord = [joint_coord[i*3+1] for i in rng if joint_coord[i*3+2]>0.6]
                    y_av = np.average(y_coord)
                    
                    # coord of cropped frames in a packed_image
                    crf_coord_in_pim = [(i,info_array[i][3]) for i in range(len(info_array)) if info_array[i][2]==num_p_im]
                    frame_id = crf_coord_in_pim[0][0]
                    for i, c in crf_coord_in_pim:
                        if x_av >= c[0] and x_av <= c[2] and y_av >= c[1] and y_av <= c[3]:
                            frame_id = i
                            break
                    #print(frame_id, info_array[frame_id])
                    info_array[frame_id].append(joint_coord)
    return info_array

def output_openpose_coords(frames, bbs, json_output_folder='knapsack/json_output/', net_resolution='"720x480"'):
    
    info_array = get_openpose_coords(frames=frames, bbs=bbs, net_resolution=net_resolution)
    num_of_frames = len(frames)
    output = [[] for _ in range(num_of_frames)]
    for ind in range(len(info_array)):
        cr_fr = info_array[ind]
        im_num = cr_fr[0]
        x_bb_rel_to_im, y_bb_rel_to_im = cr_fr[1][0], cr_fr[1][1]
        x_fr_rel_to_pim, y_fr_rel_to_pim = cr_fr[3][0], cr_fr[3][1]
        for j in range(4, len(cr_fr)):
            array = list(cr_fr[j])
            #print(x_bb_rel_to_im, y_bb_rel_to_im)
            #print(x_fr_rel_to_pim, y_fr_rel_to_pim)
            for i in range(0,len(array), 3):
                array[i] = array[i]+x_bb_rel_to_im-x_fr_rel_to_pim
            for i in range(1,len(array), 3):
                array[i] = array[i]+y_bb_rel_to_im-y_fr_rel_to_pim
            output[im_num].append(array)
            
    return output

def depict_points(frames, positions):
    for ind, frame in enumerate(frames):
        for bb in positions[ind]:
            plt.imshow(frame)
            #print(bb)
            ind = [j for j in range(0,len(bb), 3) if bb[j]>0 and bb[j+1]>0]
            x = [bb[j] for j in ind]
            y = [bb[j+1] for j in ind]
            plt.scatter(x,y,s=10, c='r')
            plt.show()     
            

# main:

os.makedirs('./knapsack/rendered_images/')
os.makedirs('./knapsack/json_output/')
os.makedirs('./knapsack/openpose_1_image/')

# generate an image array
img_array = []
img_folder = './knapsack/Initial_frames/'
for filename in sorted(listdir(img_folder)):
    img_array.append(Image.open(img_folder + filename))

# generate bounding boxes
np.random.seed(8)
num_bb_per_frame = 2
num_of_images = len(img_array)
num_cropped_frames = num_bb_per_frame * num_of_images
w_0, h_0 = 200, 80
w_1, h_1 = 370, 410
bbs = [[() for _ in range(num_bb_per_frame)] for _ in range(num_of_images)]
for i in range(num_cropped_frames):
    d1, d2 = int(10*(np.random.rand()-0.2)), int(10*(np.random.rand()-0.8))
    bbs[i//num_bb_per_frame][i%num_bb_per_frame]=(int(w_0-d1*1.5*i//3), int(h_0+d2*1.5*i//3), w_1-10*d1, h_1-10*d2)
    

t = time()
output = output_openpose_coords(img_array, bbs)
print(time()-t, 's')

#print(output)
#depict_points(img_array, positions=output)


