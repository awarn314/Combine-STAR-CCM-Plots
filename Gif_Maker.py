# -*- coding: utf-8 -*-
"""
Created on Fri May 24 16:50:37 2019

@author: Alex
"""

import os
import shutil
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import PIL
import imageio

path = os.getcwd()  

#Inputs###########################################
Folders=['Coef','Pressure','Velocity','Vorticity']
crop_tuples=[(0,0,0,0), (0,300,2254,1000),(0,300,2254,1000),(0,300,2254,1000)]
Crop_Images=0
Clean_img_Name=0
Make_gif_imgs=0
Make_Gif=1
##################################################


Folders_Cleaned=[]
FNames_Cleaned=[]

for fol in Folders:
    fol_n=fol+'_cleaned'
    Folders_Cleaned.append(fol_n)
    if not os.path.exists(fol_n):
        os.mkdir(fol_n)
 
if not os.path.exists('Gif_Stack'):
    os.mkdir('Gif_Stack')
png_dir = path+'/Gif_Stack/'    
gif_fname='Gif_'

for fol, fol2,ct in zip(Folders,Folders_Cleaned,crop_tuples):
    files_in_fold=os.listdir(fol)
    fname_str_l=files_in_fold[0].split('_')
    fname_str_l=fname_str_l[0:-1]
    fname_str=''.join(v+'_' for v in fname_str_l)
    FNames_Cleaned.append(fname_str)
       
if Clean_img_Name==1:
    for fol, fol2,ct in zip(Folders,Folders_Cleaned,crop_tuples):
        files_in_fold=os.listdir(fol)
        fname_str_l=files_in_fold[0].split('_')
        fname_str_l=fname_str_l[0:-1]
        fname_str=''.join(v+'_' for v in fname_str_l)
        for ff in files_in_fold:
            time_str=ff.split('_')[-1]
            time_sci=time_str.split('.png')[0]
            time="{:09.4F}".format(float(time_sci))
            fname=fol2+'/'+fname_str+time+'.png'
            image_obj = Image.open(fol+'/'+ff)
            if Crop_Images==1:
                if sum(ct)>0:
                    cropped_image = image_obj.crop(ct)
                else:
                    cropped_image = image_obj
                cropped_image.save(fname)
            else:
                cropped_image = image_obj
                cropped_image.save(fname)
            
def concat_images(imga, imgb):
    """
    Combines two color image ndarrays side-by-side.
    """
    ha,wa = imga.shape[:2]
    hb,wb = imgb.shape[:2]
    max_height = np.max([ha, hb])
    total_width = wa+wb
    new_img = np.zeros(shape=(max_height, total_width, 3))
    new_img[:ha,:wa]=imga
    new_img[:hb,wa:wa+wb]=imgb
    return new_img

def concat_n_images(image_path_list):
    """
    Combines N color images from a list of image paths.
    """
    output = None
    for i, img_path in enumerate(image_path_list):
        img = plt.imread(img_path)[:,:,:3]
        if i==0:
            output = img
        else:
            output = concat_images(output, img)
    return output

if Make_gif_imgs==1:
    for ff in os.listdir(Folders_Cleaned[0]):
        pre_ind=ff.split(FNames_Cleaned[0])
        ind=pre_ind[1].split('.png')[0]
        imgs=[]
        for fna, fold in zip(FNames_Cleaned,Folders_Cleaned):
           imgs.append(fold + '/' + fna+ind+'.png')         
        imgs1    = [ PIL.Image.open(i) for i in imgs ]
        min_shape = sorted( [(np.sum(i.size), i.size ) for i in imgs1])[0][1]
        imgs_comb = np.vstack( (np.asarray( i.resize(min_shape) ) for i in imgs1 ) )
        imgs_comb = PIL.Image.fromarray( imgs_comb)
        imgs_comb.save('Gif_Stack' + '/' + gif_fname + ind + '.png')
        
if Make_Gif==1:    
    images = []
    for file_name in os.listdir(png_dir):
        if file_name.endswith('.png'):
            file_path = os.path.join(png_dir, file_name)
            images.append(imageio.imread(file_path))
    imageio.mimsave('Outout.gif', images)