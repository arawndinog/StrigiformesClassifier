from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os

f_dir = '/home/adrian/Projects/owl_pics/barn_owl/'
out_dir = ''
f_list = os.listdir(f_dir)
owl_npy = np.array([])
owl_h5f = h5py.File('owl_dataset.h5', 'w')
resized_dim = (128,128)

for f in f_list:
    img = Image.open(f_dir + f)
    if os.path.splitext(f)[1] not in ['.jpg','.jpeg']:      #fix this
        continue

    result_img = img.resize(resized_dim,Image.BILINEAR)
    #result_img.save(out_dir + os.path.splitext(f)[0] + ".png")
    img_data = np.asarray(result_img, dtype=np.uint8)
    assert img_data.shape == (resized_dim) + (3,), img_data
    img_data = np.expand_dims(img_data, axis=0)
    if not owl_npy.any():
        owl_npy = img_data
    else:
        owl_npy = np.append(owl_npy,img_data,axis=0)
        
print owl_npy.shape
owl_h5f.create_dataset('data', data=owl_npy)
owl_h5f.close()