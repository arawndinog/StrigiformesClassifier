from PIL import Image
from sys import stdout
import numpy as np
import h5py
import os

folder_dir = '/home/adrian/Projects/Owls/'
folder_list = os.listdir(folder_dir)
owl_npy = []
owl_npy_label = []
owl_h5f = h5py.File('owl_dataset.h5', 'w')
resized_dim = (128,128)

owl_dict = {}
index_csv = open("/home/adrian/Projects/StrigiformesClassifier/reference/owl_index.csv")
while True:
    csv_line = index_csv.readline()
    if not csv_line:
        break
    csv_line_list = csv_line.strip().split(",")
    owl_index = int(csv_line_list[0])
    owl_name = csv_line_list[1]
    owl_dict.update({owl_name:owl_index})       #owl_name --> owl_index
index_csv.close()

for folder in folder_list:
    if os.path.isdir(folder_dir + folder):
        current_owl_index = owl_dict[folder]
        owl_folder = folder_dir + folder + "/"
        file_list = os.listdir(owl_folder)
        for i in xrange(len(file_list)):
            img = Image.open(owl_folder + file_list[i])
            if os.path.splitext(file_list[i])[1] not in ['.jpg','.jpeg']:       #fix this
                continue
            result_img = img.resize(resized_dim,Image.BILINEAR)
            #result_img.save(out_dir + os.path.splitext(f)[0] + ".png")
            img_data = np.asarray(result_img, dtype=np.uint8)
            if img_data.shape != (resized_dim) + (3,):                          #fix this
                continue
            #assert img_data.shape == (resized_dim) + (3,), img_data.shape
            owl_npy.append(img_data)
            owl_npy_label.append(current_owl_index)
            stdout.write("\r%d/%d" % (i,len(file_list)))
            stdout.flush()
        stdout.write("\n")
        
owl_npy = np.array(owl_npy)
owl_npy_label = np.array(owl_npy_label)
owl_h5f.create_dataset('data', data=owl_npy)
owl_h5f.create_dataset('label', data=owl_npy_label)
owl_h5f.close()

print "data shape: " + str(owl_npy.shape)
print "label shape: " + str(owl_npy_label.shape)