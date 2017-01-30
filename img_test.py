import matplotlib.pyplot as plt
from PIL import Image
import os

f_dir = ''
out_dir = ''
f_list = os.listdir(f_dir)

for f in f_list:
    img = Image.open(f_dir + f)
    result_img = img.resize((128,128),Image.BILINEAR)
    result_img.save(out_dir + os.path.splitext(f)[0] + ".png")