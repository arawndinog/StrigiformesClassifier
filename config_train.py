import datetime
import os
current_time = datetime.datetime.now().time()
session_id = "session_" + str(current_time)

project_dir = os.path.expanduser('~') + "/Projects/"
train_dataset_dir = project_dir + "Dataset/"
val_dataset_dir = project_dir + "Dataset/"
ckpt_dir = project_dir + "Checkpoints/"
session_dir = project_dir + session_id + "/"
if not os.path.isdir(session_dir):
    os.mkdir(session_dir)

train_f = "/home/adrian/Projects/owl_pics/owl_dataset.h5"
valid_f = "/home/adrian/Projects/owl_pics/owl_dataset.h5"

ckpt_name = session_id

use_ckpt = False
batch_size = 20
learning_rate = 0.01
epoch = 1000000
