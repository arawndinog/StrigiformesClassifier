import datetime
import os
current_time = datetime.datetime.now().time()
session_id = "session_" + str(current_time)

project_dir = os.path.expanduser('~') + "/Projects/CASIA/"
train_dataset_dir = project_dir + "Dataset/"
val_dataset_dir = project_dir + "Dataset/"
ckpt_dir = project_dir + "Checkpoints/"
session_dir = project_dir + session_id + "/"
if not os.path.isdir(session_dir):
    os.mkdir(session_dir)

train_f = [""]
valid_f = [""]

ckpt_name = session_id

use_ckpt = True
batch_size = 20
learning_rate_init = 0.01
learning_rate_decay_rate = 0.1
