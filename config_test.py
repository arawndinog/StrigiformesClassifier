import datetime
import os
current_time = datetime.datetime.now().time()

project_dir = os.path.expanduser('~') + "/Projects/"
test_dataset_dir = project_dir + "Dataset/"
ckpt_dir = project_dir + "Checkpoints/"

test_files = [""]

test_batch_size = 3000