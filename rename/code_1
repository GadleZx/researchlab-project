import os
import shutil

dir_path = 'runs/detect/predict4/labels/'

dir_list = os.listdir(dir_path)

num = 1

for i in range(len(dir_list)):

    if ".txt" == os.path.splitext(dir_list[i])[1]:
        s = str(num).zfill(10)
        new_file_name = "data_slamr_{}.txt".format(s)

        # Copy file to new name instead of renaming
        src = os.path.join(dir_path, dir_list[i])
        dst = os.path.join(dir_path, new_file_name)
        shutil.copy(src, dst)

        num += 1
