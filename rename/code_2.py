import os

dir_path = 'runs/detect/predict3/labels/' 

dir_list = os.listdir(dir_path)

num = 1

for i in range(len(dir_list)):

if ".txt" == os.path.splitext(dir_list[i])[1]: 

s = str(num).zfill(10)

new_file_name = "data_slamr_{}.txt".format(s)

os.rename(os.path.join(dir_path, dir_list[i]), os.path.join(dir_path, new_file_name))

num += 1
