import argparse
import os
import shutil

def rename_file(dir_path_src : str, dir_path_dst : str) -> None:
    """
    Args:
        -**dir_path_src**: Source directory.
        -**dir_path_dst**: Destination directory (must exist?)
    """
    dir_list = os.listdir(dir_path_src)
    print(f'dir_list:{dir_list}')

    for i in range(len(dir_list)):

        if ".txt" == os.path.splitext(dir_list[i])[1]:
            # expected data_slamr_number
            filename = os.path.splitext(dir_list[i])[0]
            # split on _ and get the third element (the number)
            num = int(filename.split('_')[2])
            s = str(num).zfill(10)
            new_file_name = "data_slamr_{}.txt".format(s)

            # Copy file to new name instead of renaming
            src = os.path.join(dir_path_src, dir_list[i])
            dst = os.path.join(dir_path_dst, new_file_name)
            # Save the file at the destination path
            print(f'src:{src}')
            print(f'dst:{dst}')
            shutil.copy(src, dst)

# example
# python rename/rename_filename.py --path_in=rename/data/ --path_out=rename/dataout/
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="training and testing script")
    parser.add_argument("--path_in", default='runs/detect/predict4/labels/', help="Path with list of files to rename.")
    parser.add_argument("--path_out", default='rename/data/', help="Path of the destination folder.")

    args = parser.parse_args()
    dir_path_src = args.path_in
    dir_path_dst = args.path_out
    rename_file(dir_path_src=dir_path_src, dir_path_dst=dir_path_dst)
