from tqdm import tqdm
import shutil
import os

src_folder = 'E:/ffhq_unzip'
dest_folder = 'E:/ffhq_merge'

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

num_kept = 0

for folder_1 in tqdm(os.listdir(src_folder)):
    if '.' not in folder_1:
        for folder_2 in os.listdir(src_folder + '/' + folder_1):
            if '.' not in folder_2:
                for folder_3 in os.listdir(src_folder + '/' + folder_1 + '/' + folder_2):
                    if '.' not in folder_3:
                        for file in os.listdir(src_folder + '/' + folder_1 + '/' + folder_2 + '/' + folder_3):
                            if file.endswith(".png"):
                                src_img = src_folder + '/' + folder_1 + '/' + folder_2 + '/' + folder_3 + '/' + file
                                dest_img = dest_folder + '/' + file
                                shutil.copyfile(src_img, dest_img)
                                num_kept += 1
                            else:
                                print(file)

print('kept: ' + str(num_kept))
