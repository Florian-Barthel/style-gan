from tqdm import tqdm
import shutil
import os
from PIL import Image
import warnings

src_folder = 'E:/carswithcolors/only images'
dest_folder = 'E:/carswithcolors/ratio'

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

num_kept = 0
num_removed = 0
num_corrupt_EXIF = 0

for file in tqdm(os.listdir(src_folder)):
    src_img = src_folder + '/' + file
    dest_img = dest_folder + '/' + file
    with warnings.catch_warnings() as my_warning:
        warnings.simplefilter('error', UserWarning)
        try:
            img = Image.open(src_img)
            w, h = img.size
            if w < h:
                print('removed invalid ratio')
                num_removed += 1
                continue
            shutil.copyfile(src_img, dest_img)
            num_kept += 1
        except:
            print('removed invalid format')
            num_corrupt_EXIF += 1

print('Summary:')
print('removed corrupt_exif: ' + str(num_corrupt_EXIF))
print('removed: ' + str(num_removed))
print('kept: ' + str(num_kept))
