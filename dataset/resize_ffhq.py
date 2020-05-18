from tqdm import tqdm
import os
from PIL import Image

src_folder = 'E:/ffhq_merge'
dest_folder = 'E:/ffhq_256'

if not os.path.exists(dest_folder):
    os.makedirs(dest_folder)

num_kept = 0

for file in tqdm(os.listdir(src_folder)):
    src_img = src_folder + '/' + file
    dest_img = dest_folder + '/' + file
    img = Image.open(src_img)
    img.thumbnail((256, 256))
    img.save(dest_img, "PNG")
    num_kept += 1


print('Summary:')
print('kept: ' + str(num_kept))
