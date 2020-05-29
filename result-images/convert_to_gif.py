import glob
from PIL import Image
import os


def create_video(size=(400, 400), duration=200):
    last_entry = os.listdir('./../runs')[-1]
    folder = os.path.join('./../runs', str(last_entry), 'images')
    fp_in = folder + '/*.png'
    fp_out = folder + '/' + str(last_entry) + '_animation.gif'

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.thumbnail(size)
    for image in imgs:
        image.thumbnail(size)
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=duration, loop=0)


create_video()
