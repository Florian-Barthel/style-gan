import glob
from PIL import Image


def create_video(folder, size=(400, 400), duration=200):
    fp_in = folder + '/*.png'
    fp_out = folder + '/animation.gif'

    img, *imgs = [Image.open(f) for f in sorted(glob.glob(fp_in))]
    img.thumbnail(size)
    for image in imgs:
        image.thumbnail(size)
    img.save(fp=fp_out, format='GIF', append_images=imgs,
             save_all=True, duration=duration, loop=0)


create_video('./../runs/2020-05-24_02-22-22')
