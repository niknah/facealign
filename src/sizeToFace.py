#!/usr/bin/python
 
# sizeToFace.py
 
# Face Detection using OpenCV
 
# Usage: python sizeToFace.py <image_directory> optional: <output_directory> <start_num>,<end_num>
# ffmpeg -r 15 -b 1800 -i %4d.JPG -i testSong.mp3 test1800.avi

import sys
import os
import FaceImage
from multiprocessing import Pool
from operator import itemgetter
from PIL import Image


def main():
    # Print usage if no args specified
    args = sys.argv[1:]
    if len(args) == 0:
        print('Usage: python sizeToFace.py <image_directory> optional: <output_directory> <start_num>,<end_num>')
        return

    # Get input files, sort by last modified time
    files = sorted_images(args[0])

    if len(files) == 0:
        print('No jpg files found in ' + args[0])
        return

    if len(sys.argv) > 1:
        outdir = args[1]
    else:
        outdir = '.'

    start, end = 0, len(files)-1
    if len(args) > 2:
        if ',' in args[2]:
            start, end = map(lambda x: int(x)-1, args[2].split(','))
        else:
            start = int(args[2])-1
   
    files = files[start:end+1]
    i = start
    pool = Pool()

    # For every JPG in the given directory
    results = {}
    for f in files:
        i += 1
        savename = os.path.join(outdir, '%04d.jpg' % i)

        print('Added to pool: ' + f[1] + ' with output path: ' + savename)
        results[f[1]] = pool.apply_async(FaceImage.runFaceImage, (f[1], savename))

    pool.close()
    pool.join()

    # Check the results :
    frames_validity = [res[1].get() for res in results.items()]
    print(frames_validity)


def sorted_images(input_dir):
    files = []
    for dirpath, dirnames, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.upper().endswith('.JPG') or filename.upper().endswith('.JPEG'):
                file_path = os.path.join(dirpath, filename)
                files.append((get_image_date(file_path), file_path))

    # Sort by last modified, then by path
    files.sort(key=itemgetter(0, 1))
    return files


def get_image_date(file_path):
    """ This returns the date as a formatted string like yyyy:mm:dd hh:mm:ss. Which is good enough for sorting. """
    DateTimeOriginalKey = 36867
    return Image.open(file_path)._getexif()[DateTimeOriginalKey]

if __name__ == "__main__":
    main()

