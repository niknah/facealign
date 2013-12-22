#!/usr/bin/python
 
# sizeToFace.py
 
# Face Detection using OpenCV
 
# Usage: python sizeToFace.py <image_directory> optional: <output_directory> <start_num>,<end_num>
# ffmpeg -r 15 -b 1800 -i %4d.JPG -i testSong.mp3 test1800.avi
 
import FaceImage
from multiprocessing import Pool
import sys, os
from operator import itemgetter

def main():
    # Print usage if no args specified
    args = sys.argv[1:]
    if len(args) == 0:
        print('Usage: python sizeToFace.py <image_directory> optional: <output_directory> <start_num>,<end_num>')
        return

    # Get input files, sort by last modified time
    files = sortedImages(args[0])

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
    i=start
    pool = Pool()

    # For every JPG in the given directory
    for file in files:
        filepath = file[1]
        i += 1
        savename = os.path.join(outdir, '%04d.jpg' % i)

        print('Added to pool: ' + filepath + ' with output path: ' + savename)
        pool.apply_async(FaceImage.runFaceImage, (filepath, savename))

    pool.close()
    pool.join()

def sortedImages(inputDir):
    files = []
    for dirpath, dirnames, filenames in os.walk(inputDir):
        for file in filenames:
            if file.upper().endswith('.JPG') or file.upper().endswith('.JPEG'):
                # If a jpeg file
                filePath = os.path.join(dirpath, file)
                files.append((os.stat(filePath).st_mtime, filePath))

    # Sort by last modified, then by path
    # (some old pics in my set have an equal recent modified time)
    files.sort(key=itemgetter(0,1))
    return files

if __name__ == "__main__":
    main()

