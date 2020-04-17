import imageio
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt
import cv2


filename = 'video.mkv'
vid = imageio.get_reader(filename,  'ffmpeg')

dataset = []
distances = []


metadata = vid.get_meta_data()
w, h = metadata['source_size']
prev_im = np.zeros((h,w,3))

last_saved = 0
i = 0
for image in vid.iter_data():
    i+=1
    if i%30 != 1: continue
    
    print(i)

    # vector = np.reshape(image, (w*h*3,)).squeeze()    
    d = np.abs(np.sum(image - prev_im)/(image.mean()*w*h*3) + 1e-10)

    print(d)
    # if d > np.median(distances):
    delta = i - last_saved

    included = True if d > 1.4 * np.median(distances) else False

    if included:
        cv2.imwrite(f'included/frame{i:04d}_ditance={d:.2f}_median={np.median(distances):.2f}_included={included}.jpg', image)
        last_saved = i
    else:
        cv2.imwrite(f'non_included/frame{i:04d}_ditance={d:.2f}_median={np.median(distances):.2f}_included={included}.jpg', image)


    prev_im = image

    distances.append(d)
    # dist = metrics.euclidean_distances(vector, prev_im)

    # print(dist)