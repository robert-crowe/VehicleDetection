
# Read in an image

# Convert to RGB and normalize

# Features Module: Used for both training and pipeline
#     Create spatial features
#     Create histogram features
#     Create HOG features
#     Assemble feature vectors

# Train Classifier - Done once, then used in pipeline

# Pipeline
#     Read and mask out the parts where there will be no vehicles
#     Slide window (64x64 to match training)
#     Assemble heat map
#     Compare consecutive frames and eliminate false positives
#     Draw onto video and save

import numpy as np
import cv2
from sklearn.externals import joblib
from scipy.ndimage.measurements import label

from features import *
from windows import *
from heat import *
from config import env

debug = False # In debug we use the test images instead of the video
videodebug = False

def detect_vehicles(img):
    global clf
    global X_scaler
    global env
    global windows

    color_space = env['color_space'] # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = env['orient']  # HOG orientations
    pix_per_cell = env['pix_per_cell'] # HOG pixels per cell
    cell_per_block = env['cell_per_block'] # HOG cells per block
    hog_channel = env['hog_channel'] # Can be 0, 1, 2, or "ALL"
    spatial_size = env['spatial_size'] # Spatial binning dimensions
    hist_bins = env['hist_bins']    # Number of histogram bins
    spatial_feat = env['spatial_feat'] # Spatial features on or off
    hist_feat = env['hist_feat'] # Histogram features on or off
    hog_feat = env['hog_feat'] # HOG features on or off

    heat_memory = 25
    heat_thresh = 50

    hot_windows = search_windows(img, windows, clf, X_scaler, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)                       
        
    # Add heat to each box in box list
    heatmap = np.zeros_like(img[:,:,0]).astype(np.float)
    heatmap = add_heat(heatmap, hot_windows)

    # Take temperature
    if not debug:
        heatmap = temperature(recent_heatmaps, heatmap, memory=heat_memory)
        
    # Apply threshold to help remove false positives
    heatmap = apply_threshold(heatmap, heat_thresh)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = np.copy(img)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    if videodebug:
        draw_img = draw_boxes(draw_img, hot_windows, color=(255, 0, 0), thick=1)

    return draw_img                    



# Calling
mod_scale = joblib.load('SVCmodel.pkl') # Load trained model and feature scaler
clf = mod_scale['model']
X_scaler = mod_scale['scaler']
recent_heatmaps = []

# Create the sliding windows.  Do this once.
windows = []
x_start_stop = [None, None] # Min and max in y to search in slide_window()
img_shape = (720,1280,3)
window_groups = [
    # {'xy_window':(150, 150), 'y_start_stop':[550, 700], 'xy_overlap':(0.7, 0.0), 'x_start_stop':[None, None]},
    {'xy_window':(64, 64),   'y_start_stop':[400, 528], 'xy_overlap':(0.5, 0.5), 'x_start_stop':[500, img_shape[1]-300]},
    {'xy_window':(100, 100), 'y_start_stop':[400, 600], 'xy_overlap':(0.7, 0.7), 'x_start_stop':[200, img_shape[1]]},
    # {'xy_window':(100, 100), 'y_start_stop':[500, 600], 'xy_overlap':(0.7, 0.7), 'x_start_stop':[None,None]},
]

for win in window_groups:
    windows = windows + slide_window(img_shape, x_start_stop=x_start_stop,
        y_start_stop=win['y_start_stop'], xy_window=win['xy_window'], xy_overlap=win['xy_overlap'])

print('Start')
if debug:
    for i in range(1, 7):
        fname = 'test{}.jpg'.format(i)
        img = cv2.imread('test_images/{}'.format(fname))
        result = detect_vehicles(img)
        cv2.imshow(fname, result)
        print('Finished {}'.format(fname))
    print('Waiting')
    cv2.waitKey(0)
else:
    print('Processing video')
    from moviepy.editor import VideoFileClip
    clip1 = VideoFileClip("project_video.mp4")
    result_clip = clip1.fl_image(detect_vehicles)
    result_clip.write_videofile('project_video_result.mp4', audio=False)
    print('Video done')
