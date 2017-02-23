
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

debug = False

def detect_vehicles(img):
    global clf
    global X_scaler

    ### TODO: Tweak these parameters and see how the results change.
    color_space = 'HSV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 2 # Can be 0, 1, 2, or "ALL"
    spatial_size = (16, 16) # Spatial binning dimensions
    hist_bins = 16    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    x_start_stop = [None, None] # Min and max in y to search in slide_window()
    y_start_stop = [400, 660] # Min and max in y to search in slide_window()

    # Notes: xy_window=(128, 128), xy_overlap=(0.9, 0.9) worked well for close cars, misses middle distance
    # In debug, xy_window=(96. 96), xy_overlap=(0.9, 0.9) threshold 1 gets all cars

    windows = slide_window(img, x_start_stop=x_start_stop, y_start_stop=y_start_stop, 
                    xy_window=(96, 96), xy_overlap=(0.5, 0.5))

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
    heatmap = temperature(recent_heatmaps, heatmap, memory=50)
        
    # Apply threshold to help remove false positives
    heatmap = apply_threshold(heatmap, 50)

    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    draw_img = np.copy(img)
    draw_img = draw_labeled_bboxes(np.copy(img), labels)

    if debug:
        draw_img = draw_boxes(draw_img, hot_windows, color=(255, 0, 0), thick=2)

    return draw_img                    



# Calling
mod_scale = joblib.load('SVCmodel.pkl') # Load trained model and feature scaler
clf = mod_scale['model']
X_scaler = mod_scale['scaler']
recent_heatmaps = []

print('Start')
if debug:
    for i in range(1, 7):
        fname = 'test{}.jpg'.format(i)
        img = cv2.imread('test_images/{}'.format(fname))
        result = detect_vehicles(img, mod_scale['model'], mod_scale['scaler'])
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
