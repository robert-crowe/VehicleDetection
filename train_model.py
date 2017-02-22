import numpy as np
import cv2
import glob
import random
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
from features import *

start=time.time()

# Read in cars and notcars
car_files = glob.glob('.\\vehicles\\vehicles\\*\\*.png')
notcar_files = glob.glob('.\\non-vehicles\\non-vehicles\\*\\*.png')

num_cars = len(car_files)
num_notcars = len(notcar_files)
print('Got {} car image filenames and {} notcar image filenames: {:.1f}s'.format(num_cars, num_notcars, time.time()-start))

random.shuffle(car_files)
cars = [car for car in car_files]
random.shuffle(notcar_files)
notcars = [notcar for notcar in notcar_files]

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

print('Begin car features: {:.1f}s'.format(time.time()-start))

car_features = extract_features(cars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

print('Finished car features, begin notcar features: {:.1f}s'.format(time.time()-start))

notcar_features = extract_features(notcars, color_space=color_space, 
                        spatial_size=spatial_size, hist_bins=hist_bins, 
                        orient=orient, pix_per_cell=pix_per_cell, 
                        cell_per_block=cell_per_block, 
                        hog_channel=hog_channel, spatial_feat=spatial_feat, 
                        hist_feat=hist_feat, hog_feat=hog_feat)

print('Notcar features complete: {:.1f}s'.format(time.time()-start))

X = np.vstack((car_features, notcar_features)).astype(np.float64)                        

X_scaler = StandardScaler().fit(X) # Fit a per-column scaler

scaled_X = X_scaler.transform(X) # Apply the scaler to X
print('Stacked and standardized: {:.1f}s'.format(time.time()-start))

# Define the labels vector
y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

# Split up data into randomized training and test sets
rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)
print('Split training and test: {:.1f}s'.format(time.time()-start))

print('Using: {} orientations, {} pixels per cell and {} cells per block'.format(orient, pix_per_cell, cell_per_block))
print('Feature vector length: {}'.format(len(X_train[0])))

svc = LinearSVC() # Use a linear SVC 
print('SVC created, begin training: {:.1f}s'.format(time.time()-start))

svc.fit(X_train, y_train)
print('Training complete, begin saving: {:.1f}s'.format(time.time()-start))

joblib.dump(svc, 'SVCmodel.pkl') # Save model
print('Saved, begin reload: {:.1f}s'.format(time.time()-start))

svc2 = joblib.load('SVCmodel.pkl') # Reload model to confirm

# Check the score of the SVC2
print('Test Accuracy of SVC2 = {}'.format(round(svc2.score(X_test, y_test), 4)))
print('Done: {:.1f}s'.format(time.time()-start))

