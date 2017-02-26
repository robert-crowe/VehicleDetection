
env = {
    ### TODO: Tweak these parameters and see how the results change.
    'color_space': 'YUV', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    'orient': 18,  # HOG orientations
    'pix_per_cell': 16, # HOG pixels per cell
    'cell_per_block': 3, # HOG cells per block
    'hog_channel': 'ALL', # Can be 0, 1, 2, or "ALL"
    'spatial_size': (16,16), # Spatial binning dimensions
    'hist_bins': 32,    # Number of histogram bins
    'spatial_feat': True, # Spatial features on or off
    'hist_feat': True, # Histogram features on or off
    'hog_feat': True, # HOG features on or off
    'x_start_stop': [None, None], # Min and max in y to search in slide_window()
    'y_start_stop': [400, 660], # Min and max in y to search in slide_window()
}
