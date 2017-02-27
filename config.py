
env = {
    ### TODO: Tweak these parameters and see how the results change.
    'color_space': 'LUV', # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    'orient': 18,  # HOG orientations
    'pix_per_cell': 16, # HOG pixels per cell
    'cell_per_block': 3, # HOG cells per block
    'hog_channel': 0, # Can be 0, 1, 2, or "ALL"
    'spatial_size': (16,16), # Spatial binning dimensions
    'hist_bins': 32,    # Number of histogram bins
    'spatial_feat': True, # Spatial features on or off
    'hist_feat': True, # Histogram features on or off
    'hog_feat': True, # HOG features on or off
    'x_start_stop': [None, None], # Min and max in y to search in slide_window()
    'y_start_stop': [400, 660], # Min and max in y to search in slide_window()
    'window_groups': [
        {'xy_window':(200,200), 'x_start_stop':[960, 1280], 'y_start_stop':[385, 656], 'xy_overlap':(0.9, 0.9)},
        {'xy_window':(200,200), 'x_start_stop':[0, 1280-960], 'y_start_stop':[385, 656], 'xy_overlap':(0.5, 0.5)},
        {'xy_window':(90,90), 'x_start_stop':[1280-1090, 1090], 'y_start_stop':[407,497], 'xy_overlap':(0.8, 0.0)},
        {'xy_window':(90,90), 'x_start_stop':[1280-1090, 1090], 'y_start_stop':[457,547], 'xy_overlap':(0.8, 0.0)},
        {'xy_window':(64,64), 'x_start_stop':[1280-990, 990], 'y_start_stop':[410,474], 'xy_overlap':(0.7, 0.0)},
    ]
}
