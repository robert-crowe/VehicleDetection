
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

