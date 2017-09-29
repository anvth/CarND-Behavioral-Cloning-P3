from augment import *
from model import *

# Visualizations will be shown in the notebook.
import pandas as pd
import numpy as np


def load_csv(file_path, col_names, remove_header=False):
    csv = pd.read_csv(file_path, header=None, names=col_names)
    csv = csv.sample(frac=1)
    if remove_header:
        csv = csv[1:]
    
    return csv


# Define our column headers
col_header_names = ["Center", "Left", "Right", "Steering Angle", "Throttle", "Brake","Speed"]


# Let's load our standard driving dataset, where we drive in both directions
csv = load_csv(path + "/driving_log.csv", col_header_names)
csv["Steering Angle"] = csv["Steering Angle"].astype(float) 
print("Dataset has {0} rows".format(len(csv)))


def get_steering_angles(data, st_column, st_calibrations, filtering_f=None):
    """
    Returns the steering angles for images referenced by the dataframe
    The caller must pass the name of the colum containing the steering angle 
    along with the appropriate steering angle corrections to apply
    """
    cols = len(st_calibrations)
    print("CALIBRATIONS={0}, ROWS={1}".format(cols, data.shape[0]))
    angles = np.zeros(data.shape[0] * cols, dtype=np.float32)
    
    i = 0
    for indx, row in data.iterrows():        
        st_angle = row[st_column]
        for (j,st_calib) in enumerate(st_calibrations):  
            angles[i * cols + j] = st_angle + st_calib
        i += 1
    
    # Let's not forget to ALWAYS clip our angles within the [-1,1] range
    return np.clip(angles, -1, 1)

# Defining our columns of interests as well as steering angle corrections
st_angle_names = ["Center", "Left", "Right"]
st_angle_calibrations = [0, 0.25, -0.25]

# In this section we load and train the model...

b_divider = 20
# Multiplying by 3 since we have center, left and right images per row
b_size = len(csv)  * 3 // b_divider

train = csv[:5800]
validation = csv[5801:]

m = nvidia_model()
gen_train = generate_images(train, (160, 320, 3), st_angle_names, "Steering Angle", st_angle_calibrations,  batch_size=b_size)

gen_val = generate_images(validation, (160, 320, 3), st_angle_names, "Steering Angle", st_angle_calibrations,  batch_size=(b_size * b_divider) // 5, data_aug_pct=0.0)
x_val, y_val = next(gen_val)

# Train the model
m.fit_generator(gen_train, validation_data=(x_val, y_val), samples_per_epoch=b_size * b_divider, nb_epoch=5, verbose=1)

m.save("udacity_nvidia_model.h5")

print("Successfully saved model")

