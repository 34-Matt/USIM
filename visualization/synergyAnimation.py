import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import pandas as pd
import glob
import cv2
import re
import argparse

from dataManipulation.MymeekerStructure import MeekerObject, humanHandStruct, allegroHandStruct, shadowHandStruct, shadow4HandStruct, shadow3HandStruct

plt.rcParams['lines.linewidth'] = 4.0

# Read command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("groundtruth_directory", help="Path to groundtruth directory")
parser.add_argument("groundtruth_domain", help="The structure domain of the groundtruth")
parser.add_argument("predictions", help="Path to predictions.csv")
parser.add_argument("predictions_domain", help="The structure domain of the predictions")
parser.add_argument("focuslist", nargs='+', type=int, help="List of points to focus on")
parser.add_argument("--output", default="animation.gif", help="Path to output animation file")
parser.add_argument("--focusdelay", type=int, default=50, help="Number of frames to focus on each point")

args = parser.parse_args()

# Load Data
## Meeker object for forward projection
def createMeekerObject(handType:str) -> MeekerObject:
    if handType.lower() == 'human':
        return MeekerObject(humanHandStruct, ['rh_WRJ1', 'rh_WRJ2'])
    elif handType.lower() == 'allegro':
        return MeekerObject(allegroHandStruct)
    elif handType.lower() == 'shadow' or handType.lower() == 'shadow5':
        return MeekerObject(shadowHandStruct, ['rh_WRJ1', 'rh_WRJ2'])
    elif handType.lower() == 'shadow4':
        return MeekerObject(shadow4HandStruct)
    elif handType.lower() == 'shadow3':
        return MeekerObject(shadow3HandStruct)
    else:
        raise ValueError('Hand type not recognized', handType)

# Ground truth data
csv_list = glob.glob(args.groundtruth_directory + "/*.csv")
if len(csv_list) != 1:
    raise ValueError("There should be exactly one csv file in the groundtruth directory")
df_gt = pd.read_csv(csv_list[0], index_col=0)
if args.groundtruth_domain.lower() != "synergy":
    meeker = createMeekerObject(args.groundtruth_domain)
    if 'rh_WRJ1' in df_gt.columns:
        df_gt = df_gt.drop('rh_WRJ1', axis=1)
    if 'rh_WRJ2' in df_gt.columns:
        df_gt = df_gt.drop('rh_WRJ2', axis=1)
    df_gt = df_gt.apply(meeker.forwardProject, axis=1)

# Predictions data
df_pred = pd.read_csv(args.predictions, index_col=0)
if args.predictions_domain.lower() != "synergy":
    meeker = createMeekerObject(args.predictions_domain)
    if 'rh_WRJ1' in df_pred.columns:
        df_pred = df_pred.drop('rh_WRJ1', axis=1)
    if 'rh_WRJ2' in df_pred.columns:
        df_pred = df_pred.drop('rh_WRJ2', axis=1)
    df_pred = df_pred.apply(meeker.forwardProject, axis=1)

# Get image name format
imgname_example = glob.glob(args.groundtruth_directory + "/*.png")[0]
numbers = re.findall(r'\d+', imgname_example)
if len(numbers) == 0:
    raise ValueError("No digits found in the image name")
digits = len(numbers[-1])
digits_loc = imgname_example.find(numbers[-1])
prefix = imgname_example[:digits_loc]
suffix = imgname_example[digits_loc+digits:]

# Check if the number of rows in the groundtruth and predictions match
if df_gt.shape[0] != df_pred.shape[0]:
    raise ValueError("The number of rows in the groundtruth and predictions do not match")
data_count = df_gt.shape[0]

# Setup figure
syn_height = 1
split = 0.1
img_height = 0.6
max_height = img_height + 3*(syn_height + split)

imgbox_size = 0.2

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlim(0, data_count)
ax.set_ylim(0, max_height)
ax.set_axis_off()

poseline = ax.plot([0, 0], [0, max_height], "k--", linewidth=2)[0]
truthlines = ax.plot([], [], ":", [], [], ":", [], [], ":", color="#377eb8", label="Ground Truth")
predlines = ax.plot([], [], "-", [], [], "-", [], [], "-", color="#e41a1c", label="Predictions")

imgbox_start, imgbox_y = ax.transData.transform([0, 0])
imgbox_end, _ = ax.transData.transform([data_count, 0])
fig_width = fig.bbox.width
fig_height = fig.bbox.height
imgbox_start = imgbox_start / fig_width
imgbox_end = imgbox_end / fig_width
imgbox_y = imgbox_y / fig_height - 0.1
imgbox = fig.add_axes([imgbox_start-0.5*imgbox_size, imgbox_y, imgbox_size, imgbox_size])
imgbox_position = np.linspace(imgbox_start, imgbox_end, data_count)

# Setup labels along y-axis
ax.text(-0.1, 0, "Image", transform=ax.transAxes, rotation=90, ha='center', va='center')
ax.text(-0.1, 0.3, "Curl", transform=ax.transAxes, rotation=90, ha='center', va='center')
ax.text(-0.1, 0.6, "Bend", transform=ax.transAxes, rotation=90, ha='center', va='center')
ax.text(-0.1, 0.9, "Spread", transform=ax.transAxes, rotation=90, ha='center', va='center')

# Setup legend
ax.legend([truthlines[0], predlines[0]], ["Ground Truth", "Prediction"], loc='upper right')

# Setup and hide key images
keybox_list = [fig.add_axes([imgbox_start-0.5*imgbox_size, imgbox_y, imgbox_size, imgbox_size]) for _ in args.focuslist]
keyline_list = [ax.plot([], [], "k--", linewidth=2)[0] for _ in args.focuslist]
for keybox_count in range(len(args.focuslist)):
    key_x_point = args.focuslist[keybox_count]
    key_x_fig = ax.transData.transform([key_x_point,0])[0] / fig_width

    keybox_list[keybox_count].axis('off')
    keybox_list[keybox_count].set_position([key_x_fig-0.5*imgbox_size, imgbox_y, imgbox_size, imgbox_size])
    image = cv2.imread(prefix + str(key_x_point).zfill(digits) + suffix)
    keybox_list[keybox_count].imshow(image)
    keybox_list[keybox_count].set_visible(False)

    keyline_list[keybox_count].set_data([key_x_point, key_x_point], [0, max_height])
    keyline_list[keybox_count].set_visible(False)

# Setup animation
total_preframes = data_count + len(args.focuslist)*args.focusdelay
total_frames =  total_preframes+ 2*args.focusdelay
frame_offset = 0
skipcount = 0

def update(frame: int):
    if frame < total_preframes:
        return update_preview(frame)
    else:
        return update_focus(frame)

def update_preview(frame: int):
    global frame_offset, skipcount
    movement_frame = frame - frame_offset
    x = np.arange(movement_frame)
    if movement_frame in args.focuslist and skipcount < args.focusdelay:
        frame_offset += 1
        skipcount += 1
    elif skipcount >= args.focusdelay:
        skipcount = 0
    # Update current position
    poseline.set_data([movement_frame, movement_frame], [0, max_height])

    # Plot Spread
    offset = img_height + split + 2*(syn_height + split)
    truthlines[0].set_data(x, df_gt['alpha'][:movement_frame]+offset)
    predlines[0].set_data(x, df_pred['alpha'][:movement_frame]+offset)

    # Plot Bend
    offset = img_height + split + syn_height + split
    truthlines[1].set_data(x, df_gt['sigma'][:movement_frame]+offset)
    predlines[1].set_data(x, df_pred['sigma'][:movement_frame]+offset)

    # Plot Curl
    offset = img_height + split
    truthlines[2].set_data(x, df_gt['epsilon'][:movement_frame]+offset)
    predlines[2].set_data(x, df_pred['epsilon'][:movement_frame]+offset)

    # Plot Image
    ## Read image
    imgbox.clear()
    image_name = prefix + str(movement_frame).zfill(digits) + suffix
    image = cv2.imread(image_name)
    imgbox.imshow(image)
    imgbox.axis('off')

    ## Get new pos
    imgbox.set_position([imgbox_position[movement_frame]-0.5*imgbox_size, imgbox_y, imgbox_size, imgbox_size])

    return [poseline, *truthlines, *predlines, imgbox]

def update_focus(frame: int):
    for keybox, keyline in zip(keybox_list, keyline_list):
        keybox.set_visible(True)
        keyline.set_visible(True)
    poseline.set_visible(False)
    imgbox.set_visible(False)
    return keybox_list

# Create and save animation
anime = FuncAnimation(
    fig=fig,
    func=update,
    frames=total_frames
    #frames=200 # For testing
)

FFwriter = FFMpegWriter(fps=20)
anime.save(args.output, writer = FFwriter)
