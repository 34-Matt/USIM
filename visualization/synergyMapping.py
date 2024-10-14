import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
from typing import List, Tuple, Union
from dataManipulation.MymeekerStructure import MeekerObject, humanHandStruct, allegroHandStruct, shadowHandStruct, shadow4HandStruct, shadow3HandStruct


# Read command line arguments
# Print help functions
def print_help():
    print("Usage: `python synergyMapping.py -h`")
    print("Usage: `python synergyMapping.py filename1 struct1 filename2 struct2 ...`")
    print("Usage: `python synergyMapping.py filename1 struct1 style1 filename2 struct2 style2 ...`")
    print("Usage: `python synergyMapping.py -s plotstyle filename1 struct1 filename2 struct2 ...`")
    print("Usage: `python synergyMapping.py -o savename filename1 struct1 filename2 struct2 ...`")
    print("filename1, filename2, ...: The names of the files with save data.")
    print("struct1, struct2, ...: The names of the structures in the files. Either `synergy` or hand structure (i.e. `human`, `shadow5`, ...).")
    print("style1, style2, ...: The styles of the plots. Such as `--k`.")
    print("plotstyle: The style of the entire figure. Either a pre-existing sytle like `ggplot` or a link to a custom style sheet.")
    print("savename: The name of the file to save the plot to.")
    sys.exit()

if len(sys.argv) == 1:
    print_help()
elif '-h' in sys.argv:
    print_help()

args = sys.argv[1:]

# Check for plot style
if '-s' in args:
    style = args[args.index('-s')+1]
    args.remove('-s')
    args.remove(style)
else:
    style = os.path.join(os.path.dirname(__file__), 'powerpoint.mplstyle')

plt.style.use(style)

# Check for save name
if '-o' in args:
    savename = args[args.index('-o')+1]
    args.remove('-o')
    args.remove(savename)
else:
    savename = 'output'

# Check for correct number of arguments
if len(args) == 0:
    raise ValueError("No arguments given. Expect arguemtents either multiple of filenames, structs or filenames, structs, styles.")

def extract_pairs(args: List[str]) -> Tuple[List[Tuple[str, str]], bool]:
    pairs = []
    if len(args) % 2 != 0:
        return pairs, False
    for i in range(0, len(args), 2):
        pairs.append((args[i], args[i+1]))
    
    # Check that every file name is a .csv file
    if not all([f.endswith('.csv') for f, _ in pairs]):
        return pairs, False
    return pairs, True

def extract_triples(args: List[str]) -> Tuple[List[Tuple[str, str, str]], bool]:
    triples = []
    if len(args) % 3 != 0:
        return triples, False
    for i in range(0, len(args), 3):
        triples.append((args[i], args[i+1], args[i+2]))
    
    # Check that every file name is a .csv file
    if not all([f.endswith('.csv') for f, _, _ in triples]):
        return triples, False
    return triples, True

# Extract pairs and triples
pairs, is_pair = extract_pairs(args)
triples, is_triple = extract_triples(args)

# Check that the arguments are in the correct format
if not is_pair and not is_triple:
    raise ValueError("Arguments are not in the correct format. Expecting either pairs of filenames and structs or triples of filenames, structs, and styles.")
elif is_pair:
    data = pairs
else:
    data = triples

# Garbage collection
del pairs
del triples

# Read data
def createMeekerObject(handType:str) -> MeekerObject:
    if handType.lower() == 'human':
        return MeekerObject(humanHandStruct, ['rh_WRJ1', 'rh_WRJ2'])
    elif handType.lower() == 'allegro':
        return MeekerObject(allegroHandStruct)
    elif handType.lower() == 'shadow':
        return MeekerObject(shadowHandStruct, ['rh_WRJ1', 'rh_WRJ2'])
    elif handType.lower() == 'shadow4':
        return MeekerObject(shadow4HandStruct)
    elif handType.lower() == 'shadow3':
        return MeekerObject(shadow3HandStruct)
    else:
        raise ValueError('Hand type not recognized', handType)

df_list = []
for filename, struct, *style in data:
    df = pd.read_csv(filename, index_col=0)
    if struct == 'synergy':
        df_list.append(df)
    else:
        meeker = createMeekerObject(struct)
        df = df.apply(meeker.forwardProject, axis=1)
        df_list.append(df)

# Plot data
fig, ax = plt.subplots(3, 1)
for count in range(len(df_list)):
    if is_pair:
        ax[0].plot(df_list[count]['alpha'])
        ax[1].plot(df_list[count]['sigma'])
        ax[2].plot(df_list[count]['epsilon'])

    elif is_triple:
        ax[0].plot(df_list[count]['alpha'],data[count][2])
        ax[1].plot(df_list[count]['sigma'],data[count][2])
        ax[2].plot(df_list[count]['epsilon'],data[count][2])

# Format plot
ax[0].set_ylabel('Spread')
ax[1].set_ylabel('Bend')
ax[2].set_ylabel('Curl')

ax[0].get_xaxis().set_visible(False)
ax[1].get_xaxis().set_visible(False)

for axis in ax:
    axis.set_ylim(0,1)
    
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['bottom'].set_visible(False)

fig.tight_layout()
plt.savefig(savename+"_continuous.png")
plt.close(fig)

# Plot comparison
for count1 in range(len(df_list)-1):
    for count2 in range(count1+1, len(df_list)):
        diff = df_list[count1] - df_list[count2]
        fig, ax = plt.subplots(3)
        ax[0].plot(diff['alpha'])
        ax[1].plot(diff['sigma'])
        ax[2].plot(diff['epsilon'])

        ax[0].set_ylabel('Spread')
        ax[1].set_ylabel('Bend')
        ax[2].set_ylabel('Curl')

        # Remove x-axis labels
        ax[0].get_xaxis().set_visible(False)
        ax[1].get_xaxis().set_visible(False)

        for axis in ax:
            axis.set_ylim(-1,1)

            axis.spines['top'].set_visible(False)
            axis.spines['right'].set_visible(False)
            axis.spines['bottom'].set_visible(False)

        fig.tight_layout()
        plt.savefig(savename+f"_comparison_{count1}_{count2}.png")
        plt.close(fig)

        fig = plt.figure()
        axis = fig.add_subplot(111)
        average = diff.mean()
        axis.bar(['Spread', 'Bend', 'Curl'], average)
        plt.savefig(savename+f"_comparison_{count1}_{count2}_bar.png")
        plt.close(fig)
