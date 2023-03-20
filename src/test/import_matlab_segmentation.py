from utils.z_modules import get_matlab_array, dump_dict_to_json
from utils.prepare_data import prepare_t7data
from utils.general import create_df_of_empty_lists, load_t7_audio, get_subset_idx
from utils.backend import FeaturesPar, dump_python_object_to_file, read_yaml_from_path
from segmentation.src.utils.general import convert_states_to_interval_representation, idx_to_label

import json
import os
import numpy as np
from numpy.matlib import repmat

# Load matlab segmentation
mat_file = "segmentation_matlab"
states_matlab = get_matlab_array("states")
nodes_matlab = get_matlab_array("state_change_points")

# Create dataframe that
t7data = prepare_t7data()
segmentation_df = create_df_of_empty_lists(2124, 4)
for idx, states_ij in np.ndenumerate(states_matlab):
    print(idx)
    state_jumps_ij = nodes_matlab[idx[0]][idx[1]].flatten()-1
    audio_ij = load_t7_audio(
        t7data.id[idx[0]],
        ausc_index=idx[1],
    )
    segmentation_df.loc[idx[0], idx[1]] = append_states(
        state_values=states_ij.flatten()-1,
        nodes=state_jumps_ij,
        n_audio=len(audio_ij),
        sampling_rate=FeaturesPar["audio_sr"]
    )


# Create dictionary
path_directory = "Medsensio/External data/t7_heart/Data - Lars Ailo Bongos files"
segmentation_dict = {}
for i, row_i in segmentation_df.iterrows():
    print(i)
    id = t7data["id"].loc[i]
    list_id_i = []
    
    for j_ausc_pos in range(4):
        file_name = f"{id}_hjertelyd_{j_ausc_pos + 1}.wav"
        path_file = os.path.join(path_directory, file_name)
        dictionary_pos_j = {"location": j_ausc_pos,
                            "path": path_file,
                            "segmentation": row_i[j_ausc_pos]}
        list_id_i.append(dictionary_pos_j)
    
    segmentation_dict[id.astype(str)] = list_id_i

# Write to json file
with open("segmentation_matlab.json", "w") as outfile:
    json.dump(segmentation_dict, outfile)


json_object = json.dumps(segmentation_dict[""], indent = 4)
dump_dict_to_json()

segmentation_matlab_df.loc[0,0]
segmentation_matlab_df["id"] = t7data.id

nodes = np.array([2,4,6,9])-1
state_values = np.array([1,2,3,4])-1
n_audio = 11


def get_previous_state(state):
    previous_state = state - 1 + (state==0) * 4
    return previous_state


def expand_states(nodes, state_labels, n_audio):
    init_state = get_previous_state(state_labels[0])
    state_labels = np.concatenate(([init_state], state_labels))
    nodes_extended = np.concatenate(([0], nodes, [n_audio]))
    state_durations = np.diff(nodes_extended)
    states_expanded = np.zeros(n_audio)
    for i, state_duration in enumerate(state_durations):
        state_expanded = repmat(state_labels[i], 1, state_duration).flatten()
        indices = np.arange(nodes_extended[i], nodes_extended[i+1])
        states_expanded[indices] = state_expanded
    
    return states_expanded


def append_states(
        state_values,
        nodes,
        n_audio,
        sampling_rate,
):
    init_state = get_previous_state(state_values[0])
    state_values_extended = np.concatenate(([init_state], state_values))
    nodes_extended = np.concatenate(([0], nodes, [n_audio]))
    left_endpoints = nodes_extended[:-1]
    right_endpoints = nodes_extended[1:]
    interval_dictionaries = []
    for i, state_index in enumerate(state_values_extended):
        interval_dictionaries.append(
            {"start": left_endpoints[i]/sampling_rate,
             "end": right_endpoints[i]/sampling_rate,
             "label": idx_to_label(state_index),
             }
        )
    return interval_dictionaries



# %%

segmentation_dict_subset = {
    "10003711": segmentation_dict["10003711"],
    "10006209": segmentation_dict["10006209"],
    "10006815": segmentation_dict["10006815"],
}

with open("segmentation_matlab.json", "w") as outfile:
    json.dump(segmentation_dict, outfile)

x = read_yaml_from_path("segmentation_matlab.json")