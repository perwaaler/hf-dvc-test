from segmentation.src.utils.general import read_json_from_path

from segmentation.src.utils.segmentation import segment_audio, segment_audio_jointly, LOGGER

LOGGER.setLevel(level="ERROR")

segment_audio_jointly(
    [
        "40170820_hjertelyd_1.wav",
        "40170820_hjertelyd_2.wav",
        "40170820_hjertelyd_3.wav",
        "40170820_hjertelyd_4.wav"
    ]
)




# %%

np.logical_and(t7data.TDIESEPT_T72.values < 0.007,
                t7data.TDIESEPT_T72.values > 0)*1


import numpy as np
arr = np.array([1, 2, 3, 4, np.nan])
result = np.where(np.isnan(arr), arr, arr > 2)
print(result)
# %%


import logging

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)

def logging_function():
    LOGGER.info("some logging information")


#%%





    