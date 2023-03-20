
import sys


def prepare_segmentation_paths():
    """Run this function if you are outside segmentation project folder and want
    to use segmentation scripts. Uggly hack to be removed in the future."""
    SEG_SOURCE_DIR = '/home/per/medsensio/HS-projects/segmentation/src'
    if sys.path.count(SEG_SOURCE_DIR) == 0:
        # Add path to segmentation src folder
        sys.path.insert(0, SEG_SOURCE_DIR)



def and_recursive(statement_array):
    """Version of np.logical_and that takes a list of boolean arrays as input,
    and allows specification of how to handle NaN values. statement_array is a
    list of boolean arrays."""
    n_statement_lists = len(statement_array)
    n_statements = len(statement_array[0])
    and_tracker = [False]*n_statements
    for i in range(n_statement_lists):
        and_tracker = np.logical_and(and_tracker, statement_array[i])
    and_tracker = list(and_tracker)
    # If a statement contains only nan, set the output for that comparison to
    # nan
    return and_tracker