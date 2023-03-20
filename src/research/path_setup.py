import sys
import os


curr_dir = os.getcwd()
src_dir = os.path.abspath(
    os.path.join(os.getcwd(), os.pardir)
)
sys.path.insert(0, src_dir)