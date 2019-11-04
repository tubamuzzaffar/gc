import subprocess
from sklearn.grid_search import ParameterGrid

subprocess.call("python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e 1000 --features --feat_hidden 10 --testing", shell=True)