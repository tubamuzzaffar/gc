import subprocess
from sklearn.model_selection import ParameterGrid


param_grid = {'epoch' :[1000], 'lr':[0.0005, 0.001,0.00146,0.01,0.015,0.0175,0.02], 'dr':[0.03,0.4,0.5,0.6,0.7,0.8], 
              'hn':[[500,75],[100,75],[500,25],[100,25],[500,150],[500,300],[450,150]], 'hf':[5,10,15]}

grid = ParameterGrid(param_grid)

for params in grid:
  subprocess.call("python train.py -d ml_100k --accum stack -do " + str(params['dr']) + " -nleft -nb 2 -e " + str(params['epoch']) + " --features --feat_hidden "+ str(params['hf']) +" --testing", shell=True)
