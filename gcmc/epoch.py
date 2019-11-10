import subprocess

from sklearn.model_selection import ParameterGrid

import random

random.seed(50)

param_grid = {'epoch' :[1000, 100, 50,500]}

#param_grid = {'hf':[5,10,15]}

grid = list(ParameterGrid(param_grid))


for params in grid:
  epoch = str(params['epoch'])

  #subprocess.check_call(["./train.py", "-d", "ml_100k","--accum",  str(params['ac']),"-do",str(params['dr']),"-nleft", "-nb" , "2", "-e",str(params['epoch']),"--features", "--feat_hidden", str(params['hf']),"--testing" ])
  subprocess.call("python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e "+ epoch + " --features --feat_hidden 10 --testing", shell=True)
  #subprocess.call("python train.py -d ml_100k --accum " + ac +" -do " + dr + " -nleft -nb 2 -e " + epoch + " --features --feat_hidden "+ hf +" --testing --learning_rate " + lr + " --hidden " + first +" "+ second ,shell=True)

  #subprocess.call("python train.py -d ml_100k --accum stack -do 0.7 -nleft -nb 2 -e 1000 --features --feat_hidden "+ hf +" --testing",shell=True)

