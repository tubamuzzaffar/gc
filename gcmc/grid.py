import subprocess
from sklearn.model_selection import ParameterGrid
import random

random.seed(50)

param_grid = {'epoch' :[1000, 100, 50,500], 'lr':[0.0005, 0.001,0.00146,0.01,0.015,0.0175,0.02], 'dr':[0.03,0.4,0.5,0.6,0.7,0.8],'hf':[5,10,15], 'ac':['stack','sum'], 'hn':[(50,8),(10,7),(50,3),(100,25),(40,15),(500,75),(45,15)]}

grid = list(ParameterGrid(param_grid))

for params in grid:
  print(params['ac'], str(params['dr']), str(params['lr']), str(params['hf']))
  epoch = str(random.sample(params['epoch'],1)[0])
  ac = str(random.sample(params['ac'],1)[0])
  dr = str(random.sample(params['dr'],1)[0])
  lr = str(random.sample(params['lr'],1)[0])
  hf = str(random.sample(params['hf'],1)[0])
  x = random.sample(params['hn'],1)[0])
  first = str(x[0])
  second = str(x[1])

  #subprocess.check_call(["./train.py", "-d", "ml_100k","--accum",  str(params['ac']),"-do",str(params['dr']),"-nleft", "-nb" , "2", "-e",str(params['epoch']),"--features", "--feat_hidden", str(params['hf']),"--testing" ])
  subprocess.call("python train.py -d ml_100k --accum " + ac +" -do " + dr + " -nleft -nb 2 -e " + epoch + " --features --feat_hidden "+ hf +" --testing --learning_rate " + lr + " --hidden " + first +" "+ second ,shell=True)

