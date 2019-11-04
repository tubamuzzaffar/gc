import subprocess
from sklearn.model_selection import ParameterGrid

param_grid = {'epoch' :[10], 'lr':[0.0005, 0.001,0.00146,0.01,0.015,0.0175,0.02], 'dr':[0.03,0.4,0.5,0.6,0.7,0.8],'hf':[5,10,15], 'ac':['stack','sum'], 'hn':[(500,75),(100,75),(500,25),(100,25),(500,150),(500,300),(450,150)]}

grid = list(ParameterGrid(param_grid))

for params in grid:
  print(params['ac'], str(params['dr']), str(params['lr']), str(params['hf']))
  epoch = str(params['epoch'])
  ac = str(params['ac'])
  dr = str(params['dr'])
  lr = str(params['lr'])
  hf = str(params['hf'])
  x = params['hn']
  hid = "["+ str(x[0]) + "," + str(x[1])+"]"
  print(hid)
  #subprocess.check_call(["./train.py", "-d", "ml_100k","--accum",  str(params['ac']),"-do",str(params['dr']),"-nleft", "-nb" , "2", "-e",str(params['epoch']),"--features", "--feat_hidden", str(params['hf']),"--testing" ])
  subprocess.call("python train.py -d ml_100k --accum " + ac +" -do " + dr + " -nleft -nb 2 -e " + epoch + " --features --feat_hidden "+ hf +" --testing --learning_rate " + lr + " --hidden " + "[500,75]" ,shell=True)

