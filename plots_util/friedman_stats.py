import  numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import rankdata
from scipy.stats import friedmanchisquare
from matplotlib import rcParams
import  matplotlib
import  matplotlib.ticker as ticker
import  Orange as orange
rcParams['font.family'] = 'Times New Roman'
rcParams['mathtext.default']='regular'
rcParams['mathtext.default']='regular'
rcParams['font.size']=14 #14 5 16
#del matplotlib.font_manager.weight_dict['roman']
#matplotlib.font_manager._rebuild()


#Accuracies for 11 datasets on 12 distinct methods
all_averages =[
    [93.36,  78.71,   61.24,  42.88,   67.61,  58.73, 66.29,  53.22,  56.64,  71.65,  78.76,  72.87],
    [84.25,  64.73,   76.19,  73.91,   79.94,  77.90, 83.32,  80.26,  83.18,  74.36,  63.35,  64.81],
    [99.32,  97.99,   97.21,  97.69,   99.74,  99.59, 99.54,  96.86,  92.25,  90.51,  98.29,  95.94],
    [98.97,  96.70,   86.52,  98.61,   99.39,  99.49, 99.66,  99.62,  96.04,  96.28,  97.47,  91.66],
    [97.62,  81.98,   82.83,  40.63,   85.89,  80.00, 87.35,  89.60,  69.84,  57.72,  72.91,  41.40],
    [95.12,  73.09,   3.20 ,  5.75 ,   50.30,  35.81, 59.54,  83.66,  35.44,  84.37,  16.90,  38.66],
    [99.84,  98.84,   95.68,  82.80,   99.75,  99.68, 99.68,  90.98,  54.28,  98.13,  99.00,  94.24],
    [99.90,  99.70,   24.99,  43.51,   27.31,  23.91, 13.61,  98.32,  50.10,  90.54,  66.79,  94.52],
    [88.53,  74.84,   25.00,  26.38,   15.26,  17.94, 7.10 ,  74.09,  18.70,  75.24,  40.40,  68.23],
    [93.69,  93.62,   86.17,  92.51,   90.19,  88.44, 93.58,  92.54,  64.76,  92.80,  94.26,  92.49],
    [99.61,  95.28,   95.71,  95.68,   95.64,  95.63, 95.66,  95.66,  95.59,  95.87,  95.33,  93.78]

]

title_list = np.asarray(['Our','Client Stream','FedAvg', 'FedProx', 'U-Ensemble', 'PFNM', 'Decentralized', 'FedPer', 'pFedMe','DWM','ADWIN','LSNE'])

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 2000)

mean_all = np.around(np.mean(all_averages,axis=0),2)
df = pd.DataFrame(mean_all,index=title_list)


data_rank =np.asarray(all_averages)

for d in range(len(data_rank)):
    data_rank[d]=len(data_rank[d])-rankdata(data_rank[d])+1


fried_null_hypo, pvalue = friedmanchisquare(*all_averages)
print(fried_null_hypo, pvalue)


mean_rank = np.mean(data_rank,axis=0)
print(np.around(mean_rank,2))

names = title_list
avranks =  mean_rank
cd = orange.evaluation.scoring.compute_CD(mean_rank, 11) #tested on 14 datasets for 12 distinct methods
print(cd)


orange.evaluation.scoring.graph_ranks(avranks, names,width=5,cd=cd,reverse=True)
plt.savefig('../main_plots/nemeyi_cyber2.png',bbox_inches="tight")
plt.show()



