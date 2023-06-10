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
    [83.04,  72.68,   72.82,  74.76,  72.63,  64.37,  74.46,  77.06,  64.43],
    [99.43,  99.03,   92.75,  90.56,  98.83,  95.45,  93.13,  96.85,  92.73],
    [97.32,  82.36,   61.10,  61.46,  80.90,  42.60,  60.42,  64.38,  43.16],
    [84.05,  77.70,   60.48,  68.08,  78.05,  73.58,  61.40,  80.50,  65.78],
    [99.58,  99.00,   97.85,  98.60,  99.00,  93.58,  98.63,  99.66,  35.63],
    [99.06,  66.60,   25.32,  90.62,  71.72,  94.26,  29.78,  97.48,  92.60],
    [85.42,  37.78,   24.15,  75.52,  49.22,  67.13,  31.23,  83.32,  72.75],
    [99.36,  95.45,   95.90,  95.90,  95.40,  94.10,  95.80,  95.72,  94.73],
    [94.79,  94.00,   84.98,  92.95,  94.03,  92.43,  90.38,  94.20,  91.86]

]

title_list = np.asarray(['Our',' OZAG','ADDEXP', 'DWM', 'ADWIN', 'LNSE', 'HTC', 'SRP', 'AWEC'])

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 2000)

mean_all = np.around(np.mean(all_averages,axis=0),2)
df = pd.DataFrame(mean_all,index=title_list)
print(df.transpose())

data_rank =np.asarray(all_averages)

for d in range(len(data_rank)):
    data_rank[d]=len(data_rank[d])-rankdata(data_rank[d])+1


fried_null_hypo, pvalue = friedmanchisquare(*all_averages)
print(fried_null_hypo, pvalue)


mean_rank = np.mean(data_rank,axis=0)
print(np.around(mean_rank,2))

names = title_list
avranks =  mean_rank
cd = orange.evaluation.scoring.compute_CD(mean_rank, 9) #tested on 14 datasets for 12 distinct methods
print(cd)
orange.evaluation.scoring.graph_ranks(avranks, names,width=5,height=2,cd=cd,reverse=True)
plt.savefig('../plot_results_v2/supervised_fried_v2.png',bbox_inches="tight")
plt.show()



