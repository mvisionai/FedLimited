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

art_ten = [

        [99.83     , 99.66    , 0.63   ,  0.64    , 0.63   , 0.66   , 0.63   , 0.64 ],
        [87.35     , 83.01    , 1.19   ,  1.15    , 1.56   , 1.77   , 1.77   , 1.76 ],
        [94.15     , 93.61    , 58.35  ,  59.50   , 56.75  , 61.46  , 61.41  , 58.96 ],
        [99.64     , 97.71    , 95.70  , 95.70    , 95.72  , 95.72  , 95.72  , 95.72 ],
        [97.50     , 83.79    , 0.54   , 0.53     , 0.60   , 0.62    , 0.61   , 0.60 ],
        [81.71     , 58.08    , 0.95   , 1.09     , 0.94   , 1.49    , 1.54   , 1.21 ],
        [94.26     , 93.30    , 58.87  , 57.93    , 57.86  , 56.17   , 56.17  , 55.29 ],
        [98.89     , 95.24    , 95.71  , 95.71    , 95.71  , 95.71   , 95.71  , 95.71 ]
  ]

art_fifteen = [
        [99.82     , 99.63     , 0.59   , 0.60     , 0.61   , 0.63   , 0.62   , 0.62 ],
        [88.0      , 82.25     , 0.97   , 0.90     , 1.05   , 0.97   , 0.98   , 1.20 ],
        [94.15     , 93.71     , 57.97  , 59.08    , 58.57  , 56.32  , 60.91  , 57.27],
        [99.68     , 97.73     , 95.72  , 95.71    , 95.71  , 95.71  , 95.71  , 95.71],
        [98.02     , 93.81     , 0.61   , 0.62     , 0.61   , 0.62   , 0.62    , 0.61 ],
        [83.01     , 58.48     , 1.06   , 1.18     , 0.89   , 1.02   , 0.99    , 1.10 ],
        [94.18     , 93.34     , 61.05  , 57.99    , 58.54  , 59.84  , 59.99   , 58.70 ],
        [98.88     , 95.25     , 95.71  , 95.73    , 95.70  , 95.72  , 95.73   , 95.72 ]
]


art_twenty = [

        [99.81      ,  99.62    , 0.71    , 0.69     , 0.69   , 0.72    , 0.71   , 0.73 ],
        [88.40      ,  82.56    , 1.14    , 1.54     , 2.13   , 1.0     , 1.02   , 1.32 ],
        [94.29      ,  93.75    , 59.32   , 59.37    , 61.33  , 61.78   , 61.78  , 58.86 ],
        [99.70      ,  97.78    , 95.71   , 95.72    , 95.71  , 95.73   , 95.74  , 95.73 ],
        [99.13      , 94.01     , 0.63    , 0.62     , 0.63   , 0.64    , 0.65   , 0.64 ],
        [84.06      , 58.05     , 0.91    , 1.00     , 0.90   , 0.95    , 0.94   , 1.08 ],
        [94.21      , 93.39     , 57.87   , 57.93    , 59.07  , 60.71   , 60.71  , 60.95 ],
        [98.75      , 95.29     , 95.71   , 95.72    , 95.71  , 95.73   , 95.73  , 95.72 ]
]

nat_ten = [

        [78.83      , 71.22     , 69.46    , 67.17      , 68.05      , 66.11    , 71.08    , 64.66 ],
        [99.02      , 98.63     , 96.14    , 96.20      , 96.03      , 95.58    , 95.95    , 96.31 ],
        [94.97      , 92.70     , 92.37    , 92.84      , 92.52      , 17.79    , 95.15    , 95.07] ,
        [89.99      , 85.84     , 60.75    , 60.72      , 60.57      , 60.17    , 58.67    , 11.71 ] ,
        [99.79      , 99.72     , 92.51    , 95.62      , 98.79      , 25.96    , 92.72    , 62.26] ,
        [79.53      , 69.25     , 75.43    , 75.49      , 74.97     , 69.23    , 69.36    , 67.82 ],
        [99.36      , 98.66     , 95.36    , 95.34      , 95.31     , 95.59    , 96.33    , 95.40 ],
        [93.60      , 86.46     , 89.64    , 90.83      , 89.86     , 91.87    , 10.46    , 93.06] ,
        [89.72      ,  71.39    , 60.11    , 60.54      , 58.45     , 61.66    , 61.75    , 61.69],
        [99.70      , 99.47     , 88.40    , 91.92      , 78.11     , 25.96    , 36.99    , 71.03 ]
]

nat_fifteen = [
        [80.38      , 71.54     , 66.05    , 66.36      , 66.15      , 66.45    , 67.47    , 63.78] ,
        [99.13      , 98.62     , 96.28    , 96.41      , 96.27      , 96.47    , 95.12    , 95.05] ,
        [95.94      , 92.64     , 92.73    , 93.17      , 94.68      , 17.39    , 94.71    , 94.78 ],
        [91.10      ,  85.88    , 60.13    , 60.31      , 60.74      , 63.19    , 60.04    , 15.99 ],
        [99.81      , 99.72     , 98.19    , 98.53      , 95.93      , 25.96    , 66.85    , 67.09],
        [80.24      , 69.35     , 71.00    , 70.24      , 73.78      , 68.18    , 65.15    , 65.90] ,
        [99.40      , 98.65     , 95.98    , 95.43      , 95.92      , 95.50    , 95.99    , 96.0] ,
        [94.60      , 86.46     , 90.83    , 91.91      , 94.06      , 92.63    , 94.96    , 93.70 ],
        [90.36      ,  80.02    , 60.66    , 60.46      , 60.54      , 61.37    , 62.14    , 15.35] ,
        [99.80      , 99.46     , 92.40    , 98.31      , 98.68      , 25.96    , 33.74    , 73.48 ]
]


nat_twenty = [
    [81.23      , 71.65     , 66.37    , 65.68      , 65.30     , 66.92    , 66.51    , 66.51],
    [99.27      , 98.62     , 95.89    , 95.25      , 95.29     , 96.39    , 95.84    , 95.87] ,
    [95.94      , 92.81     , 94.50    , 93.06      , 94.78     , 17.79    , 13.91    , 94.93] ,
    [91.84      , 85.88     , 59.94    , 60.01      , 60.98     , 61.02    , 61.54    , 23.30] ,
    [99.85      , 99.72     , 98.93    , 98.96      , 98.70     , 25.96    , 93.68    , 35.59 ],
    [81.01      , 69.45     , 70.87    , 69.57      , 71.22     , 67.18    , 67.11    , 66.98 ] ,
    [98.59      , 98.63     , 96.07    , 96.06      , 96.12     , 96.40    , 95.77    , 96.24],
    [94.50      , 86.51     , 91.80    , 91.26      , 94.06     , 91.26    , 91.94    , 89.46],
    [89.62      , 80.13     , 60.55    , 59.77      , 60.16     , 61.63    , 61.32    , 10.59] ,
    [99.23      , 99.45     , 93.08    , 91.92      , 94.37     , 25.96    , 32.86    , 43.52]
]



title_list = np.asarray(np.asarray(['Our','ClientStream','FedAvg', 'FedProx', 'FedSem', 'FedMatch', 'FedFixMatch', 'FedSiam']))

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', -1)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 2000)



def friedman_test(all_averages):

    mean_all = np.around(np.mean(all_averages, axis=0), 2)
    df = pd.DataFrame(mean_all, index=title_list)

    data_rank = np.around(np.asarray(all_averages),2)



    for d in range(len(data_rank)):
        data_rank[d] = len(data_rank[d]) - rankdata(data_rank[d]) + 1

    fried_null_hypo, pvalue = friedmanchisquare(*all_averages)
    print(fried_null_hypo, pvalue)

    mean_rank = np.mean(data_rank, axis=0)
    #print(np.around(mean_rank, 2))
    mean_df = pd.DataFrame(np.around(mean_rank, 2).reshape(1,-1),columns=title_list)

    print(mean_df)

    names = title_list
    avranks = mean_rank
    cd = orange.evaluation.scoring.compute_CD(mean_rank, 8)  # tested on 9 datasets for 8 distinct methods
    print(cd)

    orange.evaluation.scoring.graph_ranks(avranks, names, width=5, cd=cd, reverse=True)
    plt.savefig('../plot_results_v2/semi_supervised_fried_v2.png', bbox_inches="tight")
    plt.show()


average = [
     [92.08222222, 88.55074074, 62.28777778, 62.45148148, 61.61333333, 60.18888889,60.08444444, 61.16740741],
     [93.17962963, 87.90703704, 62.7937037,  63.07851852, 62.8837037,  52.85185185,57.18407407, 56.86925926],
     [93.68518519, 88.59185185, 63.06851852, 63.1937037,  63.72259259, 49.84740741, 59.97555556, 57.80888889],
     [93.93111111, 88.59592593, 63.47407407, 63.14333333, 63.80296296, 52.55777778, 54.78555556, 49.79074074]
]

if __name__ == '__main__':

    # all_com_five = np.mean(np.vstack([nat_five,art_five]),axis=0)
    # all_com_ten = np.mean(np.vstack([nat_ten, art_ten]),axis=0)
    # all_com_fifteen = np.mean(np.vstack([nat_fifteen, art_fifteen]),axis=0)
    # all_com_twenty = np.mean(np.vstack([nat_twenty, art_twenty]),axis=0)
    # all_com = np.vstack([all_com_five,all_com_ten,all_com_fifteen,all_com_twenty])

    all_v = np.vstack([nat_ten, art_ten,nat_fifteen, art_fifteen,nat_twenty, art_twenty])
    all_com = all_v.tolist()
    friedman_test(all_com)
