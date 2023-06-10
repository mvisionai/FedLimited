import matplotlib.pyplot as plt
import  pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import rcParams
from matplotlib.pyplot import figure
rcParams['font.family'] = 'Times New Roman'
rcParams['mathtext.default']='regular'
from matplotlib import rcParams
import  matplotlib.ticker as ticker
from matplotlib.pyplot import figure
rcParams['font.family'] = 'Times New Roman'
rcParams['mathtext.default']='regular'


left = 0.167
bottom= 0.18
right=0.97
top=0.95
marker_size = 11
markerwidth = 2.4  # 2
line_width = 2
legend_size = 28
xylabel_size=24
xytick_font_size =16
fig, ax = plt.subplots(figsize=(7, 5.1))
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
y_ticks_label = "Stream"
x_ticks_label = "Accuracy(%)"


x_ticks = [0, 10, 20, 30, 40]


def parameter_max():

    hun = 93.50/100
    two =  np.mean([93.3,93.398,93.221,93.329,93.338,93.585,93.392,93.4,93.313,93.356])/100
    three =  np.mean([93.587,93.523,93.45,93.433,93.513,93.633,93.527,93.613,93.46,93.36])/100
    four =  np.mean([93.683,93.694,93.588,93.608,93.694,93.788,93.542,93.802,93.606,93.579])/100
    five =  np.mean([93.706,93.829,93.644,93.746,93.765,93.963,93.829,93.777,93.69,93.648])/100
    six =   np.mean([93.846,93.888,93.904,93.76,93.852,94.017,93.873,93.937,93.875,93.829])/100

    x_values =[100,200,300,400,500,600]
    y_c = np.around([hun,two,three,four,five,six],2)

    print(y_c)

    markers_on = [0, 1, 2, 3, 4,5]
    ax.plot(x_values, y_c, '-', markevery=markers_on, marker='D', color='k',
            markersize=marker_size,
            fillstyle='none',
            markeredgewidth=2)

    y_ticks_label = "Accuracy(%)"
    x_ticks_label = "maxMC (local prototypes)"
    plt.ylim(0.70, 1.0)
    plt.xlim(98, max(x_values) + 50)
    plt.yticks([0.70, 0.75, 0.80,0.85,0.90,0.95,1.00], fontsize=xytick_font_size + 1)
    plt.xticks([100, 200, 300, 400,500,600,700], fontsize=xytick_font_size + 1)

    plt.ylabel(y_ticks_label, fontsize=xytick_font_size)
    plt.xlabel(x_ticks_label, fontsize=xytick_font_size)
    plt.savefig('../main_plots/local_max.png')

    plt.show()


def fed_proto():

    hun = 93/100
    two =  np.mean([92.765,92.8,92.906,92.852,92.838,93.008,92.921,92.883,92.923,92.869])/100
    three =  np.mean([92.919,92.99,92.871,92.856,92.923,93.027,92.944,92.779,92.742,92.89])/100
    four =  np.mean([93.008,92.996,92.965,93.038,92.915,93.11,93.044,92.923,92.844,92.996])/100
    five =  np.mean([92.923,92.971,92.96,92.963,93.019,93.208,93.019,92.94,92.854,92.892])/100
    six =   np.mean([92.869,92.892,92.887,92.833,92.937,93.113,92.846,92.887,92.846,92.852])/100

    x_values =[50,100,150,200,250,300]
    y_c = [hun,np.round(two,2),np.round(three,2),np.round(four,2),np.round(five,2),np.round(six,2)]

    print(y_c)

    markers_on = [0, 1, 2, 3, 4,5]
    ax.plot(x_values, y_c, '-', markevery=markers_on, marker='h', color='k',
            markersize=marker_size,
            fillstyle='none',
            markeredgewidth=2)

    y_ticks_label = "Accuracy(%)"
    x_ticks_label = r'$K$ (number of clusters per class)'
    plt.ylim(0.70, 1.0)
    plt.xlim(50, max(x_values) + 50)
    #plt.yticks([0.70, .85, 1.00], fontsize=xytick_font_size + 1)
    plt.xticks([50, 100, 150, 200,250,300,350], fontsize=xytick_font_size + 1)

    plt.ylabel(y_ticks_label, fontsize=xytick_font_size)
    plt.xlabel(x_ticks_label, fontsize=xytick_font_size)
    plt.yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00], fontsize=xytick_font_size + 1)
    plt.savefig('../main_plots/fed_proto.png')
    plt.show()

def max_cluster():

    hun = 93.50/100
    two =  np.mean([93.512,93.744,93.508,93.6,93.637,93.642,93.617,93.533,93.515,93.521])/100
    three =  np.mean([93.952,94.004,94.025,93.877,93.937,93.988,93.912,94.106,93.848,93.942])/100
    four =  np.mean([94.131,93.998,94.017,93.992,94.058,94.125,93.933,94.035,93.881,94.021])/100
    five =  np.mean([94.146,94.133,94.112,94.092,94.125,94.113,94.069,94.233,93.946,94.069])/100
    six =   np.mean([94.212,94.25,94.117,94.09,94.215,94.285,94.223,94.213,94.052,94.213])/100

    x_values =[1,2,3,4,5,6]
    y_c = [hun,two,three,four,five,six]

    markers_on = [0, 1, 2, 3, 4,5]
    ax.plot(x_values, y_c, '-', markevery=markers_on, marker="*", color='k',
            markersize=marker_size,
            fillstyle='none',
            markeredgewidth=2)


    y_ticks_label = "Accuracy(%)"
    x_ticks_label = r'$MaxMC_g$ (global prototypes) '
    plt.ylim(0.70, 1.0)
    plt.xlim(1, max(x_values) + 0.5)

    plt.xticks([1, 2, 3, 4,5,6], fontsize=xytick_font_size + 1)
    plt.yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00], fontsize=xytick_font_size + 1)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%dK"))

    plt.ylabel(y_ticks_label, fontsize=xytick_font_size)
    plt.xlabel(x_ticks_label, fontsize=xytick_font_size)
    plt.savefig('../main_plots/global_max.png')
    plt.show()

if __name__ == "__main__":
    fed_proto()

