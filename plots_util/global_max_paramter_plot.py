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
xytick_font_size =24
fig, ax = plt.subplots(figsize=(5.8, 5.1)) #(6.8, 6.1) #5.8 5.1
plt.subplots_adjust(left=left, bottom=bottom, right=right, top=top)
y_ticks_label = "Stream"
x_ticks_label = "Accuracy(%)"


x_ticks = [0, 10, 20, 30, 40]

def parameter_max_covtype_k():

    x_values = [50, 100, 150, 200, 250, 300]

    ten = np.asarray([89.99, 89.80, 90.04, 90.07, 90.8, 90.12])
    fifteen = np.asarray([91.10, 91, 91.14, 91.10, 91.12, 91.11])
    twenty = np.asarray([91.84, 91.62, 91.89, 91.82, 91.94, 91.80])

    y_c =   np.around(np.asarray([ten,fifteen,twenty]),2)/100


    markers_on = [0, 1, 2, 3, 4, 5]
    label = ["0.10", "0.15", "0.20"]
    markers = ["o", '+', "p", "P", "h", "*", "D", "x", "^", "3", "4"]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'coral']
    markers_on = [0, 1, 2, 3, 4, 5]
    for i, ds in enumerate(y_c):
        ax.plot(x_values, ds, '-', markevery=markers_on, marker=markers[i], color=colors[i],
                markersize=marker_size, label=label[i],
                fillstyle='none',
                markeredgewidth=2)



    y_ticks_label = "Accuracy(%)"
    x_ticks_label = r'$\mathcal{K}$ (number of clusters per class)'
    plt.ylim(0.75, 1.0)
    plt.xlim(46, max(x_values) + 10)
    # plt.yticks([0.70, .85, 1.00], fontsize=xytick_font_size + 1)
    plt.xticks([50, 100, 150, 200, 250, 300], fontsize=xytick_font_size + 1)


    plt.xlabel(x_ticks_label, fontsize=xytick_font_size)
    plt.yticks([0.75, 0.80, 0.85, 0.90, 0.95, 1.00], fontsize=xytick_font_size + 1)

    lg=ax.legend(prop={'size': xytick_font_size - 2.3}, handletextpad=0.2,title="Label Ratio",  loc="lower right",
               ncol=1, borderaxespad=0.,fancybox=True,fontsize='big')
    title = lg.get_title()
    title.set_fontsize(20)

    plt.ylabel(y_ticks_label, fontsize=xytick_font_size)
    plt.savefig('../plot_results_v2/global_max_k.png', bbox_inches="tight")

    plt.show()

def parameter_max_covtype():

    ten =     np.asarray([89.99, 90.00, 90.00, 90.00 , 90.00 , 90.00])
    fifteen = np.asarray([91.10, 91.12, 91.12, 91.12, 91.12, 91.12])
    twenty =  np.asarray([91.84, 91.85, 91.85, 91.85, 91.85, 91.85 ])


    all_list = np.around(np.asarray([ten,fifteen,twenty]),2)/100
    print(all_list)
    x_values =[1.1,1.5,2,2.5,3,3.5]

    label = ["0.10","0.15","0.20"]
    markers = ["o", '+', "p", "P", "h", "*", "D", "x", "^", "3", "4"]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'coral']
    markers_on = [0, 1, 2, 3, 4, 5]
    for i,ds in enumerate(all_list):
        ax.plot(x_values, ds, '-', markevery=markers_on, marker=markers[i], color=colors[i],
                markersize=marker_size,label=label[i],
                fillstyle='none',
                markeredgewidth=2)

    y_ticks_label = "Accuracy(%)"
    x_ticks_label = "maxP (global prototypes)"
    plt.ylim(0.90, 1.0)
    plt.xlim(1.0, max(x_values)+0.1)
    plt.yticks([0.80, 0.85, 0.90, 0.95, 1.00], fontsize=xytick_font_size + 1)
    #plt.yticks([0.90,0.92,  0.94,0.96, 0.98, 1.00], fontsize=xytick_font_size + 1)
    plt.xticks([1.08,1.5,2,2.5,3,3.5],labels=['1K','1.5K','2K','2.5K','3K','3.5K'], fontsize=xytick_font_size + 1)

    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1fK"))
    lg=ax.legend(prop={'size': xytick_font_size - 2.3}, handletextpad=0.2,title="Label Ratio",  loc="lower right",
               ncol=1, borderaxespad=0.,fancybox=True,fontsize='big')
    title = lg.get_title()
    title.set_fontsize(20)

    plt.ylabel(y_ticks_label, fontsize=xytick_font_size)
    plt.xlabel(x_ticks_label, fontsize=xytick_font_size)
    plt.savefig('../plot_results_v2/global_max_v3.png', bbox_inches="tight")

    plt.show()


def parameter_max():

    ten =     np.asarray([97.50,97.451,97.451,97.45,97.45,97.45])
    fifteen = np.asarray([98.02,98.0,98.0,98.0,98.0,98.0])
    twenty =  np.asarray([99.1,98.61,98.61,98.61,98.61,98.61])
    all_list = np.around(np.asarray([ten,fifteen,twenty]),2)/100
    x_values =[1.1,1.5,2,2.5,3,3.5]

    label = ["0.10","0.15","0.20"]
    markers = ["o", '+', "p", "P", "h", "*", "D", "x", "^", "3", "4"]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'coral']
    markers_on = [0, 1, 2, 3, 4, 5]
    for i,ds in enumerate(all_list):
        ax.plot(x_values, ds, '-', markevery=markers_on, marker=markers[i], color=colors[i],
                markersize=marker_size,label=label[i],
                fillstyle='none',
                markeredgewidth=2)

    y_ticks_label = "Accuracy(%)"
    x_ticks_label = "maxP (global prototypes)"
    plt.ylim(0.80, 1.0)
    plt.xlim(1.0, max(x_values)+0.1)
    plt.yticks([0.80,0.85,0.90,0.95,1.00], fontsize=xytick_font_size + 1)
    plt.xticks([1.08,1.5,2,2.5,3,3.5],labels=['1K','1.5K','2K','2.5K','3K','3.5K'], fontsize=xytick_font_size + 1)

    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1fK"))
    lg=ax.legend(prop={'size': xytick_font_size - 2.3}, handletextpad=0.2,title="Label Ratio",  loc="lower right",
               ncol=1, borderaxespad=0.,fancybox=True,fontsize='big')
    title = lg.get_title()
    title.set_fontsize(20)

    plt.ylabel(y_ticks_label, fontsize=xytick_font_size)
    plt.xlabel(x_ticks_label, fontsize=xytick_font_size)
    plt.savefig('../plot_results_v2/global_max_v2.png', bbox_inches="tight")

    plt.show()



def art_client_ratio():


    ten_10 = [99.83, 87.35, 94.15, 99.64]
    ten_30 = [97.50, 81.71, 94.26, 98.89]

    fifteen_10 = [99.82, 88.0, 94.15, 99.68]
    fifteen_30 = [98.02, 83.01, 94.18, 98.88]

    twenty_10 = [99.81, 88.40, 94.29, 99.70]
    twenty_30 = [99.13, 84.06, 94.21, 98.75]

    client_art_10 = [np.mean(ten_10), np.mean(fifteen_10), np.mean(twenty_10)]
    client_art_30 = [ np.mean(ten_30), np.mean(fifteen_30), np.mean(twenty_30)]

    all_list = np.around(np.asarray([client_art_10,client_art_30]),2)/100
    x_values =[1.05,2,3]

    label = ["Client 10","Client 30"]
    markers = ["o", '+', "p", "P", "h", "*", "D", "x", "^", "3", "4"]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'coral']
    markers_on = [0, 1, 2]
    for i,ds in enumerate(all_list):
        ax.plot(x_values, ds, '-', markevery=markers_on, marker=markers[i], color=colors[i],
                markersize=marker_size,label=label[i],
                fillstyle='none',
                markeredgewidth=2)

    y_ticks_label = "Accuracy(%)"
    x_ticks_label = "Label Ratio"
    plt.ylim(0.80, 1.0)
    plt.xlim(1.0, max(x_values)+0.1)
    plt.yticks([0.80,0.85,0.90,0.95,1.00], fontsize=xytick_font_size + 1)
    plt.xticks([1.05,2,3],labels=['0.10','0.15','0.20'], fontsize=xytick_font_size + 1)

    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1fK"))
    lg=ax.legend(prop={'size': xytick_font_size - 2.3}, handletextpad=0.2,title="Client Number",  loc="lower right",
               ncol=1, borderaxespad=0.,fancybox=True,fontsize='big')
    title = lg.get_title()
    title.set_fontsize(20)

    plt.ylabel(y_ticks_label, fontsize=xytick_font_size)
    plt.xlabel(x_ticks_label, fontsize=xytick_font_size)
    plt.savefig('../plot_results_v2/artifical_semi_clients_v2.png',bbox_inches="tight")

    plt.show()


def natural_client_ratio():


    ten_10 = [78.83, 99.02, 94.97, 89.99, 99.79]
    ten_30 = [79.53, 99.36, 93.60, 89.72, 99.70]

    fifteen_10 = [80.38, 99.13, 95.94, 91.10, 99.81]
    fifteen_30 = [80.24, 99.40, 94.60, 90.36, 99.80]

    twenty_10 = [81.23, 99.27, 95.94, 91.84, 99.85]
    twenty_30 = [81.01, 98.59, 94.50, 89.62, 99.23]

    client_art_10 = [np.mean(ten_10), np.mean(fifteen_10), np.mean(twenty_10)]
    client_art_30 = [ np.mean(ten_30), np.mean(fifteen_30), np.mean(twenty_30)]

    all_list = np.around(np.asarray([client_art_10,client_art_30]),2)/100
    x_values =[1.05,2,3]

    label = ["Client 10","Client 30"]
    markers = ["o", '+', "p", "P", "h", "*", "D", "x", "^", "3", "4"]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'coral']
    markers_on = [0, 1, 2]
    for i,ds in enumerate(all_list):
        ax.plot(x_values, ds, '-', markevery=markers_on, marker=markers[i], color=colors[i],
                markersize=marker_size,label=label[i],
                fillstyle='none',
                markeredgewidth=2)

    y_ticks_label = "Accuracy(%)"
    x_ticks_label = "Label Ratio"
    plt.ylim(0.80, 1.0)
    plt.xlim(1.0, max(x_values)+0.1)
    plt.yticks([0.80,0.85,0.90,0.95,1.00], fontsize=xytick_font_size + 1)
    plt.xticks([1.05,2,3],labels=['0.10','0.15','0.20'], fontsize=xytick_font_size + 1)

    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1fK"))
    lg=ax.legend(prop={'size': xytick_font_size - 2.3}, handletextpad=0.2,title="Client Number",  loc="lower right",
               ncol=1, borderaxespad=0.,fancybox=True,fontsize='big')
    title = lg.get_title()
    title.set_fontsize(20)

    plt.ylabel(y_ticks_label, fontsize=xytick_font_size)
    plt.xlabel(x_ticks_label, fontsize=xytick_font_size)
    plt.savefig('../plot_results_v2/natural_semi_clients_v2.png',bbox_inches="tight")

    plt.show()


def art_client_all_semi_methods():
    art_10 = [
        [99.80, 99.66, 0.62, 0.67, 0.65, 0.69, 0.69, 0.66],
        [86.02, 83.05, 1.06, 1.68, 1.19, 2.09, 2.09, 1.14],
        [93.99, 93.67, 57.46, 59.36, 57.89, 59.44, 59.44, 58.58],
        [99.54, 97.69, 95.72, 95.72, 95.72, 95.71, 95.71, 95.72],
        [99.83, 99.66, 0.63, 0.64, 0.63, 0.66, 0.63, 0.64],
        [87.35, 83.01, 1.19, 1.15, 1.56, 1.77, 1.77, 1.76],
        [94.15, 93.61, 58.35, 59.50, 56.75, 61.46, 61.41, 58.96],
        [99.64, 97.71, 95.70, 95.70, 95.72, 95.72, 95.72, 95.72],
        [99.82, 99.63, 0.59, 0.60, 0.61, 0.63, 0.62, 0.62],
        [88.0, 82.25, 0.97, 0.90, 1.05, 0.97, 0.98, 1.20],
        [94.15, 93.71, 57.97, 59.08, 58.57, 56.32, 60.91, 57.27],
        [99.68, 97.73, 95.72, 95.71, 95.71, 95.71, 95.71, 95.71],
        [99.81, 99.62, 0.71, 0.69, 0.69, 0.72, 0.71, 0.73],
        [88.40, 82.56, 1.14, 1.54, 2.13, 1.0, 1.02, 1.32],
        [94.29, 93.75, 59.32, 59.37, 61.33, 61.78, 61.78, 58.86],
        [99.70, 97.78, 95.71, 95.72, 95.71, 95.73, 95.74, 95.73]
    ]
    # art_20 = [
    #     [97.44, 98.68, 0.54, 0.55, 0.56, 0.66, 0.65, 0.67],
    #     [81.50, 69.98, 1.18, 1.08, 0.96, 1.30, 1.28, 1.88],
    #     [94.05, 93.64, 58.61, 56.56, 58.62, 62.91, 62.91, 60.13],
    #     [99.28, 95.41, 95.72, 95.74, 95.72, 95.72, 95.73, 95.73],
    #     [98.51, 98.71, 0.60, 0.61, 0.64, 0.65, 0.65, 0.64],
    #     [84.16, 68.78, 0.99, 0.93, 1.22, 1.33, 1.32, 1.33],
    #     [94.15, 93.65, 59.86, 59.37, 58.93, 60.49, 60.40, 60.42],
    #     [99.36, 95.44, 95.71, 95.71, 95.75, 95.71, 95.72, 95.70],
    #     [98.81, 98.70, 0.62, 0.63, 0.62, 0.63, 0.61, 0.62],
    #     [85.52, 68.49, 1.07, 1.04, 1.14, 1.13, 1.33, 1.19],
    #     [94.37, 93.68, 58.43, 58.49, 58.23, 58.06, 58.0, 58.0],
    #     [99.41, 95.51, 95.70, 95.72, 95.72, 95.74, 95.74, 95.73],
    #     [99.26, 98.75, 0.62, 0.63, 0.64, 0.66, 0.67, 0.65],
    #     [86.15, 67.43, 0.92, 1.23, 0.97, 0.96, 0.96, 1.05],
    #     [94.37, 93.69, 59.58, 59.37, 59.47, 60.61, 60.60, 59.09],
    #     [99.47, 95.53, 95.68, 95.73, 95.72, 95.72, 95.74, 95.71]
    #
    # ]

    art_30 = [
        [96.01, 93.56, 0.54, 0.53, 0.52, 0.67, 0.66, 0.64],
        [78.82, 56.71, 1.27, 1.14, 1.34, 1.04, 1.08, 1.06],
        [93.92, 93.31, 58.73, 57.82, 54.95, 57.63, 57.56, 57.51],
        [98.48, 95.21, 95.72, 95.72, 95.72, 95.74, 95.75, 95.75],
        [97.50, 83.79, 0.54, 0.53, 0.60, 0.62, 0.61, 0.60],
        [81.71, 58.08, 0.95, 1.09, 0.94, 1.49, 1.54, 1.21],
        [94.26, 93.30, 58.87, 57.93, 57.86, 56.17, 56.17, 55.29],
        [98.89, 95.24, 95.71, 95.71, 95.71, 95.71, 95.71, 95.71],
        [98.02, 93.81, 0.61, 0.62, 0.61, 0.62, 0.62, 0.61],
        [83.01, 58.48, 1.06, 1.18, 0.89, 1.02, 0.99, 1.10],
        [94.18, 93.34, 61.05, 57.99, 58.54, 59.84, 59.99, 58.70],
        [98.88, 95.25, 95.71, 95.73, 95.70, 95.72, 95.73, 95.72],
        [99.13, 94.01, 0.63, 0.62, 0.63, 0.64, 0.65, 0.64],
        [84.06, 58.05, 0.91, 1.00, 0.90, 0.95, 0.94, 1.08],
        [94.21, 93.39, 57.87, 57.93, 59.07, 60.71, 60.71, 60.95],
        [98.75, 95.29, 95.71, 95.72, 95.71, 95.73, 95.73, 95.72]
    ]

    client_art_10 = np.mean(art_10, axis=0)
    #client_art_20 = np.mean(art_20, axis=0)
    client_art_30 = np.mean(art_30, axis=0)

    all_stack = np.vstack([client_art_10, client_art_30]).T

    all_list = np.around(all_stack,2)/100
    x_values =[1.05,3]

    label = ['Our','ClientStream','FedAvg', 'FedProx', 'FedSem', 'FedMatch', 'FedFixMatch', 'FedSiam']
    markers = ["o", '+', "p", "P", "h", "*", "D", "x", "^", "3", "4"]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'coral']
    markers_on = [0, 1]
    for i,ds in enumerate(all_list):
        ax.plot(x_values, ds, '-', markevery=markers_on, marker=markers[i], color=colors[i],
                markersize=marker_size,label=label[i],
                fillstyle='none',
                markeredgewidth=2)

    y_ticks_label = "Accuracy(%)"
    x_ticks_label = "Client K"
    plt.ylim(0.20, 1.0)
    plt.xlim(1.0, max(x_values)+0.05)
    plt.yticks([0.20,0.40,0.60,0.80,1.0], fontsize=xytick_font_size)
    plt.xticks([1.05,3],labels=['10','30'], fontsize=xytick_font_size )

    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1fK"))
    plt.legend(prop={'size': xytick_font_size - 5.0}, handletextpad=0.2, bbox_to_anchor=(0, 1.02), loc="lower left",
               ncol=2, borderaxespad=0.)
    #title = lg.get_title()
    #title.set_fontsize(20)

    plt.ylabel(y_ticks_label, fontsize=xytick_font_size)
    plt.xlabel(x_ticks_label, fontsize=xytick_font_size)
    plt.savefig('../plot_results_v2/semi_all_clients_artificial_methods_v2.png',bbox_inches="tight")

    plt.show()


def natural_client_all_semi_methods():
    nat_10 = [
        [76.34, 71.18, 73.97, 69.85, 71.80, 66.35, 66.18, 66.01],
        [98.73, 98.63, 96.09, 95.90, 95.89, 96.64, 96.06, 96.0],
        [93.88, 92.82, 92.41, 92.84, 88.64, 93.56, 93.09, 92.99],
        [87.83, 85.89, 60.28, 60.57, 60.38, 60.71, 59.12, 24.74],
        [99.76, 99.72, 83.98, 86.80, 91.29, 88.17, 88.23, 88.69],
        [78.83, 71.22, 69.46, 67.17, 68.05, 66.11, 71.08, 64.66],
        [99.02, 98.63, 96.14, 96.20, 96.03, 95.58, 95.95, 96.31],
        [94.97, 92.70, 92.37, 92.84, 92.52, 17.79, 95.15, 95.07],
        [89.99, 85.84, 60.75, 60.72, 60.57, 60.17, 58.67, 11.71],
        [99.79, 99.72, 92.51, 95.62, 98.79, 25.96, 92.72, 62.26],
        [80.38, 71.54, 66.05, 66.36, 66.15, 66.45, 67.47, 63.78],
        [99.13, 98.62, 96.28, 96.41, 96.27, 96.47, 95.12, 95.05],
        [95.94, 92.64, 92.73, 93.17, 94.68, 17.39, 94.71, 94.78],
        [91.10, 85.88, 60.13, 60.31, 60.74, 63.19, 60.04, 15.99],
        [99.81, 99.72, 98.19, 98.53, 95.93, 25.96, 66.85, 67.09],
        [81.23, 71.65, 66.37, 65.68, 65.30, 66.92, 66.51, 66.51],
        [99.27, 98.62, 95.89, 95.25, 95.29, 96.39, 95.84, 95.87],
        [95.94, 92.81, 94.50, 93.06, 94.78, 17.79, 13.91, 94.93],
        [91.84, 85.88, 59.94, 60.01, 60.98, 61.02, 61.54, 23.30],
        [99.85, 99.72, 98.93, 98.96, 98.70, 25.96, 93.68, 35.59]

    ]
    # nat_20 = [
    #
    #     [76.56, 69.20, 75.18, 77.02, 73.49, 70.62, 72.82, 72.63],
    #     [98.92, 98.58, 95.24, 95.27, 95.25, 95.36, 96.26, 95.39],
    #     [92.45, 89.20, 85.47, 87.85, 82.38, 89.75, 86.08, 91.98],
    #     [87.59, 82.24, 59.84, 60.31, 60.14, 60.91, 60.97, 60.61],
    #     [99.72, 99.59, 88.72, 89.24, 85.81, 91.95, 88.54, 88.38],
    #     [79.37, 69.44, 71.47, 72.66, 73.22, 68.32, 64.85, 64.87],
    #     [99.19, 98.61, 96.04, 96.06, 95.99, 96.08, 95.54, 96.03],
    #     [93.80, 89.25, 88.96, 90.25, 92.09, 92.99, 94.72, 95.11],
    #     [89.69, 82.28, 60.46, 60.05, 60.30, 61.95, 58.34, 11.01],
    #     [99.78, 99.59, 89.23, 88.56, 97.29, 25.96, 70.41, 81.46],
    #     [80.42, 69.37, 71.46, 69.38, 70.09, 67.58, 66.74, 66.74],
    #     [99.39, 98.61, 95.94, 96.03, 96.06, 95.97, 95.43, 96.10],
    #     [94.55, 89.20, 92.27, 92.30, 93.09, 17.79, 93.60, 95.11],
    #     [90.70, 82.30, 60.66, 60.75, 60.16, 59.09, 58.61, 61.38],
    #     [99.83, 99.58, 88.77, 88.95, 96.97, 25.96, 97.56, 93.92],
    #     [81.20, 69.68, 70.21, 69.83, 70.09, 67.28, 68.54, 68.13],
    #     [99.41, 98.58, 96.10, 96.06, 95.76, 95.91, 95.88, 95.96],
    #     [95.55, 89.24, 92.63, 90.18, 94.24, 82.74, 22.17, 23.29],
    #     [91.47, 82.31, 60.34, 60.58, 59.84, 61.45, 61.05, 20.77],
    #     [99.83, 99.58, 97.72, 97.13, 98.80, 25.96, 75.84, 41.68]
    #
    # ]

    nat_30 = [
        [77.74, 69.09, 75.51, 75.15, 73.98, 72.84, 70.09, 70.19],
        [99.31, 98.66, 95.91, 95.87, 95.96, 96.12, 95.49, 96.07],
        [91.79, 86.36, 85.47, 84.75, 83.75, 81.45, 24.08, 90.76],
        [87.05, 79.68, 59.41, 59.77, 59.50, 61.11, 61.10, 60.93],
        [99.70, 99.46, 87.12, 88.43, 81.46, 25.96, 90.62, 86.68],
        [79.53, 69.25, 75.43, 75.49, 74.97, 69.23, 69.36, 67.82],
        [99.36, 98.66, 95.36, 95.34, 95.31, 95.59, 96.33, 95.40],
        [93.60, 86.46, 89.64, 90.83, 89.86, 91.87, 10.46, 93.06],
        [89.72, 71.39, 60.11, 60.54, 58.45, 61.66, 61.75, 61.69],
        [99.70, 99.47, 88.40, 91.92, 78.11, 25.96, 36.99, 71.03],
        [80.24, 69.35, 71.00, 70.24, 73.78, 68.18, 65.15, 65.90],
        [99.40, 98.65, 95.98, 95.43, 95.92, 95.50, 95.99, 96.0],
        [94.60, 86.46, 90.83, 91.91, 94.06, 92.63, 94.96, 93.70],
        [90.36, 80.02, 60.66, 60.46, 60.54, 61.37, 62.14, 15.35],
        [99.80, 99.46, 92.40, 98.31, 98.68, 25.96, 33.74, 73.48],
        [81.01, 69.45, 70.87, 69.57, 71.22, 67.18, 67.11, 66.98],
        [98.59, 98.63, 96.07, 96.06, 96.12, 96.40, 95.77, 96.24],
        [94.50, 86.51, 91.80, 91.26, 94.06, 91.26, 91.94, 89.46],
        [89.62, 80.13, 60.55, 59.77, 60.16, 61.63, 61.32, 10.59],
        [99.23, 99.45, 93.08, 91.92, 94.37, 25.96, 32.86, 43.52]

    ]

    client_nat_10 = np.mean(nat_10, axis=0)
    #client_nat_20 = np.mean(nat_20, axis=0)
    client_nat_30 = np.mean(nat_30, axis=0)

    all_stack = np.vstack([client_nat_10, client_nat_30]).T

    all_list = np.around(all_stack,2)/100
    x_values =[1.05,3]

    label = ['Our','ClientStream','FedAvg', 'FedProx', 'FedSem', 'FedMatch', 'FedFixMatch', 'FedSiam']
    markers = ["o", '+', "p", "P", "h", "*", "D", "x", "^", "3", "4"]
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'coral']
    markers_on = [0, 1]
    for i,ds in enumerate(all_list):
        ax.plot(x_values, ds, '-', markevery=markers_on, marker=markers[i], color=colors[i],
                markersize=marker_size,label=label[i],
                fillstyle='none',
                markeredgewidth=2)

    y_ticks_label = "Accuracy(%)"
    x_ticks_label = 'Client K'
    plt.ylim(0.20, 1.0)
    plt.xlim(1.0, max(x_values)+0.05)
    plt.yticks([0.20,0.40,0.60,0.80,1.0], fontsize=xytick_font_size)
    plt.xticks([1.05,3],labels=['10','30'], fontsize=xytick_font_size )

    #ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1fK"))
    plt.legend(prop={'size': xytick_font_size - 5.0}, handletextpad=0.2, bbox_to_anchor=(0, 1.02), loc="lower left",
               ncol=2, borderaxespad=0.)
    #title = lg.get_title()
    #title.set_fontsize(20)

    plt.ylabel(y_ticks_label, fontsize=xytick_font_size)
    plt.xlabel(x_ticks_label, fontsize=xytick_font_size)
    plt.savefig('../plot_results_v2/semi_all_clients_natural_methods_v2.png',bbox_inches="tight")

    plt.show()
if __name__ == "__main__":
    #parameter_max()
    #parameter_max_covtype()
    parameter_max_covtype_k()
    #natural_client_all_semi_methods()
    #art_client_all_semi_methods()

