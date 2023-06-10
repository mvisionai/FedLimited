#!/usr/bin/env python3
# Copyright (c) Cobbinah.
# All rights reserved.
# This source code is licensed under the license found in the

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from sklearn.preprocessing import MinMaxScaler,StandardScaler,normalize
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from local_microcluster import  MicroClsuters
from server_microcluster import  ServerMicroClsuters
from sklearn.neighbors import NearestNeighbors
import random
import tqdm
import scipy as sp
import collections
import time
import socket
import argparse
import torch as torch
import syft as sy
import warnings
from scipy import stats
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN

hook = sy.TorchHook(torch)
hostname=str(socket.gethostname())
print("Experiment running on server "+hostname,end="\n")
acc_win_max_size=100
k_neigbours=[1]
#rng = np.random.RandomState(2021)

np.random.seed(17)

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
def args_parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cr4', help="name of dataset")
    parser.add_argument('--clients', type=int, default=10, help='Number of clients')
    parser.add_argument('--hetero', type=str2bool, default=False, const=True, nargs='?', help='Enable true if train and test needs mutual labels')
    parser.add_argument('--max_mc', type=int, default=200, help='max client micro-cluster')
    parser.add_argument('--global_mc', type=int, default=1000, help='max global micro-cluster')
    parser.add_argument('--features', type=int, default=2, help='Number of dataset features')
    parser.add_argument('--clustering', type=str, choices=["kmeans","dbscan"], default="kmeans", help='Method for clustering')
    parser.add_argument('--decay_rate', type=float, default=0.000002, help='Number of dataset features')
    parser.add_argument('--weight_const', type=float, default=0.06, help='Weight threshold constant')
    parser.add_argument('--global_weight', type=float, default=0.50, help='Global Weight threshold constant, ignore')
    parser.add_argument('--local_init', type=int, default=50, help='Local initial cluster for single train')
    parser.add_argument('--data_part', type=str, default="iid",choices=["iid","non_iid"], help='simulate a non-iid and iid data partition')
    parser.add_argument('--global_init', type=int, default=50, help='global initial cluster for fed train')
    parser.add_argument('--reporting_interval', type=int, default=100, help='global initial cluster for fed train')
    parser.add_argument('--percent_init', type=float, default=0.01, help='set initial cluster number with percentage')
    parser.add_argument('--available_label', type=list, default=[0.10,0.15,0.20], help='set initial cluster number with percentage')
    parser.add_argument('--run_type', choices=['fed','single','client'], default='fed',help='set initial cluster number with percentage')
    args = parser.parse_args()
    return args

def load_data(use_data=None):

    data_load = np.load('dataset/'+use_data+str('.npy'))
    scalar  = StandardScaler()
    data_load[:,0:-1] = scalar.fit_transform(data_load[:,0:-1])
    print(data_load.shape)
    return  np.asarray(data_load).astype(np.float)
def load_initial(data,args):
    class_data={}

    limit_size=int(args.percent_init *len(data))

    initial_load=data[0:limit_size,:]
    all_classes=np.unique(initial_load[:,args.features])

    for aclass in list(all_classes):
        class_data[int(aclass)]=initial_load[initial_load[:,args.features]==aclass]

    return  class_data
def partition_client_class(initial_load,zero_index_features):

    class_data={}

    all_classes=np.unique(initial_load[:,zero_index_features])

    #print("unique_class",all_classes)
    for aclass in list(all_classes):
        class_data[int(aclass)]=initial_load[initial_load[:,zero_index_features]==aclass]

    return  class_data
def load_stream_data(data,args, partial_label):

    initial_size = int(args.percent_init*len(data))
    stream_load = data[initial_size+1:len(data)+1,:]

    # Account for unlabel stream
    stream_len = len(stream_load)
    numpy_zeros = np.expand_dims(np.zeros((stream_len)), 1)
    stream_load = np.append(stream_load, numpy_zeros, axis=1)
    unlabel_stream = int(stream_len * partial_label)
    selected_indices = np.random.choice(range(stream_len - 1), unlabel_stream, replace=False)

    # set a flag of 1 for unlabeled stream
    stream_load[selected_indices, -1] = 1
    print(stream_load.shape)
    return stream_load
def initial_model(class_data,args):
    cluster_no,zero_index_features = args.global_init, args.features
    microClusters=MicroClsuters()
    microClusters.emptyMicrocluster()
    cluster_val=cluster_no

    #iterate over each class data
    for keys,data_clus in class_data.items():

        #check condition for microclusters
        if data_clus.shape[0]<=cluster_val:
            cluster_val=int(data_clus.shape[0])//2

            if cluster_val==0:
                cluster_val=1
        cluster_val = int(cluster_val)

        if args.clustering == "kmeans":
            # Create cluster for each class data using KMEANS
            kmeans = KMeans(n_clusters=cluster_val, random_state=0).fit(data_clus[:,0:zero_index_features])
            clu_center = kmeans.cluster_centers_
            clus_label = np.asarray(kmeans.labels_)

            for i in range(cluster_val):
                each_cluster=data_clus[ClusterIndicesNumpy(i,clus_label)][:,0:zero_index_features]
                num_points= each_cluster.shape[0]
                #creating microcluster
                microClusters.setMicrocluster(each_cluster,int(keys),0,num_points,[])
            cluster_val = cluster_no
        elif args.clustering == "dbscan":
            db = DBSCAN(eps=0.5, min_samples=10).fit(data_clus[:,0:zero_index_features])
            clus_label = np.asarray(db.labels_)
            # Get number of clusters generated for each class
            number_clusters = set(clus_label)
            print("db scan clusters: ", len(number_clusters), number_clusters)
            if -1 in number_clusters:
                # noise available
                cluster_val = len(number_clusters) - 1
                start_iterate = -1
            else:
                # no noise available
                cluster_val = len(number_clusters)
                start_iterate = 0
            # create micro cluster for each cluster data
            for i in range(start_iterate, cluster_val):
                each_cluster = data_clus[ClusterIndicesNumpy(i, clus_label)][:, 0:zero_index_features]
                num_points = each_cluster.shape[0]
                microClusters.setMicrocluster(each_cluster, int(keys), 0, num_points,[])
    return  microClusters
def ClusterIndicesNumpy(clustNum, labels_array): #numpy
    return np.where(labels_array == clustNum)[0]
def convert_to_numpy(data={}):

    numpy_data=[]

    for keys, data_clus in data.items():
        numpy_data.append(list(data_clus))

    return np.asarray(numpy_data)
def parallize_fd_limited(client_key,client_data,data_instance,client_prototype,correct_count,acc_window,weights,global_prototype,args,workers_client,label_number):

    max_weight=  weights.argmax(axis=0)
    numpy_convert = convert_to_numpy(client_prototype.getMicrocluster())

    data_t = client_data[client_key][data_instance][:-2]
    class_data = int(client_data[client_key][data_instance][-2])
    flag_label = int(client_data[client_key][data_instance][-1])
    currentTime =data_instance

    selected_cluster = {}
    p_label = {}

    for j, kc in enumerate(k_neigbours):

        # print("len ",len(micro_model.getMicrocluster()))
        tem_center = [list(ex) for ex in np.asarray(numpy_convert[:, 5]).tolist()]
        knn_search = NearestNeighbors(n_neighbors=kc)
        knn_search.fit(tem_center)
        neighboaurs = knn_search.kneighbors(data_t.reshape(1, -1), return_distance=False)
        neighboaurs = neighboaurs[0]

        best_clusters = numpy_convert[neighboaurs]
        selected_cluster[j] = (best_clusters, neighboaurs)
        predicted_labels = numpy_convert[neighboaurs][:, 3]
        unique_predicted = np.unique(predicted_labels)

        p_label[j] = predicted_labels

        if acc_window.shape[1] == acc_win_max_size:
            acc_window = np.zeros((len(k_neigbours), 1))

        if j == 0:

            if acc_window.shape[1] > 1:
                eidx = acc_window.shape[1] - 1
            else:
                eidx = acc_window.shape[1]

            new_acc_adj = np.zeros((len(k_neigbours), 1))
            acc_window = np.column_stack((acc_window, new_acc_adj))

        else:
            eidx = acc_window.shape[1] - 1

        if flag_label == 1:
            if class_data == int(predicted_labels[0]):
                acc_window[j, eidx] = 1
            else:
                acc_window[j, eidx] = 0

    #print("My weights",max_weight)
    weighted_cluster, cluster_indices = selected_cluster[max_weight[0]]
    weighted_label = p_label[max_weight[0]]

    #weighted label
    label_weighted=weighted_label[0]

    # class consistency check
    if flag_label == 1 :
        correct_label_index = np.where(weighted_cluster[:, 3] == int(class_data))[0]
        incorrect_label_index = np.where(weighted_cluster[:, 3] != int(class_data))[0]

        incorrect_micro_index = np.asarray(cluster_indices)[incorrect_label_index].tolist()
        correct_micro_index = np.asarray(cluster_indices)[correct_label_index].tolist()

        # update of current available microclusters by index
        client_prototype.updateMicroClsuter(incorrect_micro_index, 7, -1)
        client_prototype.updateMicroClsuter(correct_micro_index, 7, 1, currentTime)

    # update model
    client_prototype = client_prototype.updateReliability(currentTime, args.decay_rate, args.weight_const)
    numpy_convert_2 = convert_to_numpy(client_prototype.getMicrocluster())
    cluster_center = numpy_convert_2[:, 5]

    neigh_search = NearestNeighbors(n_neighbors=1)
    neigh_search.fit(np.asarray(cluster_center).tolist())
    neighs = neigh_search.kneighbors(data_t.reshape(1, -1), return_distance=True)

    # picking cluster minimum cluster distance and cluster predicted
    predicted_distance = neighs[0][0][0]
    predicted_cluster = neighs[1][0][0]

    current_clus = np.asarray(client_prototype.getSingleMC(predicted_cluster))
    original_radius = current_clus[2]
    clus_label = current_clus[3]

    global_radius = original_radius

    #if data_instance%10==0:
    current_global_concept = global_prototype.getCurrentGlobalConcept(client_key,args,label_number,args.global_mc)
    #current_global_concept = {}
    #print(len(current_global_concept))

    if len(current_global_concept) > 0:

        global_convert = convert_to_numpy(current_global_concept)
        global_search = NearestNeighbors(n_neighbors=1)
        global_search.fit(np.asarray(global_convert[:, 5]).tolist())
        gneighboaurs = global_search.kneighbors(data_t.reshape(1, -1), return_distance=True)

        global_distance = gneighboaurs[0][0][0]
        predicted_global = gneighboaurs[1][0][0]

        single_proto = global_convert[predicted_global]
        current_global_clus = np.asarray(single_proto)
        global_radius = current_global_clus[2]
        global_label = current_global_clus[3]
        client_sample = current_global_clus[10]
        client_instance = current_global_clus[11]
        global_flag = current_global_clus[4]

        if global_label == class_data:
            global_prototype = global_prototype.updateSingleReliability(client_sample,client_instance,
                                                             currentTime, args.decay_rate, args.weight_const)

        # and global_flag==1
        if global_distance < predicted_distance:
            label_weighted=global_label

        #delete cluster  if condition
        if (global_distance<predicted_distance) and global_label !=class_data  and global_flag==1 :
            global_prototype=global_prototype.deleteMC(client_sample,client_instance)


    # get correctly predicted label
    if label_weighted == class_data:
        correct_count = correct_count + 1

    if ((predicted_distance <= original_radius and class_data == clus_label and flag_label==1) or
            (predicted_distance <= original_radius and flag_label!=1)):

        client_prototype = client_prototype.updateMcInfo(client_data[client_key][data_instance], predicted_cluster, currentTime)

    else:
        client_prototype = client_prototype.createNewMc(client_data[client_key][data_instance], original_radius, currentTime, args.max_mc)

    #added to delete
    # if (predicted_distance < original_radius) and weighted_label[0] != class_data and flag_label == 1:
    #     client_prototype = client_prototype.deleteMC(predicted_cluster)

    #remember original radiusp
    local_high_reliability = client_prototype.getHighReliabilty(args.weight_const,client_key,workers_client,map_state=False)
    global_prototype = global_prototype.uploadReliability(client_key,local_high_reliability,unmap_state=False)
    weights = np.sum(acc_window, axis=1) / acc_window.shape[1]
    global_prototype = global_prototype.globalUpdateReliability(args.weight_const,data_instance,args.decay_rate)
    return {client_key:[client_prototype,correct_count,acc_window,weights,global_prototype]}


def parallize_client_limited(client_key,client_data,data_instance,client_prototype,correct_count,acc_window,weights,args):

    max_weight=  weights.argmax(axis=0)
    numpy_convert = convert_to_numpy(client_prototype.getMicrocluster())

    data_t = client_data[client_key][data_instance][:-2]
    class_data = int(client_data[client_key][data_instance][-2])
    flag_label = int(client_data[client_key][data_instance][-1])
    currentTime =data_instance

    selected_cluster = {}
    p_label = {}

    for j, kc in enumerate(k_neigbours):

        # print("len ",len(micro_model.getMicrocluster()))
        tem_center = [list(ex) for ex in np.asarray(numpy_convert[:, 5]).tolist()]
        knn_search = NearestNeighbors(n_neighbors=kc)
        knn_search.fit(tem_center)
        neighboaurs = knn_search.kneighbors(data_t.reshape(1, -1), return_distance=False)
        neighboaurs = neighboaurs[0]

        best_clusters = numpy_convert[neighboaurs]
        selected_cluster[j] = (best_clusters, neighboaurs)
        predicted_labels = numpy_convert[neighboaurs][:, 3]
        unique_predicted = np.unique(predicted_labels)

        p_label[j] = predicted_labels

        if acc_window.shape[1] == acc_win_max_size:
            acc_window = np.zeros((len(k_neigbours), 1))

        if j == 0:

            if acc_window.shape[1] > 1:
                eidx = acc_window.shape[1] - 1
            else:
                eidx = acc_window.shape[1]

            new_acc_adj = np.zeros((len(k_neigbours), 1))
            acc_window = np.column_stack((acc_window, new_acc_adj))

        else:
            eidx = acc_window.shape[1] - 1

        if flag_label == 1:
            if class_data == int(predicted_labels[0]):
                acc_window[j, eidx] = 1
            else:
                acc_window[j, eidx] = 0

        # print("acc",acc_window)
    #print("My weights",max_weight)
    weighted_cluster, cluster_indices = selected_cluster[max_weight[0]]
    weighted_label = p_label[max_weight[0]]
    label_weighted=weighted_label[0]

    # class consistency check
    if flag_label == 1 :
        correct_label_index = np.where(weighted_cluster[:, 3] == int(class_data))[0]
        incorrect_label_index = np.where(weighted_cluster[:, 3] != int(class_data))[0]

        incorrect_micro_index = np.asarray(cluster_indices)[incorrect_label_index].tolist()
        correct_micro_index = np.asarray(cluster_indices)[correct_label_index].tolist()

        # update of current available microclusters by index
        client_prototype.updateMicroClsuter(incorrect_micro_index, 7, -1)
        client_prototype.updateMicroClsuter(correct_micro_index, 7, 1, currentTime)

    # update model
    client_prototype = client_prototype.updateReliability(currentTime, args.decay_rate, args.weight_const)

    numpy_convert_2 = convert_to_numpy(client_prototype.getMicrocluster())
    cluster_center = numpy_convert_2[:, 5]

    neigh_search = NearestNeighbors(n_neighbors=1)
    neigh_search.fit(np.asarray(cluster_center).tolist())
    neighs = neigh_search.kneighbors(data_t.reshape(1, -1), return_distance=True)

    # picking cluster minimum cluster distance and cluster predicted
    predicted_distance = neighs[0][0][0]
    predicted_cluster = neighs[1][0][0]

    current_clus = np.asarray(client_prototype.getSingleMC(predicted_cluster))
    original_radius = current_clus[2]
    clus_label = current_clus[3]

    # get correctly predicted label
    if label_weighted == class_data:
        correct_count = correct_count + 1


    if ((predicted_distance <= original_radius and class_data == clus_label and flag_label==1) or
            (predicted_distance <= original_radius and flag_label!=1)):

        client_prototype = client_prototype.updateMcInfo(client_data[client_key][data_instance], predicted_cluster, currentTime)

    else:
        client_prototype = client_prototype.createNewMc(client_data[client_key][data_instance], original_radius, currentTime, args.max_mc)

    weights = np.sum(acc_window, axis=1) / acc_window.shape[1]
    return {client_key:[client_prototype,correct_count,acc_window,weights]}
def FederatedStreamV2(cleints_data,proto_data,args,partial_label,label_number):
    correct_count=0
    #initializing client prototype
    client_prototypes=collections.OrderedDict()
    global_iterate=collections.OrderedDict()
    csv_client_keys=[]

    for cl_key in cleints_data.keys():
        client_prototypes[cl_key]=initial_model(partition_client_class(proto_data[cl_key],args.features), args)
        #print("client ", cl_key, client_prototypes[cl_key].getClusInstances())
        csv_client_keys.append(cl_key)
    pysyft_hooks = []
    for sy_keys in csv_client_keys:
        key_hook = sy.VirtualWorker(hook, id=sy_keys)
        pysyft_hooks.append(key_hook)

    pysyft_workers = []
    for sy_key in csv_client_keys:
        temp_hook = []
        for hooky in pysyft_hooks:
            if sy_key != hooky.id:
                temp_hook.append(hooky)
            else:
                current_hook = hooky
        warnings.filterwarnings('ignore')
        current_hook.add_workers(temp_hook)
        pysyft_workers.append(current_hook)

    del pysyft_hooks

    server_model = ServerMicroClsuters()
    server_model.emptyMicrocluster()

    acc_window = np.zeros((len(k_neigbours), 1))
    weights = np.ones((len(k_neigbours), 1))
    counter_flag=0
    main_path = 'results' + os.sep + 'fed' + os.sep + args.dataset + os.sep
    make_dir(main_path)
    #looping through clients

    accuracy_step_list=[]
    runtime_file= open(main_path+hostname+"_"+args.clustering+"_"+args.dataset+"_"+str(partial_label)+"_"+str(args.clients)+'_'+str(args.max_mc)+"_"+str(args.global_init)+"_local_fedstream_runtime.txt", "a+")
    continue_status = False
    for i in tqdm.tqdm(range(len(cleints_data['c_1']))):

        if counter_flag == 0:
         start_t=time.time()

        for keys, clt_data in cleints_data.items():

            # initialization for each client
            # looping through client data
            client_prototype=client_prototypes[keys]
            try:
                if i > 0:
                    select_instance = global_iterate[keys + '_' + str(i - 1)]
                    client_prototype = select_instance[0]
                    correct_count = select_instance[1]
                    acc_window = select_instance[2]
                    weights = np.asarray([select_instance[3]])
                error_check=cleints_data[keys][i]
                continue_status = False
            except Exception:
                print("Running Experiment Ended for Client",keys,end="\n")
                continue_status = True
                continue
            fun_ret = parallize_fd_limited(keys, cleints_data, i, client_prototype, correct_count, acc_window,
                                    weights,server_model,args,pysyft_workers,label_number)

            global_iterate[keys+'_'+str(i)]=fun_ret[keys]
            server_model=fun_ret[keys][4]

        #if continue_status:
            #continue

        counter_flag=counter_flag+1
        if (i + 1) % args.reporting_interval == 0:

            counter_flag=0

            local_accuracies=[]
            print("\n")
            for  skeys in  cleints_data.keys():
              stream_instance_check = len(cleints_data[skeys])
              if i+1>=stream_instance_check:
                  accuracy_select = global_iterate[skeys + '_' + str(stream_instance_check - 1)]
                  current_acc = round((accuracy_select[1] / (stream_instance_check + 1)) * 100, 3)
              else:
                accuracy_select=global_iterate[skeys+'_'+str(i-1)]
                current_acc= round((accuracy_select[1] / (i + 1)) * 100,3)
              print("Client-{} Streamed {} data samples with accuracy : {}%".format(skeys, i + 1, current_acc),end="\n")
              local_accuracies.append(current_acc)

            #saving results
            accuracy_step_list.append([i + 1,*local_accuracies])
            df = pd.DataFrame(accuracy_step_list, columns=['step', *csv_client_keys])
            df.to_csv(main_path+hostname+"_"+args.clustering+"_"+args.dataset+"_"+str(partial_label)+"_"+str(args.clients)+"_"+str(args.max_mc)+"_"+str(args.global_mc)+"_"+str(args.global_init)+"_local_fedstream.csv",index=False)

            end_time = time.time()
            seconds_calculate=end_time-start_t
            minutes=int((seconds_calculate)//60)
            seconds=int(seconds_calculate%60)
            print("Execution time for "+str(i + 1)+" ",str(minutes)+":"+str(seconds)+" Global Instances : "+str(len(server_model.getCurrentGlobalConcept(None,args,label_number,100))),end="\n")
            runtime_file.write(str(i+1)+ " " +str(seconds_calculate)+ "\n")

    return convert_to_numpy(server_model.getMicrocluster(keys))
def ClientStreamV2(cleints_data,proto_data,args, partial_label):

    correct_count=0
    #initializing client prototype
    client_prototypes=collections.OrderedDict()
    global_iterate=collections.OrderedDict()
    csv_client_keys=[]


    for cl_key in  cleints_data.keys():
        client_prototypes[cl_key]=initial_model(partition_client_class(proto_data[cl_key],args.features), args.global_init, args.features)
        csv_client_keys.append(cl_key)


    acc_window = np.zeros((len(k_neigbours), 1))
    weights = np.ones((len(k_neigbours), 1))
    flags=0
    counter_flag=0
    #looping through clients
    main_path = 'results' + os.sep + 'client' + os.sep + args.dataset + os.sep
    make_dir(main_path)

    accuracy_step_list=[]
    runtime_file = open(main_path+hostname + "_" + args.dataset + "_" +str(partial_label)+"_"+ str(args.clients) + "_client_runtime.txt", "a+")

    continue_status = False
    for i in tqdm.tqdm(range(len(cleints_data['c_1']))):


        if counter_flag == 0:
         start_t=time.time()

        for keys, clt_data in cleints_data.items():

            # initialization for each client
            # looping through client data
            client_prototype=client_prototypes[keys]

            if i>0:
                select_instance = global_iterate[keys+'_'+str(i-1)]
                client_prototype=select_instance[0]
                correct_count=select_instance[1]
                acc_window=select_instance[2]
                weights=np.asarray([select_instance[3]])

            try:

                error_check = cleints_data[keys][i]

            except Exception:
                print("Running Experiment Ended",end="\n")
                continue_status =True
                continue


            fun_ret=parallize_client_limited(keys,cleints_data, i, client_prototype, correct_count, acc_window,weights,args)
            global_iterate[keys+'_'+str(i)]=fun_ret[keys]

        if continue_status:
            continue

        counter_flag=counter_flag+1
        #print results
        if (i + 1) % args.reporting_interval  == 0:

            counter_flag=0

            local_accuracies=[]
            print("\n")
            for  skeys in  cleints_data.keys():
              accuracy_select=global_iterate[skeys+'_'+str(i-1)]
              current_acc= round((accuracy_select[1] / (i + 1)) * 100,3)
              print("Client-{} Streamed {} data samples with accuracy : {}%".format(skeys, i + 1, current_acc),end="\n")
              local_accuracies.append(current_acc)

            #saving results
            accuracy_step_list.append([i + 1,*local_accuracies])
            df = pd.DataFrame(accuracy_step_list, columns=['step', *csv_client_keys])
            df.to_csv(main_path+hostname+"_"+args.dataset+"_"+str(partial_label)+"_"+str(args.clients)+"_clientstream.csv",index=False)

            end_time = time.time()
            seconds_calculate=end_time-start_t
            minutes=int((seconds_calculate)//60)
            seconds=int(seconds_calculate%60)
            print("Execution time for "+str(i + 1)+" ",str(minutes)+":"+str(seconds),end="\n")
            runtime_file.write(str(i + 1) + " " + str(seconds_calculate) + "\n")
def make_dir(paths):
    if os.path.exists(paths) == False:
        os.makedirs(paths)

def StreamLearning(data,micro_model,args,partial_label):


    acc_window = np.zeros((len(k_neigbours), 1))
    weights = np.ones((len(k_neigbours), 1))
    #max_weight=np.max(weights)
    max_weight=weights.argmax(axis=0)
    accuracy_list=[]
    correct_count=0
    accuracy_step_list=[]
    counter_flag = 0
    main_path = 'results'+os.sep+'single'+os.sep+args.dataset+os.sep
    make_dir(main_path)
    runtime_file = open(main_path+hostname+"_"+args.dataset+"_"+str(partial_label)+'_'+str(args.max_mc)+ "_single_runtime.txt", "a+")

    for i,np_d in enumerate(data):

          if counter_flag == 0:
            start_t = time.time()
         # get center clusters bases on cluster flage
          numpy_convert = convert_to_numpy(micro_model.getMicrocluster())
          data_t=np_d[0:-2]
          class_data=int(np_d[-2])
          flag_label = int(np_d[-1])
          currentTime=i

          selected_cluster = {}
          p_label = {}

          for j,kc in enumerate(k_neigbours):

              knn_search = NearestNeighbors(n_neighbors=kc)
              knn_search.fit(np.asarray(numpy_convert[:,5]).tolist())
              neighboaurs=knn_search.kneighbors(data_t.reshape(1,-1), return_distance=False)
              neighboaurs=neighboaurs[0]

              selected_cluster[j] =(numpy_convert[neighboaurs],neighboaurs)
              predicted_labels=numpy_convert[neighboaurs][:,3]
              unique_predicted=np.unique(predicted_labels)
              p_label[j]=predicted_labels

              if acc_window.shape[1] == acc_win_max_size:
                  acc_window = np.zeros((len(k_neigbours),1))

              if j==0 :

                if acc_window.shape[1]>1:
                    eidx = acc_window.shape[1]-1
                else:
                  eidx = acc_window.shape[1]

                new_acc_adj = np.zeros((len(k_neigbours), 1))
                acc_window = np.column_stack((acc_window, new_acc_adj))

              else:
                  eidx = acc_window.shape[1]-1

              if class_data == int(predicted_labels[0]):
                  acc_window[j, eidx] = 1
              else:
                  acc_window[j, eidx] = 0

          weighted_cluster,cluster_indices = selected_cluster[max_weight[0]]
          weighted_label = p_label[max_weight[0]]

          if flag_label == 1:
              correct_label_index = np.where(weighted_cluster[:, 3] == int(class_data))[0]
              incorrect_label_index = np.where(weighted_cluster[:, 3] != int(class_data))[0]

              incorrect_micro_index = np.asarray(cluster_indices)[incorrect_label_index].tolist()
              correct_micro_index = np.asarray(cluster_indices)[correct_label_index].tolist()

              # update of current available microclusters by index
              micro_model.updateMicroClsuter(incorrect_micro_index, 7, -1)
              micro_model.updateMicroClsuter(correct_micro_index, 7, 1, currentTime)

          # update model
          micro_model = micro_model.updateReliability(currentTime, args.decay_rate, args.weight_const)

          numpy_convert_2 = convert_to_numpy(micro_model.getMicrocluster())
          cluster_center = numpy_convert_2[:, 5]

          neigh_search = NearestNeighbors(n_neighbors=1)
          neigh_search.fit(np.asarray(cluster_center).tolist())
          neighs = neigh_search.kneighbors(data_t.reshape(1, -1), return_distance=True)

          # picking cluster minimum cluster distance and cluster predicted
          predicted_distance = neighs[0][0][0]
          predicted_cluster = neighs[1][0][0]

          current_clus = np.asarray(micro_model.getSingleMC(predicted_cluster))
          original_radius = current_clus[2]
          clus_label = current_clus[3]
          init_psd = current_clus[9]

          #get correctly predicted label
          if weighted_label[0] == class_data:
                correct_count = correct_count + 1

          if ((predicted_distance <= original_radius and class_data == clus_label and flag_label == 1) or
                  (predicted_distance <= original_radius and flag_label != 1)):

              micro_model = micro_model.updateMcInfo(np_d,predicted_cluster, currentTime)
          else:
              micro_model = micro_model.createNewMc(np_d, original_radius, currentTime, args.max_mc)

          #Accuracy Calculation
          current_acc = round((correct_count / (i + 1)) * 100,3)

          if (i+1)% args.reporting_interval == 0:

              counter_flag = 0

              accuracy_step_list.append([i+1,current_acc])

              df = pd.DataFrame(accuracy_step_list, columns=['step', "accuracy"])
              df.to_csv(main_path+hostname+"_"+args.dataset+'_'+str(partial_label)+"_"+str(args.max_mc)+"_singlestream.csv", index=False)
              print("\n Streamed {} data samples with accuracy : {}%".format(i + 1, current_acc))

              end_time = time.time()
              seconds_calculate = end_time - start_t
              minutes = int((seconds_calculate) // 60)
              seconds = int(seconds_calculate % 60)
              print("Execution time for " + str(i + 1) + " ",
                    str(minutes) + ":" + str(seconds) + " Global Instances : " + str(micro_model.getClusInstances()),
                    end="\n")

              runtime_file.write(str(i + 1) + " " + str(seconds_calculate) + "\n")

          weights = np.sum(acc_window,axis=1)/acc_window.shape[1]

    return convert_to_numpy(micro_model.getMicrocluster())
def ClientData(data,num_clients=10,client_initial="c"):
    # create a list of client names

    order_dict=collections.OrderedDict();
    client_names = ['{}_{}'.format(client_initial, i + 1) for i in range(num_clients)]

    # randomize the data
    random.shuffle(data)

    # shard data and place at each client
    size = len(data) // num_clients
    shards = [data[i:i + size] for i in range(0, size * num_clients, size)]

    # number of clients must equal number of shards
    assert (len(shards) == len(client_names))

    return {client_names[i]: shards[i] for i in range(len(client_names))},client_names
def StreamClientData(data,num_clients=10,client_initial="c", data_partition='iid'):
    # create a list of client names

    order_dict=collections.OrderedDict()
    client_names = ['{}_{}'.format(client_initial, i + 1) for i in range(num_clients)]

    # shard data and place at each client
    shards = [[] for i in range(0,num_clients)]

    if data_partition=="iid":
        #new client partition
        counter_check=0

        for i in range(0, len(data), 1):
            shards[counter_check].append(data[i])
            counter_check = counter_check + 1

            if (i+1)%num_clients == 0:
              counter_check=0

        for i in range(len(client_names)):
            order_dict[client_names[i]] = np.asarray(shards[i])

    elif data_partition == "non_iid":
        #num_samples_per_client = np.random.dirichlet(np.ones(num_clients)*100,size=1).flatten() * len(data)
        #num_samples_per_client = np.round(num_samples_per_client).astype(int)
        #start = 0
        #for i in range(len(client_names)):
            #end = start + num_samples_per_client[i]
            #order_dict[client_names[i]] = data[start:end,:]
            #start = end
        for i in range(0, len(data), 1):
            counter_check = np.random.choice(10, 1, p=[0.15, 0.1, 0.1, 0.10, 0.10, 0.10, 0.1, 0.10, 0.05, 0.1])[0]
            shards[counter_check].append(data[i])

        for i in range(len(client_names)):
            print("client ",client_names[i], "data count:",len(shards[i]) )
            order_dict[client_names[i]] = np.asarray(shards[i])

    # number of clients must equal number of shards
    assert (len(shards) == len(client_names))
    return order_dict,client_names

def clientProStreams(cleints_data,args,label_partial):

    proto_data=collections.OrderedDict()
    stream_data=collections.OrderedDict()
    for cl_key in cleints_data.keys():

        client_records = len(cleints_data[cl_key])
        client_init = int(args.percent_init * client_records)
        #print("oo ",client_init, client_records)
        proto_data[cl_key]=cleints_data[cl_key][0:client_init, :]
        client_stream = cleints_data[cl_key][client_init+1:client_records+1, :]

        # Account for unlabel stream
        stream_len = len(client_stream)
        numpy_zeros = np.expand_dims(np.zeros((stream_len)),1)
        client_stream = np.append(client_stream,numpy_zeros,axis=1)
        label_stream = int(stream_len * label_partial)
        selected_indices = np.random.choice(range(stream_len), label_stream, replace=False)

        #set a flag of 1 for labeled stream
        client_stream[selected_indices,-1] = 1
        stream_data[cl_key]= client_stream

    return  stream_data,proto_data
def yieldClientData(client_data):
    yield client_data
def run(args=None):

    # load dataset
    run_type = args.run_type
    data_load = load_data(use_data=args.dataset)

    if (data_load[:, -1] == 0).any():
        print('Okay')
    else:
        print('Label Transformation')
        data_load[:, -1] = data_load[:, -1] - 1

    label_number = len(np.unique(data_load[:, -1]))

    if run_type == "fed":
        #client stream

        for cli in [args.clients]:
            args.clients = cli
            data_client, client_name = StreamClientData(data_load,args.clients,data_partition=args.data_part)

            for label_partial in args.available_label:
              print("Current Limited label: ", str(label_partial * 100), '%')
              # get initial prototype data and each data
              stream_data,proto_data = clientProStreams(data_client,args,label_partial)
              #client stream with fed
              fed_learning = FederatedStreamV2(stream_data,proto_data,args,label_partial,label_number)

    elif run_type == "client":
        # client stream
        #data_client, client_name = StreamClientData(data_load, args.clients)
        # get initial prototype data and each data

        for cli in [10, 20, 30]:
            args.clients = cli
            data_client, client_name = StreamClientData(data_load, args.clients, data_partition=args.data_part)
            for label_partial in  args.available_label:

                print("Current Limited label: ", str(label_partial * 100), '%')
                stream_data, proto_data = clientProStreams(data_client, args, label_partial)
                # individual client stream without fed
                clients_stream = ClientStreamV2(stream_data, proto_data,args, label_partial)


    elif run_type == "single":

        # load initial data
        initial_load = load_initial(data_load,args)
        # create initail micro-cluster prototype
        model_init = initial_model(initial_load, args.local_init, args.features)
        # load stream data

        for label_partial in  args.available_label:
           print("Current Limited label: ",str(label_partial*100),'%')
           stream_data = load_stream_data(data_load,args, label_partial)
           #stream learning
           learned_prototype=StreamLearning(stream_data, model_init, args,label_partial)


if __name__ ==  "__main__":

    arg = args_parser()
    print(arg)
    run(args= arg)

















