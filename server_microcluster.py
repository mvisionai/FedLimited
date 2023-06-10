#!/usr/bin/env python3
# Copyright (c) Cobbinah.
# All rights reserved.
# This source code is licensed under the license found in the

"""
Server Micro-cluster Class.
"""
import numpy as np
from scipy.stats import logistic
from sklearn.datasets import make_classification
import numpy as np
from scipy.spatial.distance import cdist
import warnings
import  collections
import sys
from sklearn.mixture import GaussianMixture,BayesianGaussianMixture

class ServerMicroClsuters:

    def __init__( self, data=None,label=None,extime=0,data_pt=0 ):

        self.data=data
        self.extime=extime
        self.label=label
        self.data_pt=data_pt
        self.microclusters={}

    def getClusInstances(self):
       return  len(self.microclusters)

    def getMicrocluster(self,client_key):

        try:
            return self.microclusters[client_key]
        except KeyError as ex:
           return  {}

    def convert_to_numpy(self, data={}):

        numpy_data = []

        for keys, data_clus in data.items():
            numpy_data.append(list(data_clus))

        return np.asarray(numpy_data)


    def getCurrentGlobalConcept(self, client_key,args,label_number,threshold_strong):

        selected_microclusters = {}

        check_trans = 0
        wt = args.weight_const
        features = args.features
        try:

            sed = self.microclusters[client_key]
            sed = self.convert_to_numpy(sed)
            cluster_c_label = np.where(sed[:, 4] == 1)[0]
            label_clu_cen = sed[cluster_c_label]
            center_radius =label_clu_cen[:,5].tolist()
            label = label_clu_cen[:, 3]
            unique_label = np.unique(label)

            for i, va in enumerate(unique_label):
                label[label==va]=i

            lmnn = GaussianMixture(n_components=len(unique_label), random_state=0)   #LFDA(k=len(unique_label))  covariance_type='tied',
            lmnn.fit(center_radius)
            check_trans = 1

        except Exception as e:
            print(e)
            pass

        temporary_check = []
        distance_check = []
        for clients in self.getAllMicrocluster().copy():
            if clients==client_key and client_key is not  None:
                continue
            for index, mc in enumerate(self.microclusters[clients].copy()):

                if check_trans == 1:
                    #if  self.microclusters[clients][mc][7] >= wt:
                    current_microcluster = self.microclusters[clients][mc]
                    #print(current_microcluster[4])
                    if current_microcluster[4]==1:
                      #selected_microclusters[index+1] = current_microcluster
                      #distance_meas = lmnn.score_samples([current_microcluster[5]])
                      #print(distance_meas)
                      #distance_check.append(distance_meas[0])
                      temporary_check.append(current_microcluster)

        #lmnn.score_samples([current_microcluster[5]])
        if check_trans == 1:
            if len(temporary_check)>=1:
                temporary_check=np.asarray(temporary_check)
                #print(len(temporary_check))
                temporary_check_ind = np.where(temporary_check[:, 4] == 1)[0]
                temporary_check_label = temporary_check[temporary_check_ind]
                log_scores = lmnn._estimate_log_prob(np.asarray(temporary_check_label[:,5].tolist()))
                log_scores = np.max(log_scores, axis=-1)

                #print("gh ",lmnn._estimate_log_weights())
                #print("gh 2 ",lmnn._estimate_log_prob(np.asarray(temporary_check_label[:,5].tolist())))
                similar_concepts = np.argsort(log_scores)[-threshold_strong:]

                #similar_concepts = np.arange(len(log_scores))[-threshold_strong:]
                #temporary_check = np.asarray(temporary_check)
                counter_key = 0
                for index,  reliable in enumerate(similar_concepts):
                     if log_scores[reliable]>0.8:
                         counter_key = counter_key+1
                         #current_index = index+1
                         selected_microclusters[counter_key] = temporary_check_label[reliable,:]

        return selected_microclusters

    def getAllMicrocluster(self):

         return self.microclusters


    def getClusterLabelCondition(self):

        numpy_data=[]

        for keys in self.getMicrocluster().copy():

            numpy_data.append(list(self.microclusters[keys]))

        new_data=np.asarray(numpy_data)
        labels_data=new_data[:,3]
        occurs=collections.Counter(labels_data)
        max_occur=occurs.most_common(1)[0][0]

        max_occur_index = np.where(new_data[:, 3] == int(max_occur))[0]
        max_occur_clust = new_data[max_occur_index]

        return  max_occur_index,max_occur_clust


    def mergeMC(self,cluster_index,cluster_s,limit=2,psd=[]):

        clsuter_np=np.asarray(cluster_s[:,5].tolist())
        D=cdist(clsuter_np,clsuter_np)


        #set zero values to 1000

        D[D==0]=1000
        min_value=np.min(D,axis=0)

        min_mc_ind = np.where(D==min_value)[0]
        ##max_occur_clust = new_data[min_mc_ind]

        micro_1_select=min_mc_ind[0]
        micro_2_select=min_mc_ind[2]

        micro_1_map=cluster_index[micro_1_select]
        micro_2_map=cluster_index[micro_2_select]

        first_mc = self.getSingleMC(micro_1_map+1)
        second_mc= self.getSingleMC(micro_2_map+1)

        no_Instances = self.getClusInstances() + 1

        LS=np.add(first_mc[0],second_mc[0])
        SS=np.add(first_mc[1],second_mc[1])
        N_pt=first_mc[8]+second_mc[8]
        label = first_mc[3]
        label_flag = 1
        mc_center = LS/N_pt
        mc_radius = np.sqrt(np.sum(SS/N_pt) - np.sum(np.square((LS/N_pt))))

        mc_time = max(first_mc[6],second_mc[6])
        mc_importance = max(first_mc[7],second_mc[7])

        self.microclusters[no_Instances] = [LS, SS, mc_radius, label, label_flag, mc_center, mc_time, mc_importance, 1,psd]

        ignore_list=[micro_1_map,micro_2_map]

        for clus_in in cluster_index:
            if clus_in not in ignore_list:
                self.microclusters.pop(clus_in + 1)

        # reshuffle microclsuters keys
        new_instance_cluster = {}
        for index, keys in enumerate(self.getMicrocluster().copy()):
            new_instance_cluster[index + 1] = self.microclusters[keys]

        self.microclusters = new_instance_cluster

        return  self

    def updateMicroClsuter(self,cluster_index,data_index,data,ctime=0):

        for clus in cluster_index:

                if data==1:

                   self.microclusters[clus+1][data_index] = self.microclusters[clus+1][data_index]+ data
                   self.microclusters[clus + 1][6] = ctime

                else:
                    self.microclusters[clus + 1][data_index] = self.microclusters[clus+1][data_index]+ data


        return  self.microclusters


    def getSingleMC(self,index):

        return self.microclusters[index + 1]


    def updateSingleReliability(self,client_sample,key,currenTime,lmda,wt):


            for index, mc in enumerate(self.getMicrocluster(client_sample).copy()):
                if self.microclusters[client_sample][mc][11] == key:
                    self.microclusters[client_sample][mc][7] = self.microclusters[client_sample][mc][7] + 1
                    self.microclusters[client_sample][mc][6] = currenTime
                    break

            return self

    def globalUpdateReliability(self,wt,currenTime,lmda):

        for keys in self.getAllMicrocluster().copy():

          for mc in self.microclusters[keys].copy():

            currentImpt = self.microclusters[keys][mc][7]
            previuusTime = self.microclusters[keys][mc][6]
            self.microclusters[keys][mc][7] = currentImpt * (2 ** (-lmda * (currenTime - previuusTime)))

            if self.microclusters[keys][mc][7] < wt:
                self.microclusters[keys].pop(mc)

        # reshuffle microclsuters keys
        new_instance_cluster = {}
        for keys in self.getAllMicrocluster().copy():
            new_instance_cluster[keys] = {}
            for index, mc in enumerate(self.getMicrocluster(keys).copy()):
                new_instance_cluster[keys][index + 1] = self.microclusters[keys][mc]
                #new_instance_cluster[keys][index + 1][11] = index+1

        self.microclusters = new_instance_cluster

        return self

    def uploadReliability(self, client, client_mc = {},unmap_state=False):

        mcs = client_mc[client]

        transform_mcs = {}

        for key in mcs:


            if unmap_state:
                LS = self.unMappedPrototypes(mcs[key][0], True)
                SS = self.unMappedPrototypes(mcs[key][1], True)
                mc_radius =self.unMappedPrototypes(mcs[key][2], False)
                label = int(self.unMappedPrototypes(mcs[key][3], False))
                label_flag = int(self.unMappedPrototypes( mcs[key][4], False))
                mc_center = self.unMappedPrototypes(mcs[key][5], True)
                mc_time = self.unMappedPrototypes(mcs[key][6], False)
                mc_importance = self.unMappedPrototypes(mcs[key][7], False)
                data_pt = int(self.unMappedPrototypes( mcs[key][8],False))
            else:
                LS = mcs[key][0]
                SS = mcs[key][1]
                mc_radius = mcs[key][2]
                label = mcs[key][3]
                label_flag = mcs[key][4]
                mc_center = mcs[key][5]
                mc_time = mcs[key][6]
                mc_importance = mcs[key][7]
                data_pt = mcs[key][8]

            psd_matrix = mcs[key][9]
            id = key

            transform_mcs[key] = [LS, SS, mc_radius, label, label_flag, mc_center, mc_time, mc_importance,data_pt, psd_matrix, client, id]
        try:
           self.microclusters[client].update(transform_mcs)
           #print(client, self.microclusters[client].items())
        except KeyError as ex:
            self.microclusters[client] = transform_mcs

        return self


    def unMappedPrototypes(self,shared_data,numpy_tran=False):

        if numpy_tran:
            return shared_data.get().float_prec().data.numpy()
        else:
            return float(shared_data.get().float_prec().data)

    def updateMcInfo(self,data,clus_index,ctime):

        data_t = data[:-2]
        class_data = int(data[-2])

        mc=np.asarray(self.getSingleMC(clus_index))


        LS =np.add(mc[0] , data_t)
        SS = np.add(mc[1],np.square(data_t))
        N_pt = mc[8]+1

        warnings.filterwarnings('ignore')
        mc_radius = np.sqrt(np.sum(SS / N_pt) - np.sum(np.square((LS /N_pt))))
        mc_time = ctime
        mc_cnter=LS/N_pt

        self.microclusters[clus_index+1][0] =LS
        self.microclusters[clus_index + 1][1] = SS
        self.microclusters[clus_index + 1][8] =N_pt
        self.microclusters[clus_index + 1][5] = mc_cnter
        self.microclusters[clus_index + 1][2] = mc_radius
        self.microclusters[clus_index + 1][6] = mc_time
        self.microclusters[clus_index + 1][9] = mc[9]


        if mc[4] == 0:
            self.microclusters[clus_index + 1][3] = class_data
            self.microclusters[clus_index + 1][4] = 1

        return  self


    def insertClientMC(self,clusters):

        for clus_d in clusters:
          no_Instances = self.getClusInstances() + 1
          self.microclusters[no_Instances] = list(clus_d)

        return self


    def emptyMicrocluster(self):
         self.microclusters={}


    def deleteMC(self,client_sample,key):

        #self.microclusters[client_sample].pop(key)
        #reshuffle microclsuter keys
        new_mc = {}
        counter = 0
        for index, mc in enumerate(self.getMicrocluster(client_sample).copy()):
            if self.microclusters[client_sample][mc][11] ==  key:
                continue
            counter =counter+ 1
            new_mc[counter] = self.microclusters[client_sample][mc]

        self.microclusters[client_sample] = new_mc
        return  self