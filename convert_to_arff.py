import arff
import pandas as pd
import  numpy as np
import json
import  data_utils
import os
def load_data(use_data=None):

    data_load = np.load('dataset/'+use_data+str('.npy'))
    print(data_load.shape)

    return  np.asarray(data_load).astype(np.float)

def creat_dataset(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

if __name__ ==  "__main__":

    client_no = 30
    dataset_p = "kdd_10"
    directory_p = "arff_data/"+dataset_p+"_"+str(client_no)
    creat_dataset(directory_p)
    data_load = load_data(use_data=dataset_p)
    data_client, client_name = data_utils.StreamClientData(data_load, client_no)

    if (data_load[:, -1] == 0).any():
        print('Okay')
    else:
        print('Label Transformation')
        data_load[:, -1] = data_load[:, -1] - 1

    for cl_key in data_client.keys():
        print('Current Client: ', cl_key, end="\n")

        data = data_client[cl_key]
        #print(data.shape)
        attributes = data.shape[1]
        print(data.shape)

        df = pd.DataFrame(data=data, columns = ["attr_"+str(i+1) for i in  range(attributes-1)]+['label'])
        dict_obj = {"attributes":[(col, u'NUMERIC' if col=="label" else u'REAL')  for col in list(df.columns)],
                     "data": df.values.tolist(),
                      u'description': u'',
                     "relation": 'electricity_'+cl_key

                    }
        arff_doc = arff.dumps(dict_obj)

        output_filename = directory_p+"/"+dataset_p+"_"+cl_key +'.arff'
        with open(output_filename, "w") as fp:
            fp.write(arff_doc)
