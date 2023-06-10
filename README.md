# SFLEDS: Semi-supervised federated learning on evolving data streams
Cobbinah B. Mawuli et al. 

Published in Information Sciences Volume 643 [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0020025523008204)

Detail code implementation and experimental setting for SFLEDS. For details, see the paper: SFLEDS: Semi-supervised federated learning on evolving data streams.

Federated learning has attracted increasing attention in recent years, and many algorithms have been proposed. However, existing federated learning algorithms often focus on static data, which tend to fail in the scenarios of data streams. Due to the varying distributions/concepts within and among the clients over time, the joint learning model must learn these different emerging concepts dynamically and simultaneously. The task becomes more challenging when the continuous arriving data are partially labeled for the participating clients. In this paper, we propose SFLEDS (Semi-supervised Federated Learning on Evolving Data Streams), a new federated learning prototype-based method to tackle the problems of label scarcity, concept drift, and privacy preservation in the federated semi-supervised evolving data stream environment.





![SFLEDS Framework](https://github.com/mvisionai/FedLimited/blob/main/asset/Framework.png)

## Usage
Install python dependencies.
```shell
conda create --name sfleds python=3.8 -y
conda activate sfleds

pip install -r requirements.txt
```

Code configuration.
```
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cr4', help="name of dataset")
parser.add_argument('--clients', type=int, default=10, help='Number of clients')
parser.add_argument('--hetero', type=str2bool, default=False, const=True, nargs='?',
help='Enable true if train and test needs mutual labels')
parser.add_argument('--max_mc', type=int, default=200, help='max client micro-cluster')
parser.add_argument('--global_mc', type=int, default=1000, help='max global micro-cluster')
parser.add_argument('--features', type=int, default=2, help='Number of dataset features')
parser.add_argument('--clustering', type=str, choices=["kmeans","dbscan"], default="kmeans",
help='Method for clustering')
parser.add_argument('--decay_rate', type=float, default=0.000002, help='Number of dataset features')
parser.add_argument('--weight_const', type=float, default=0.06, help='Weight threshold constant')
parser.add_argument('--global_weight', type=float, default=0.50, 
help='Global Weight threshold constant, ignore')
parser.add_argument('--local_init', type=int, default=50, help='Local initial cluster for single train')
parser.add_argument('--data_part', type=str, default="iid",choices=["iid","non_iid"],
help='simulate a non-iid and iid data partition')
parser.add_argument('--global_init', type=int, default=50, help='global initial cluster for fed train')
parser.add_argument('--reporting_interval', type=int, default=100, 
help='global initial cluster for fed train')
parser.add_argument('--percent_init', type=float, default=0.01, 
help='set initial cluster number with percentage')
parser.add_argument('--available_label', type=list, default=[0.10,0.15,0.20],
help='set initial cluster number with percentage')
parser.add_argument('--run_type', choices=['fed','single','client'], default='fed',
help='set initial cluster number with percentage')
```

### Run
The provided `fd_limited.py` can be run on  any computer provided the python depenct requirements are statisfied. Example datasets can be found in the dataset folder. Follow the paper to configure 
other parameter values for exploration. 
```
python fd_limited.py --dataset cr4 --features 2
```

## Contributing

See [contributing](CONTRIBUTING.md).

# Citing SFLEDS
If you find this repository useful, please consider giving a star ⭐ and citation
```
@article{mawuli2023semi,
  title={Semi-supervised Federated Learning on Evolving Data Streams},
  author={Mawuli, Cobbinah B and Kumar, Jay and Nanor, Ebenezer and Fu, Shangxuan and Pan, Liangxu and Yang,
  Qinli and Zhang, Wei and Shao, Junming},
  journal={Information Sciences},
  pages={119235},
  year={2023},
  publisher={Elsevier}
}
```
