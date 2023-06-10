# SFLEDS: Semi-supervised federated learning on evolving data streams
Cobbinah B. Mawuli et al. 

Published in Information Sciences Volume 643 [[paper]](https://www.sciencedirect.com/science/article/abs/pii/S0020025523008204)

Detail code implementation and experimental setting for FedLimited. For details, see the paper: SFLEDS: Semi-supervised federated learning on evolving data streams.

Federated learning has attracted increasing attention in recent years, and many algorithms have been proposed. However, existing federated learning algorithms often focus on static data, which tend to fail in the scenarios of data streams. Due to the varying distributions/concepts within and among the clients over time, the joint learning model must learn these different emerging concepts dynamically and simultaneously. The task becomes more challenging when the continuous arriving data are partially labeled for the participating clients. In this paper, we propose SFLEDS (Semi-supervised Federated Learning on Evolving Data Streams), a new federated learning prototype-based method to tackle the problems of label scarcity, concept drift, and privacy preservation in the federated semi-supervised evolving data stream environment.





![SFLEDS Framework](https://github.com/mvisionai/FedLimited/blob/main/asset/Framework.png)

#Usage
Install python dependencies.
```shell
conda create --name sfleds python=3.8 -y
conda activate sfleds

pip install -r requirements.txt
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
