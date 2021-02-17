# Interactive Weak Supervision: Learning Useful Heuristics for Data Labeling
Code for text data experiments in: 

`Benedikt Boecking, Willie Neiswanger, Eric Xing, and Artur Dubrawski (2021). Interactive Weak Supervision: Learning Useful Heuristics for Data Labeling. International Conference on Learning Representations (ICLR).` https://arxiv.org/abs/2012.06046

Please check out the `IWS.ipynb` notebook. The notebook will walk you through an example application of IWS from start to finish. You can choose to
perform the experiment yourself, or to simulate an oracle which responds to queries about proposed labeling functions.  



## Dependencies
If you have access to GPUs and want to ensure they can be used, please first check the correct way to install torch on your system here:
https://pytorch.org  

Once you have installed pytorch or if you don't care about using a GPU for now, you can install all requirements as follows:

```
pip install -r requirements/requirements_pip.txt
# or
conda install -c conda-forge -c pytorch --file requirements/requirements_conda.txt
```

## Data

Download data to reproduce our text experiments
```
cd datasets
wget https://ndownloader.figshare.com/files/25732838?private_link=860788136944ad107def -O iws_datasets.tar.gz
tar -xzvf iws_datasets.tar.gz
rm iws_datasets.tar.gz
```

Please see `datasets/README.md` for links and references to the original data sources and please cite the original sources where appropriate.  



## License

The code in this repository is shared under the MIT license, available in the LICENSE file.
