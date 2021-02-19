# Interactive Weak Supervision: Learning Useful Heuristics for Data Labeling
Code for text data experiments in: 

> [Interactive Weak Supervision: Learning Useful Heuristics for Data Labeling](https://arxiv.org/abs/2012.06046)\
Benedikt Boecking, Willie Neiswanger, Eric Xing, and Artur Dubrawski\
International Conference on Learning Representations ([ICLR](https://iclr.cc/Conferences/2021)), 2021\
_arXiv:2012.06046_


**In brief:**
Please check out the [`IWS.ipynb`](IWS.ipynb) notebook. The notebook will walk you through an example application of IWS from start to finish. You can choose to
perform the experiment yourself, or to simulate an oracle which responds to queries about proposed labeling functions.


## Dependencies
If you have access to GPUs and want to ensure they can be used, please first check the correct way to install PyTorch on your system here:
https://pytorch.org  

Once you have installed pytorch (or if you don't care about using a GPU for now) you can install all requirements by running:

```bash
pip install -r requirements/requirements_pip.txt
# or
conda install -c conda-forge -c pytorch --file requirements/requirements_conda.txt
```

## Data

To download all data for the text experiments, run:
```bash
cd datasets
wget https://ndownloader.figshare.com/files/25732838?private_link=860788136944ad107def -O iws_datasets.tar.gz
tar -xzvf iws_datasets.tar.gz
rm iws_datasets.tar.gz
```

Please see [`datasets/README.md`](datasets/README.md) for links and references to the original data sources and please cite the original sources where appropriate.


## Running the IWS Notebook

To run text data experiments, please see the [`IWS.ipynb`](IWS.ipynb) notebook.

This notebook will walk you through a full example of interactive weak supervision
(IWS), from start to finish. It allows you to choose a text dataset, generate a family
of labeling functions (LFs), and then run IWS on this family of LFs, either with an
automated oracle or by querying you directly for feedback on LFs. It then trains a
downstream classifier via weak supervision methods, using the LFs learned during IWS.


## License

The code in this repository is shared under the MIT license, available in the [LICENSE](LICENSE) file.


## Citation
Please cite [our paper](https://arxiv.org/abs/2012.06046) if you use code from this repo:
```
@inproceedings{boecking2021interactive,
  title={Interactive Weak Supervision: Learning Useful Heuristics for Data Labeling},
  author={Boecking, Benedikt and Neiswanger, Willie and Xing, Eric and Dubrawski, Artur},
  booktitle={International Conference on Learning Representations},
  year={2021}
}
```
