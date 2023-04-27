# Connector
Connector is a novel Graph Representation Learning Framework developed by NS Lab, CUK based on pure [PyTorch](https://github.com/pytorch/pytorch) backend. 

<p align=center>
  <a href="https://www.python.org/downloads/release/python-360/">
    <img src="https://img.shields.io/badge/Python->=3.6-3776AB?logo=python&style=flat-square" alt="Python">
  </a>    
  <a href="https://github.com/pytorch/pytorch">
    <img src="https://img.shields.io/badge/PyTorch->=1.4-FF6F00?logo=pytorch&style=flat-square" alt="pytorch">
  </a>       
</p>

## Setup environment for running:

- Running: `python -m venv libenv`

### Connect to the environment:

- On window OS run: `env.bat`
- On linux OS or MAC OS run: `source image/bin/activate`

### Install pip packages:

`pip install -r requirements.txt`

### Inject the developed library: 
You should copy the graphrl folder and move it to libenv/Lib/site-packages to use this framework this time.
Reseacher members will update this part after publishing this library

### To build model:

- Running `python examples/TransE_example.py`

### Contributors

<a href="https://github.com/NSLab-CUK/GraphRL/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=NSLab-CUK/connector" />
</a>


## Citing Connector

Please cite our [paper](https://arxiv.org/abs/2304.13195) if you find *Connector* useful in your work:
```
@misc{nguyen2023connector,
      title={Connector 0.5: A unified framework for graph representation learning}, 
      author={Thanh Sang Nguyen and Jooho Lee and Van Thuy Hoang and O-Joun Lee},
      year={2023},
      eprint={2304.13195},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


<br>

***

<a href="https://nslab-cuk.github.io/"><img src="https://github.com/NSLab-CUK/NSLab-CUK/raw/main/Logo_Dual_Wide.png"/></a>
