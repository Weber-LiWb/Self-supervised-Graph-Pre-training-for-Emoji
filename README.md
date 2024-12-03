# Unleashing the Power of Emojis in Texts via Self-supervised Graph Pre-Training

## About
This repo is the official code for "Unleashing the Power of Emojis in Texts via Self-supervised Graph Pre-Training"

## Dependencies
The script has been tested running under Python 3.9, with the following packages installed (along with their dependencies):
```
dgl==1.1.2
emoji==2.12.1
pytorch==2.0.0
scikit-learn==1.2.2
numpy==1.23.5
ninja==1.10.2
tqdm==4.65.0
```

Python module dependencies are listed in requirements.txt, which can be easily installed with pip:

`pip install -r requirements.txt`

## Usage
To run a pretrain process, you can execute `train.py` as follows:
```bash
python train.py --node-feat-dim 768 --dgl_file <graph file> --moco --moco_type 0,1,2 --linkpred --lp_type 0,1,2  --gpu <gpu_id>
```

To generate the emoji code from pretrained model, you can execute `generate.py` as follows:
```bash
python generate.py --load-path <model file> --dgl_file <graph file> --gpu <gpu_id>
```

## Contact
If you have any questions about the code or the paper, feel free to contact me. Email: erictandz@gmail.com

## Cite
If you find this work helpful, please cite (to be continued)


## Acknowledgements
Part of this code is inspired by Yonglong Tian et al.'s [GCC: Graph Contrastive Coding for Graph Neural Network Pre-Training](https://github.com/THUDM/GCC).