The implementation of HyCoRec: Hypergraph-Enhanced Multi-Preference Learning for Alleviating Matthew Effect in Conversational Recommendation (ACL 2024)

The code is partially referred to MHIM (https://github.com/RUCAIBox/MHIM.)

## Requirements

```
python==3.8.12
pytorch==1.10.1
dgl==0.4.3
cudatoolkit==10.2.89
torch-geometric==2.0.3
transformers==4.15.0
```

## Datasets

[Google Drive](https://drive.google.com/drive/folders/1witl2Ga8pQzAsreQhj4QUH7TldzWKzLa?usp=sharing) | [百度网盘](https://pan.baidu.com/s/1WQoWOSrquIZtJz8AGfg9Cg?pwd=mhim)

Please download the processed datasets from the above links, unzip `data_contrast.zip` and move it to `Contrast/`, unzip `data_mhim.zip` and move it to `HyCoRec/`.

## Quick 

### Contrastive Pre-training

Pre-train the R-GCN encoder:

```
cd Contrast
python run.py -d redial -g 0
python run.py -d tgredial -g 0
```

Then, move the `save/{dataset}/{#epoch}-epoch.pth` file to `/pretrain/{dataset}/`.

The pre-trained encoder on our machine has been saved as `HyCoRec/pretrain/{dataset}/10-epoch.pth`.

### Running

```
cd ../HyCoRec
python run_crslab.py --config config/crs/mhim/hredial.yaml -g 0 -s 1 -p -e 10
python run_crslab.py --config config/crs/mhim/htgredial.yaml -g 0 -s 1 -p -e 10
```

The experiment results on our machine has been saved in `HyCoRec/log/`

## Acknowledgement

The implementation is based on the open-source CRS toolkit [CRSLab](https://github.com/RUCAIBox/CRSLab).

```
@inproceedings{shang2023mhim,
  author = {Chenzhan Shang and Yupeng Hou and Wayne Xin Zhao and Yaliang Li and Jing Zhang},
  title = {Multi-grained Hypergraph Interest Modeling for Conversational Recommendation},
  booktitle = {{arXiv preprint arXiv:2305.04798}},
  year = {2023}
}
```

