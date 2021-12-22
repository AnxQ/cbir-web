# cbir-web
Integrated CBIR web application, depend on [Swin-Transformer](https://github.com/microsoft/Swin-Transformer) for 
feature extraction and [Faiss](https://faiss.ai) for similarity search. A deep learning based classification framework 
[MMClassification](https://github.com/open-mmlab/mmclassification) is used to simplify the model migration process.
<br>
The web service architecture is Flask + VueJS.
## Requirements
- Backend
  - torch 1.7+
  - mmcv-full
  - mmclassification
  - flask
- Frontend
  - npm
  - axios
  - vuetify
## Installation
```shell
git clone --recurse-submodules https://github.com/AnxQ/cbir-web

# For CPU
conda install pytorch torchvision -c pytorch
# For CUDA 11.3
conda install pytorch==1.10.0 torchvision==0.11.1 cudatoolkit=11.3 -c pytorch

pip install -r requirements.txt
pip install git+https://github.com/open-mmlab/mim.git
mim install mmcls

cd ../cbir-front
npm install
```
## Model preparation and database generation
1. Download the [pth](https://download.openmmlab.com/mmclassification/v0/swin-transformer/convert/swin_tiny_patch4_window7_224-160bb0a5.pth) for swin-tiny and put it in checkpoints
2. Symbol link or modify the configs in [gen_vectors.py](https://github.com/AnxQ/cbir-web/blob/main/gen_vectors.py)
3. Run `python gen_vectors.py`

## Startup
- Backend
```bash
FLASK_APP = main.py
FLASK_ENV = development
FLASK_DEBUG = 1
python -m flask run
```
- Frontend
```bash
cd cbir-front
npm run serve
```
The application will be run at http://localhost:8082/
