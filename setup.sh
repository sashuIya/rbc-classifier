wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
mkdir -p model/sam
mv sam_vit_h_4b8939.pth model/sam
pip3 install 'git+https://github.com/facebookresearch/segment-anything.git'
pip3 install -U scikit-learn
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter