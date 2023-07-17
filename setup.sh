wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
mkdir -p model/sam
mv sam_vit_.*.pth model/sam
code --install-extension ms-python.python
code --install-extension ms-toolsai.jupyter