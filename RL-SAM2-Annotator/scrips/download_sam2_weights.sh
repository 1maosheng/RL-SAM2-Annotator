#!/bin/bash
mkdir -p checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2.1_hiera_large.pt
cd ..
echo "Downloaded SAM2 checkpoint"