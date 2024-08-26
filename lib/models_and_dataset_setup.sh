echo "Downloading models"

mkdir lib/pretrained_model
wget https://www.kaggle.com/api/v1/lib/andrewstell/popular-vits-for-interpretation/pyTorch/default/1/download
mv download lib/pretrained_model/download
cd lib/pretrained_model/
tar -xvf download
rm download



echo "Downloading dataset"

wget http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat
mkdir lib/dataset
mv gtsegs_ijcv.mat lib/dataset/