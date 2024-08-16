mkdir models
mkdir models/pretrained_model
wget https://www.kaggle.com/api/v1/models/andrewstell/popular-vits-for-interpretation/pyTorch/default/1/download
mv download models/pretrained_model/download
cd models/pretrained_model/
tar -xvf download
rm download


wget http://calvin-vision.net/bigstuff/proj-imagenet/data/gtsegs_ijcv.mat