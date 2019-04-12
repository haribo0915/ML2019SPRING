wget -O model.h5 https://www.dropbox.com/s/5bylh2pmzu6vfdl/model.h5?dl=1
wget -O model.pth https://www.dropbox.com/s/9pzluc5g10dkv4m/model.pth?dl=1
python saliency_map.py $1 $2
python gradient_ascent.py $1 $2
python filter_vis.py $1 $2
python lime_.py $1 $2