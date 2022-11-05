COCO_DATASET=test2017

# Downloading the coco dataset
if ! test -f "test2017.zip"
then
    wget http://images.cocodataset.org/zips/$COCO_DATASET.zip
fi

# Extracting files into subdirectories
filenames=($(7z l test2017.zip|grep .jpg|awk -F' ' '{print $6}'))
n_files=${#filenames[@]}
for i in {1..40}
do
    splice=$((i*1000))
    subdir=$splice
    if ((i < 10)); then subdir="0${splice}"; fi
    
    if ((i < 31)); then set="train"
    elif ((i < 36)); then set="valid"
    else set="test"
    fi

    echo $subdir
    7z e test2017.zip -oimages/$set/$subdir ${filenames[*]:(($splice-1000)):1000}
    mogrify -resize 256x256! images/$set/$subdir/*.jpg
done