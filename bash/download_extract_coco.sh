COCO_DATASET=test2017

# Downloading the coco dataset
wget http://images.cocodataset.org/zips/$COCO_DATASET.zip

# Extracting files into subdirectories
filenames=($(7z l test2017.zip|grep .jpg|awk -F' ' '{print $6}'))
n_files = ${#filenames[@]}
for i in {1..30}
do
    splice=$((i*1000))
    7z e test2017.zip -oimages/train/$splice ${filenames[*]:splice-1000:splice}
done

7z e test2017.zip -oimages/valid/ ${filenames[*]:30000:35000}
7z e test2017.zip -oimages/test/ ${filenames[*]:35000:n_files}