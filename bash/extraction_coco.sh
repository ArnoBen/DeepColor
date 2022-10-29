# Extracting train, valid and test archives
rm -rf images/ coco_valid.7z  coco_test.7z
mkdir images/
7z e coco.7z

# Extracting all train subarchives into a train folder
mkdir images/train && mv coco_train.7z images/train/
cd images/train && 7z e coco_train.7z && rm coco_train.7z
# Extracting images from train subarchives
for subarchive_i in {0..7}
do
    subarchive="${subarchive_i}0000"
    echo $subarchive
    7z e "coco_train_${subarchive}.7z" -o"$subarchive/"
done
find . -name '*_train_*.7z' -delete
cd ../..

# Extracting valid and test subfolders
7z e coco_valid.7z -oimages/valid/
7z e coco_test.7z -oimages/test/

# Cleaning
rm -rf coco train/coco coco_valid.7z  coco_test.7z
echo 'Done !'