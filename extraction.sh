# Extracting train, valid and test archives
mkdir images/
7z e thumbnails_jpg.7z

# Extracting all train subarchives into a train folder
mkdir images/train && mv thumbnails_jpg_train.7z images/train/
cd images/train && 7z e thumbnails_jpg_train.7z && rm thumbnails_jpg_train.7z
# Extracting images from train subarchives
for subarchive_i in {0..4}
do
    subarchive="${subarchive_i}0000"
    echo $subarchive
    7z e "thumbnails_jpg_train_${subarchive}.7z" -o"$subarchive/"
done
find . -name '*_train_*.7z' -delete

# Extracting valid and test subfolders
7z e thumbnails_jpg_valid.7z -oimages/valid/ &&\
7z e thumbnails_jpg_test.7z -oimages/test/

# Cleaning
rm -rf thumbnails_jpg train/thumbnails_jpg thumbnails_jpg_valid.7z  thumbnails_jpg_test.7z