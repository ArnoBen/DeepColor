# Extracting train, valid and test archives
rm -rf images/ ffhq_valid.7z  ffhq_test.7z
mkdir images/
7z e ffhq.7z

# Extracting all train subarchives into a train folder
mkdir images/train && mv ffhq_train.7z images/train/
cd images/train && 7z e ffhq_train.7z && rm ffhq_train.7z
# Extracting images from train subarchives
for subarchive_i in {0..4}
do
    subarchive="${subarchive_i}0000"
    echo $subarchive
    7z e "ffhq_train_${subarchive}.7z" -o"$subarchive/"
done
find . -name '*_train_*.7z' -delete
cd ../..

# Extracting valid and test subfolders
7z e ffhq_valid.7z -oimages/valid/
7z e ffhq_test.7z -oimages/test/

# Cleaning
rm -rf ffhq train/ffhq ffhq_valid.7z  ffhq_test.7z
echo 'Done !'