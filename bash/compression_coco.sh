# -m0=Copy is used to store files without compression

# Storing all jpg files NOT starting with 6 or 7 for training

cd images
for subarchive_i in {0..5}
do
    subarchive="${subarchive_i}0000"
    7z a -m0=Copy "../coco_train_${subarchive}.7z" "-ir!${subarchive_i}*.jpg"
done
cd ..
7z a -m0=Copy coco_train.7z '-ir!*_train_*.7z'
cd images

# Storing all jpg files starting with 6 for validation
7z a -m0=Copy ../coco_valid.7z '-ir!6*.jpg'

# Storing all jpg files starting with 7 for test
7z a -m0=Copy ../coco_test.7z '-ir!7*.jpg'

# Storing all these archives into a single one
cd ..
7z a -m0=Copy coco.7z coco_train.7z coco_valid.7z coco_test.7z

rm coco_train.7z coco_valid.7z coco_test.7z
find . -name '*_train_*' -delete