cd images/thumbnails_jpg

# -m0=Copy is used to store files without compression

# Storing all jpg files NOT starting with 5 or 6 for training
7z a -m0=Copy ../thumbnails_jpg_train_00000.7z '-ir!0*.jpg'
7z a -m0=Copy ../thumbnails_jpg_train_10000.7z '-ir!1*.jpg'
7z a -m0=Copy ../thumbnails_jpg_train_20000.7z '-ir!2*.jpg'
7z a -m0=Copy ../thumbnails_jpg_train_30000.7z '-ir!3*.jpg'
7z a -m0=Copy ../thumbnails_jpg_train_40000.7z '-ir!4*.jpg'
cd ..
7z a -m0=Copy thumbnails_jpg_train.7z '-ir!*_train*.7z'
cd thumbnails_jpg

# Storing all jpg files starting with 5 for validation
7z a -m0=Copy ../thumbnails_jpg_valid.7z '-ir!5*.jpg'

# Storing all jpg files starting with 6 for test
7z a -m0=Copy ../thumbnails_jpg_test.7z '-ir!6*.jpg'

# Storing all these archives into a single one
cd ..
7z a -m0=Copy thumbnails_jpg.7z thumbnails_jpg_train.7z thumbnails_jpg_valid.7z thumbnails_jpg_test.7z

rm thumbnails_jpg_train.7z thumbnails_jpg_valid.7z thumbnails_jpg_test.7z
find . -name '*_train_*' -delete