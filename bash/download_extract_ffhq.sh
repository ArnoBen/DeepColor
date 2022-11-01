%%bash
# Checking if image archive exists
if ! test -f "thumbnails128x128.zip"
then
    gdown 1Wrr6qZA1Tr6r9edNL2nSxnwopMW1n6pR
fi

# Creating new image folder
rm -rf images
mkdir images

# Extracting image in train-valid-test subfolders
for i in {0..69};
do
    if (( i < 10 ))
    then
        subarchive="0${i}"
    else
        subarchive="${i}"
    fi

    if ((i < 50))
    then 
        set_="train"
    elif ((i < 60))
    then
        set_="valid"
    else
        set_="test"
    fi
    7z e "thumbnails128x128.zip" -o"images/${set_}/${subarchive}000/" -ir!thumbnails128x128/${subarchive}*.png
done