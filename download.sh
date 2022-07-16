#!/bin/sh
mkdir -p div2k
archives=(DIV2K_train_HR.zip DIV2K_valid_HR.zip DIV2K_train_LR_bicubic_X4.zip DIV2K_valid_LR_bicubic_X4.zip)
for archive in ${archives[@]}; do
    echo "Downloading $archive"
    wget http://data.vision.ee.ethz.ch/cvl/DIV2K/${archive}
    unzip ${archive} -d div2k
    rm ${archive}
done
echo "😊 Dataset downloaded to ./div2k/"
