mkdir trainset
curl https://f.janyeh.com/ok/real_clear_14427.zip -o trainset/real_clear_14427.zip
curl https://f.janyeh.com/ok/real_hazy_13990.zip -o trainset/real_hazy_13990.zip
unzip trainset/real_hazy_13990.zip -d trainset/
unzip trainset/real_clear_14427.zip -d trainset/
mkdir testdataset/
mkdir testdataset/test_new
mkdir testdataset/test_new/hazy_new
curl https://f.janyeh.com/ok/240514_162205_l.zip -o trainset/SeaImages.zip
unzip trainset/SeaImages.zip -d trainset/SeaImages
mkdir trainset/trainA_new
mkdir trainset/trainB_new
mkdir trainset/trainB_newsize_128
cp trainset/SeaImages/l/clear1/*.* trainset/trainA_new
cp trainset/SeaImages/l/haze1/*.* trainset/trainB_new
cp trainset/SeaImages/l/haze1/*.* trainset/trainB_newsize_128/
mkdir testdataset/outdoor
mkdir testdataset/outdoor/hazy
cp trainset/trainB_new/100*.* testdataset/outdoor/hazy
mkdir testdataset/outdoor/gt
cp trainset/trainA_new/100*.* testdataset/outdoor/gt/
