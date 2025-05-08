#! bash
for filename in ./*.yml; do
    name=`basename $filename`
    nohup pippin.sh -s 2 --refresh $name > ./logs/$name.log &
    sleep 10m
done