#! bash
for filename in ./*.yml; do
    name=`basename $filename`
    nohup pippin.sh -v --refresh $name > ./logs/$name.log &
    sleep 10m
done