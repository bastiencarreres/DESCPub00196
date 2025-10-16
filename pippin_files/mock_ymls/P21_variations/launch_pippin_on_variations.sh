#! bash
for filename in ./*.yml; do
    name=`basename $filename`
    nohup pippin.sh -v $name > ./logs/$name.log &
    sleep 10m
done