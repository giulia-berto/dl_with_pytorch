#!/bin/bash

# login (you only need to run this once every 30 days) 
#bl login --ttl 30

# id of the project "DL with Pytorch"
projID=5fa925f3afcdcce87fc3facc

out_dir=../data/HCP-anat/images
json_fname=dataset38fa.json
datatype=neuro/tensor
tag=t1_warping_mni152

echo "Downloading FAs..."

#cache the list of datasets that we could download
if [ ! -f $json_fname ]; then
    bl dataset query --limit 10000 -p $projID --datatype $datatype --tag $tag --json > $json_fname
fi

n_sub=0

for subjID in $(jq -r '.[].meta.subject' $json_fname | sort -u); do

	echo "Subject:$subjID"
    ids=$(jq -r '.[] | select(.meta.subject == '\"$subjID\"') | ._id' $json_fname)
    n_obj=0

    for id in $ids
    do
        echo "downloading object:$id"
        #bl dataset download $id
        #out_fname=$out_dir/sub-${subjID}_space-mni152_fa.nii.gz
        #mv $id/fa.nii.gz $out_fname
        #rm -r $id

        ((n_obj+=1))
    done

    echo -e "Number of downloaded objects for subject $subject:$n_obj\n"
    ((n_sub+=1))

done
echo "Total number of subjects:$n_sub"