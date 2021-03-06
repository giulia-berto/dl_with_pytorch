#!/bin/bash

# login (you only need to run this once every 30 days) 
#bl login --ttl 30

# id of the project "DL with Pytorch"
projID=5fa925f3afcdcce87fc3facc

out_dir=../data/HCP-anat/images
datatype=neuro/anat/t1w
datatype_tag=!defaced

echo "Downloading T1s not defaced..."

#cache the list of datasets that we could download
if [ ! -f all.json ]; then
    echo "Retrieving IDs..."
    bl dataset query --limit 10000 -p $projID --datatype $datatype --datatype_tag $datatype_tag --json > all.json
fi

n_sub=0

for subjID in $(jq -r '.[].meta.subject' all.json | sort -u); do

	echo "Subject:$subjID"
    ids=$(jq -r '.[] | select(.meta.subject == '\"$subjID\"') | ._id' all.json)
    n_obj=0

    for id in $ids
    do
        echo "downloading object:$id"
        bl dataset download $id
        out_fname=$out_dir/sub-${subjID}_space-mni152_t1.nii.gz
        mv $id/t1.nii.gz $out_fname
        echo "$out_fname successfully saved."
        rm -r $id

        ((n_obj+=1))
    done

    echo -e "Number of downloaded objects for subject $subject:$n_obj\n"
    ((n_sub+=1))

done
rm all.json
echo "Total number of subjects:$n_sub"
