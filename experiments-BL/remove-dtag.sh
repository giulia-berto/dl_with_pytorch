#!/bin/bash

# login (you only need to run this once every 30 days) 
#bl login --ttl 30

# id of the project "DL with Pytorch"
projID=5fa925f3afcdcce87fc3facc

#cache the list of datasets that we could download
if [ ! -f all.json ]; then
    bl dataset query --limit 10000 --project $projID --datatype_tag warped --json > all.json
fi

#enumerate subjects
n_sub=0
n_obj=0
for subject in $(jq -r '.[].meta.subject' all.json | sort -u)
do
    echo "Removing tag for subject:$subject"
    ids=$(jq -r '.[] | select(.meta.subject == '\"$subject\"') | ._id' all.json)

    for id in $ids
    do
    	bl dataset update --id $id --remove_dtag warped
    	((n_obj+=1))
    done	

    echo -e "Number of modified objects for subject $subject:$n_obj\n"
    ((n_sub+=1))
    
done
echo "Total number of subjects:$n_sub"
