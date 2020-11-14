set -e

#project=5b0dad8041711001e958b519
project=5d64733db29ac960ca2e797f
datatype="neuro/tractprofile"

#cache the list of datasets that we could download
if [ ! -f all.json ]; then
    bl dataset query --limit 10000 --project $project --datatype $datatype --json > all.json
fi

#enumerate subjects
for subject in $(jq -r '.[].meta.subject' all.json | sort -u)
do
    echo "downloading subject:$subject ---------------"
    mkdir -p 2019_tractprofile/$subject/
    ids=$(jq -r '.[] | select(.meta.subject == '\"$subject\"') | ._id' all.json)
    for id in $ids
    do
        echo $id $tags
        tags=$(jq -r '.[] | select(._id=='\"$id\"') | .tags | join(".")' all.json)
        outdir=2019_tractprofile/$subject/$tags
        if [ ! -d $outdir ]; then
            bl dataset download $id --directory $outdir
        fi
    done
done