#!/bin/sh

# Get the current Git branch
current_branch=$(git rev-parse --abbrev-ref HEAD)

# Check if the current branch is master
if [ "$current_branch" = "master" ]; then
echo "on $current_branch branch"
pdoc -o docs src/clev2er --no-include-undocumented --mermaid --logo "https://www.homepages.ucl.ac.uk/~ucasamu/mssl_logo.png" --docformat google 
retVal=$?
if [ $retVal -ne 0 ]; then
    echo "pdoc failed"
    exit 1
fi
git add docs

else
    echo "Not on master branch, skipping pdocs build."
fi

exit 0
