#!/bin/bash

if [[ $# -ne 1 ]]; then
    echo "Usage: ${0} <path-to-download-folder>"
    exit 1
fi

if [[ ! -d "$1" ]]; then
    echo -e "The download folder \"${1}\" does not exist!\nCreate it and re-launch the script"
    exit 2
fi

cd "$1" || exit 3

zenodo-download -t xxx -w 20 -d . 7842598  # Download Event-Human3.6m
zip -F h36m.zip --out h36m_full.zip  # Assemble the full zip from the split archive
unzip h36m_full.zip  # Unzip the archive