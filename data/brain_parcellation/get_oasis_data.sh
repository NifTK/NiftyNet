#!/usr/bin/env sh
# Download a demo volume extracted from OASIS dataset.

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"
wget -c https://www.dropbox.com/s/zpk0mom7aoawpyq/OASIS_DATA.tar.gz
tar -xvf OASIS_DATA.tar.gz
