#!/usr/bin/env sh
# Download model and unzip

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

wget -c https://www.dropbox.com/s/bl6jwjheq1cxn2b/highres3dnet.tar.gz
tar -xvf highres3dnet.tar.gz
