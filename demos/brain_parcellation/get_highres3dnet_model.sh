#!/usr/bin/env sh
# Download model and unzip

DIR="$( cd "$(dirname "$0")" ; pwd -P )"
cd "$DIR"

wget -c https://www.dropbox.com/s/t3g8bpm7h2cztc8/highres3dnet.tar.gz
tar -xvf highres3dnet.tar.gz
