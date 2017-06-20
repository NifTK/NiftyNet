#!/usr/bin/env sh
# Download IXI dataset and unzip

cd downloaded
wget -c http://biomedic.doc.ic.ac.uk/brain-development/downloads/IXI/IXI-T1.tar
mkdir IXI-T1
(cd IXI-T1; tar -xvf ../IXI-T1.tar)
