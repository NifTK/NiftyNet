wget -N https://www.dropbox.com/s/y7mdh4m9ptkibax/example_volumes.tar.gz
tar -xzvf example_volumes.tar.gz
#wget -N https://www.dropbox.com/s/94wa4fl8f8k3aie/testing_data.tar.gz
wget -N https://www.dropbox.com/s/p7b3t2c3mewtree/testing_data_v0_2.tar.gz
tar -xzvf testing_data_v0_2.tar.gz

python -m unittest discover -s "tests" -p "*_test.py"
