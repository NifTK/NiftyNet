wget -N https://www.dropbox.com/s/y7mdh4m9ptkibax/example_volumes.tar.gz
tar -xzvf example_volumes.tar.gz
wget -N https://www.dropbox.com/s/94wa4fl8f8k3aie/testing_data.tar.gz
tar -xzvf testing_data.tar.gz

python -m unittest discover -s "tests" -p "*_test.py"
