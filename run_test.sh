wget -q https://www.dropbox.com/s/lioecnpv82r5n6e/example_volumes_v0_2.tar.gz
tar -xzvf example_volumes_v0_2.tar.gz
rm example_volumes_v0_2.tar.gz
wget -N https://www.dropbox.com/s/p7b3t2c3mewtree/testing_data_v0_2.tar.gz
tar -xzvf testing_data_v0_2.tar.gz
rm testing_data_v0_2.tar.gz
wget -N https://www.dropbox.com/s/gt0hm6o61rlsfcc/csv_data.tar.gz
tar -C data -xzvf csv_data.tar.gz
rm csv_data.tar.gz

python -m unittest discover -s "tests" -p "*_test.py"

## run global config tests
## These need to be run separately because NiftyNetGlobalConfig is a singleton, AND
## its operations pertain to a global configuration file (~/.niftynet/config.ini).
#GLOBAL_CONFIG_TEST_gcs=True python -m unittest tests.niftynet_global_config_test
#GLOBAL_CONFIG_TEST_necfc=True python -m unittest tests.niftynet_global_config_test
#GLOBAL_CONFIG_TEST_ecfl=True python -m unittest tests.niftynet_global_config_test
#GLOBAL_CONFIG_TEST_icfbu=True python -m unittest tests.niftynet_global_config_test
#GLOBAL_CONFIG_TEST_nenhc=True python -m unittest tests.niftynet_global_config_test
#GLOBAL_CONFIG_TEST_enhnt=True python -m unittest tests.niftynet_global_config_test

