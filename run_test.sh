wget -N https://www.dropbox.com/s/y7mdh4m9ptkibax/example_volumes.tar.gz
tar -xzvf example_volumes.tar.gz
#wget -N https://www.dropbox.com/s/94wa4fl8f8k3aie/testing_data.tar.gz
wget -N https://www.dropbox.com/s/p7b3t2c3mewtree/testing_data_v0_2.tar.gz
tar -xzvf testing_data_v0_2.tar.gz

#python -m unittest discover -s "tests" -p "*_test.py"

# run global config tests
# These need to be run separately because NiftyNetGlobalConfig is a singleton, AND
# its operations pertain to a global configuration file (~/.niftynet/config.ini).
GLOBAL_CONFIG_TEST_gcs=True python -m unittest tests.niftynet_global_config_test
GLOBAL_CONFIG_TEST_necfc=True python -m unittest tests.niftynet_global_config_test
GLOBAL_CONFIG_TEST_ecfl=True python -m unittest tests.niftynet_global_config_test
GLOBAL_CONFIG_TEST_icfbu=True python -m unittest tests.niftynet_global_config_test
GLOBAL_CONFIG_TEST_nenhc=True python -m unittest tests.niftynet_global_config_test
GLOBAL_CONFIG_TEST_enhnt=True python -m unittest tests.niftynet_global_config_test

