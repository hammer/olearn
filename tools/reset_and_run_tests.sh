#! /bin/bash
./tools/oasis_cleanup.sh
./tools/oasis_gen.sh
./configure --enable-tests
make
./test_olearn.native
