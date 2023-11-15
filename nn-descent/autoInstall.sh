#rm CMakeCache.txt
cmake -DCMAKE_BUILD_TYPE=release .
#cmake -DCMAKE_BUILD_TYPE=DEBUG .
make clean
make VERBOSE=1
#sudo make install
