rm CMakeCache.txt
cmake -DCMAKE_BUILD_TYPE=release .
#cmake -DCMAKE_BUILD_TYPE=DEBUG -D CMAKE_CXXFLAGS='-DSPECIFIC_TIME' .
make clean
make
sudo make install
