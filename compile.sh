g++ -O3 -Wall -shared -std=c++11 -fPIC `python3 -m pybind11 --includes` test.cpp -o example`python3-config --extension-suffix` 
