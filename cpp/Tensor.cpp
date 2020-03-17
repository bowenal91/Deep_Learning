#include "Tensor.hpp"

using namespace std;

Tensor::Tensor(vector<int> input_shape) {
    int local_size = 1;
    shape.clear();
    vals.clear();
    dimensions = input_shape.size();
    for (int i=0;i<dimensions;i++) {
        shape.push_back(input_shape[i]);
        local_size *= input_shape[i];
    }
    size = local_size;
    vals.resize(size,0);
    return;
}

int Tensor::map_id(vector<int> id) {
    int prefactor = 1;
    int output = 0;
    for (int i=0; i<dimensions; i++) {
        output += id[i]*prefactor;
        prefactor *= shape[i];
    }
    return output;
}

void Tensor::set_value(vector<int> id, double val) {
    int i = map_id(id);
    vals[i] = val;
    return;
}

double Tensor::get_value(vector<int> id) {
    int i = map_id(id);
    return vals[i];
}
