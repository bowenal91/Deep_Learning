#include "Tensor.hpp"
#include <assert.h>

using namespace std;

Tensor::Tensor() {
    shape.clear();
    vals.clear();
    dimensions = 0;
    size = 0;
}

Tensor::Tensor(const vector<int>& input_shape) {
    resize(input_shape);
    return;
}

void Tensor::resize(const vector<int>& input_shape) {
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
}

int Tensor::map_id(const vector<int>& id) {
    int prefactor = 1;
    int output = 0;
    for (int i=0; i<dimensions; i++) {
        output += id[i]*prefactor;
        prefactor *= shape[i];
    }
    return output;
}

void Tensor::set_value(const vector<int>& id, double val) {
    int i = map_id(id);
    vals[i] = val;
    return;
}

void Tensor::set_value(const int id, double val) {
    vals[i] = val;
    return;
}

double Tensor::get_value(const vector<int>& id) {
    int i = map_id(id);
    return vals[i];
}

double Tensor::get_value(const int id) {
    return vals[i];
}

void Tensor::compare_sizes (const Tensor& a) const {
    assert(dimensions == a.dimensions);
    assert(size == a.size);
    for (int i=0;i<dimensions;i++) {
        assert(shape[i] == a.shape[i]);
    }
    return;
}

double operator*(const Tensor& a, const Tensor &b) {
    a.compare_sizes(b);
    double output = 0.0;
    for (int i=0;i<a.size;i++) {
        output += a.vals[i]*b.vals[i]; 
    }
    return output;
}

Tensor operator+(const Tensor& a, const Tensor& b) {
    a.compare_sizes(b);
    Tensor out(a.shape);
    for (int i=0;i<a.size;i++) {
        out.vals[i] = a.vals[i]+b.vals[i];
    }
    return out;
}

Tensor operator-(const Tensor& a, const Tensor& b) {
    a.compare_sizes(b);
    Tensor out(a.shape);
    for (int i=0;i<a.size;i++) {
        out.vals[i] = a.vals[i]-b.vals[i];
    }
    return out;
}

Tensor operator^(const Tensor& a, const Tensor& b) {
    a.compare_sizes(b);
    Tensor out(a.shape);
    for (int i=0;i<a.size;i++) {
        out.vals[i] = a.vals[i]*b.vals[i];
    }
    return out;
}
