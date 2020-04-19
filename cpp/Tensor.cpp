#include "Tensor.hpp"
#include <assert.h>

using namespace std;

Tensor::Tensor() {
    shape.clear();
    vals.clear();
    rank = 0;
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
    rank = input_shape.size();
    for (int i=0;i<rank;i++) {
        shape.push_back(input_shape[i]);
        local_size *= input_shape[i];
    }
    size = local_size;
    vals.resize(size,0);
}

int Tensor::get_size() {
    return size;
}

int Tensor::get_rank() {
    return rank;
}

vector<int> Tensor::get_shape() {
    return shape;
}

int Tensor::map_id(const vector<int>& id) {
    int prefactor = 1;
    int output = 0;
    for (int i=0; i<rank; i++) {
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
    vals[id] = val;
    return;
}

double Tensor::get_value(const vector<int>& id) {
    int i = map_id(id);
    return vals[i];
}

double Tensor::get_value(const int id) {
    return vals[id];
}

void Tensor::compare_sizes (const Tensor& a) const {
    assert(rank == a.rank);
    assert(size == a.size);
    for (int i=0;i<rank;i++) {
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

double Tensor::iterate_indices(int d, vector<int>& size, vector<int>& start_a, vector<int>& start_b, vector<int>& a_id, vector<int>& b_id, Tensor& b) {
    //d is dimension. ids is the value to be added to starts_a and starts_b to get their index. current_sum is the current_sum up to that point
    double sum = 0.0;
    for (int i=0;i<=size[d];i++) {
        a_id[d] = start_a[d]+i;
        b_id[d] = start_b[d]+i;
        if (d==size.size()-1) {
            sum += get_value(a_id) * b.get_value(b_id);
        } else {
            sum += iterate_indices(d+1,size,start_a,start_b,a_id,b_id,b);
        }
    }
    return sum;
}

double Tensor::subset_mult(const vector<int>& a_ids, const vector<int>& b_ids, Tensor& b) {
    //Perform dot product using only a subset of the whole tensor. 
    vector<int> sizes,start_a,start_b,curr_a,curr_b;
    assert(a_ids.size() == b_ids.size() && rank == b.rank && 2*rank == a_ids.size());
    for (int i=0;i<a_ids.size();i+=2) {
        assert(a_ids[i+1]-a_ids[i] == b_ids[i+1]-b_ids[i]);
        sizes.push_back(a_ids[i+1]-a_ids[i]);
        start_a.push_back(a_ids[i]);
        start_b.push_back(b_ids[i]);
        curr_a.push_back(0.0);
        curr_b.push_back(0.0);
    }

    //If all this is true then we can sweep through each element 
    return iterate_indices(0,sizes,start_a,start_b,curr_a,curr_b,b);
    

}

void Tensor::iterate_collapse(Tensor &output, vector<int> &ids, vector<int> &collapsed_ids, int axis, int d) {
    for (int i=0;i<shape[d];i++) {
        ids[d] = i;
        if (d != axis) {
            collapsed_ids = i;
        } else {
            collapsed_ids = 0;
        }

        if (d==rank-1) {
            output.set_value(collapsed_ids, output.get_value(collapsed_ids) + get_value(ids)); 
        } else {
            iterate_collapse(output, ids, collapsed_ids, axis, d+1);
        }
    }

    return;

}

Tensor Tensor::collapse(int axis) {
    //Produces sum of all elements along axis
    vector<int> newshape = shape;
    shape[axis] = 1;
    Tensor collapsed_tensor(newshape);
    vector<int> ids, collapsed_ids;
    for (int i=0;i<rank;i++) {
        ids.push_back(0);
        collapsed_ids.push_back(0);
    }
    iterate_collapse(collapsed_tensor, ids, collapsed_ids, axis, 0);
    return collapsed_tensor;
}
