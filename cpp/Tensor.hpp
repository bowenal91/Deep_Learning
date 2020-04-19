#ifndef TENSOR_H
#define TENSOR_H

#include <vector>

class Tensor {
    std::vector<double> vals;
    int map_id(const std::vector<int>& ids);
    int size;
    int rank;
    std::vector<int> shape;
    public:
        int get_size();
        int get_rank();
        std::vector<int> get_shape();
        Tensor();
        Tensor(const std::vector<int>& input_shape);
        void set_value(const std::vector<int>& id, double val);
        double get_value(const std::vector<int>& id);
        double get_value(const int id);
        void set_value(const int id, double val);
        void compare_sizes (const Tensor& a) const;
        void resize(const std::vector<int>& input_shape);
        friend double operator*(const Tensor& a, const Tensor& b);
        friend Tensor operator+(const Tensor& a, const Tensor& b);
        friend Tensor operator-(const Tensor& a, const Tensor& b);
        friend Tensor operator^(const Tensor& a, const Tensor& b);
        double subset_mult(const std::vector<int>& a_ids, const std::vector<int>& b_ids, Tensor& b);
        double iterate_indices(int d, std::vector<int>& size, std::vector<int>& start_a, std::vector<int>& start_b, std::vector<int>& a_id, std::vector<int>& b_id, Tensor& b); 
        Tensor collapse(int axis);
        void iterate_collapse(Tensor &output, vector<int> &ids, vector<int> &collapsed_ids, int axis, int d);
};

#endif 
