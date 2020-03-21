#include <vector>

class Tensor {
    std::vector<int> shape;
    std::vector<double> vals;
    int size;
    int dimensions;
    int map_id(const std::vector<int>& ids);
    public:
        Tensor();
        Tensor(const std::vector<int>& input_shape);
        void set_value(const std::vector<int>& id, double val);
        double& get_value(const std::vector<int>& id);
        void compare_sizes (const Tensor& a) const;
        void resize(const std::vector<int>& input_shape);
        friend double operator*(const Tensor& a, const Tensor& b);
        friend Tensor operator+(const Tensor& a, const Tensor& b);
        friend Tensor operator-(const Tensor& a, const Tensor& b);
        friend Tensor operator^(const Tensor& a, const Tensor& b);
};


