#include <vector>

class Tensor {
    std::vector<double> vals;
    int map_id(const std::vector<int>& ids);
    public:
        int size;
        int dimensions;
        std::vector<int> shape;
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
};


