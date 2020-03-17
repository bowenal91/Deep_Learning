#include <vector>

class Tensor {
    std::vector<int> shape;
    std::vector<double> vals;
    int size;
    int dimensions;
    int map_id(std::vector<int> ids);
    public:
        Tensor(std::vector<int> input_shape);
        void set_value(std::vector<int> id, double val);
        double get_value(std::vector<int> id);
};


