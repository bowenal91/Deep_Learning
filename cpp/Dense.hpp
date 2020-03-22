#include <vector>
#include "Tensor.cpp"

class Dense {
    vector<int> input_shape;
    vector<int> output_shape;
    vector<Tensor> weights;
    vector<double> biases;
    Activation act;
    int numNeurons;
    public:
        Dense(int num_neurons);
        void init_layer(const vector<int>& data_shape);
        Tensor evaluate(const Tensor& input);
        Tensor back_propagate(const Tensor& input);
};
