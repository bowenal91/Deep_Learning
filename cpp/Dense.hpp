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
        vector<int> init_layer(const& vector<int> data_shape);
        Tensor evaluate(const& Tensor input);
        void Train();
};
