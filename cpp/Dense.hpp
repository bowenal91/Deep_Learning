#include <vector>
#include "Tensor.cpp"
#include "Activation.cpp"

class Dense {
    vector<int> input_shape;
    vector<int> output_shape;
    vector<Tensor> weights;
    vector<double> biases;
    Activation *act;
    int numNeurons;
    int batch_size;
    void set_activation(std::string& name);
    public:
        Dense(int num_neurons, std::string activation_name="None");
        void init_layer(const vector<int>& data_shape);
        void randomize_weights(double max);
        Tensor evaluate(const Tensor& input);
        Tensor back_propagate(const Tensor& input);
};
