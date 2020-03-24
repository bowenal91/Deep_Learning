#include <vector>
#include <functional>

class Activation {
    double ReLU(double input);
    double d_ReLU(double input);
    double d_Logistic(double input);
    double Logistic(double input); 
    public:
        Activation();
        void set_activation(std::string& name);
        std::function<double(double)> evaluate;
        std::function<double(double)> derivative;
};
