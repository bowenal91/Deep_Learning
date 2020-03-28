#include <vector>
#include "Tensor.cpp"

class Loss {
    int batchSize;
    Loss_Function *evaluator;
    public:
        Loss(std::string& name);
        
};

class Loss_Function {
     
    public:
        Loss_Function();
        virtual void evaluate(double y, double y_hat);
        double evaluate_loss();
        virtual void add_deriv(double y, double y_hat);
        double evaluate_derivative();
}
