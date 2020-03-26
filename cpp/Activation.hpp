#include <vector>
#include <functional>

class Activation {
    public:
        Activation();
        virtual double evaluate(double x);
        virtual double deriv(double x);
};

class ReLU : public Activation {
    public: 
        ReLU();
        double evaluate(double x);
        double deriv(double x);
};

class Logistic : public Activation {
    public:
        Logistic();
        double evaluate();
        double deriv();
};
