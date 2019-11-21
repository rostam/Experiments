//
// Created by rostam on 20.11.19.
//

#include <cmath>
#include <iostream>

template<int N>
constexpr int fibonacci2()
{
    if constexpr (N>=2)
        return fibonacci2<N-1>() + fibonacci2<N-2>();
    else
        return N;
}

class psc_f {
private:
    double alpha;

public:
    explicit psc_f(double alpha_) : alpha(alpha_) {};

    double operator()(double x) const {
        return sin(alpha * x) + cos(x);
    }
};

template <typename F, typename T>
class derivative {
private:
    const F &f;
    T h;

public:
    derivative(const F &f, T t) : f(f), h(t) {}

    T operator()(const T &x) const {
        return (f(x + h) - f(x)) / h;
    }
};

int main() {
    fibonacci2<10>();

    psc_f psc_o(1.0);
    derivative<psc_f, double> d_psc_o(psc_f, 0.001);
    std::cout << "derivative of sin(0) + cos(0) is " << d_psc_o(0.0) << std::endl;
}
