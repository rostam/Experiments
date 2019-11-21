//
// Created by rostam on 20.11.19.
//

template<int N>
constexpr int fibonacci2()
{
    if constexpr (N>=2)
        return fibonacci2<N-1>() + fibonacci2<N-2>();
    else
        return N;
}

class psc_f {
public:
    psc_f(double alpha) : alpha(alpha)

    double operator()(double x) const {
        return sin(alpha * x) + cos(x);
    }

private:
    double alpha;
};

template <typename F, typename T>
class derivative {
public:
    derivative(const F &f, const T &t) : f(f), t(t) {}

    T operator()(const T &x) const {
        return (f(x + h) - f(x)) / h;
    }

private:
    const F &f;
    T t;
};

int main() {
    fibonacci2<10>();
}
