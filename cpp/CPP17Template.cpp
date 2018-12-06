#include <iostream>

template<unsigned n>
struct Fibonacci {
    static const unsigned value = Fibonacci<n - 1>::value + Fibonacci<n - 2>::value;
};

template<>
struct Fibonacci<0> {
    static const unsigned value = 0;
};

template<>
struct Fibonacci<1> {
    static const unsigned value = 1;
};

int main() {
    std::cout <<  Fibonacci<8>::value;
    return 0;
}