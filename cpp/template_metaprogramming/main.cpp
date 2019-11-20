#include <iostream>
#include <memory>
#include "mybinary.h"

template <typename T>
auto get_value(T t) {
    if constexpr (std::is_pointer_v<T>)
        return *t;
    else
        return t;
}

template<int  N>
constexpr int fibonacci() {return fibonacci<N-1>() + fibonacci<N-2>(); }
template<>
constexpr int fibonacci<1>() { return 1; }
template<>
constexpr int fibonacci<0>() { return 0; }

template<int N>
constexpr int fibonacci2()
{
    if constexpr (N>=2)
        return fibonacci2<N-1>() + fibonacci2<N-2>();
    else
        return N;
}


int main() {
    unsigned const one = mybinary<1>::value;
    unsigned const three = mybinary<11>::value;
    unsigned const five = mybinary<101>::value;
    unsigned const seven = mybinary<111>::value;
    std::cout << one << std::endl;
    std::cout << three << std::endl;
    std::cout << five << std::endl;
    std::cout << seven << std::endl;

    auto pi = std::make_unique<int>(9);
    int i = 9;

    std::cout << get_value(pi.get()) << "\n";
    std::cout << get_value(i) << "\n";

    std::cout << fibonacci<10>() << std::endl;
    std::cout << fibonacci2<10>() << std::endl;
    return 0;
}