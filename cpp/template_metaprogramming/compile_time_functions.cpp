//
// Created by rostam on 22.11.19.
//

constexpr long fibonacci (long n) {
    return n <= 2 ? 1 : fibonacci(n - 1) + fibonacci(n - 2);
}

template <typename T>
constexpr T square (T x) {
    return x*x;
}

