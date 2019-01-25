#include <iostream>
#include <chrono>

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

template<typename... arguments>
auto SumWithOne(arguments... args) {
    return (1+ ... + args);
}

int sumOfAPair(std::pair<int, int> p) {
    return p.first + p.second;
}

namespace A::B::C { std::string test = "This is a string defined in a nested namespace"; }

template<auto value>
void f() {}

auto formula(int a) {
    return a * 2 + 0.1;
}

auto anySum = [](auto a, auto b) { return a + b; };

auto timer = [val = std::chrono::system_clock::now()] { return std::chrono::system_clock::now() - val; };

template<typename T>
constexpr T pi = T(3.1415926535897932385);

constexpr int square(int x) {
    return x * x;
}


struct Complex {
    constexpr Complex(double r, double i) : re(r), im(i) { }
    constexpr double real() { return re; }
    constexpr double imag() { return im; }

private:
    double re;
    double im;
};

constexpr Complex I(0, 1);


int main() {
    std::cout << "8th element of Fibonacci generated by templates at compile time: " << Fibonacci<8>::value
              << std::endl;
    auto a = {1, 2, 3};
    std::cout << "The first elemetn of initizer list: " << *std::begin(a);
    // The following should work in CPP 17
    //auto b = {1};
    //std::cout << b;

    auto test = SumWithOne(1, 2, 3, 4);
    std::cout << "The results of compact variadic template usage: " << test << std::endl;

    std::cout << A::B::C::test << std::endl;

    std::cout << "Template argument deduction for class templates: " << sumOfAPair(std::pair(3, 4)) << std::endl;

    f<10>();

    std::cout << "Automatic return type deduction for functions:" << formula(10) << std::endl;

    std::cout << "Generic lambdas:" << anySum(1.5,2) << std::endl;

    std::cout << "Lambda capturing initialization"; timer();


    //Rvalue references
    int x = 0;
    int& x1 = x;
//    int&& xr = x; compiler error
    int&& xr2 = 0;


    //Lambda expressions
    int x = 1;

    auto getX = [=] { return x; };
    getX(); // == 1

    auto addX = [=](int y) { return x + y; };
    addX(1); // == 2

    auto getXRef = [&]() -> int& { return x; };
    getXRef(); // int& to `x`


    //decltype
    int a = 1; // `a` is declared as type `int`
    decltype(a) b = a; // `decltype(a)` is `int`
    const int& c = a; // `c` is declared as type `const int&`
    decltype(c) d = a; // `decltype(c)` is `const int&`
    decltype(123) e = 123; // `decltype(123)` is `int`
    int&& f = 1; // `f` is declared as type `int&&`
    decltype(f) g = 1; // `decltype(f) is `int&&`
    decltype((a)) h = g; // `decltype((a))` is int&

    // Specifying underlying type as `unsigned int`
    enum class Color : unsigned int { Red = 0xff0000, Green = 0xff00, Blue = 0xff };
// `Red`/`Green` in `Alert` don't conflict with `Color`
    enum class Alert : bool { Red, Green };
    Color cc = Color::Red;

    const int xxx = 123;
    constexpr const int& y = x; // error -- constexpr variable `y` must be initialized by a constant expression


    int square2(int x) {
        return x * x;
    }

    int aa = square(2);  // mov DWORD PTR [rbp-4], 4

//    int b = square2(2); // mov edi, 2
// call square2(int)
// mov DWORD PTR [rbp-8], eax

    return 0;
}

// `noreturn` attribute indicates `f` doesn't return.
[[ noreturn ]] void f() {
    throw "error";
}