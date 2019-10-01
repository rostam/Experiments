//
// Created by rostam on 30.09.19.
//
#include <variant>
#include <cmath>
#include <tuple>
// aX^2 + bX + c
using TRoots = std::variant<std::monostate,
        double,
        std::pair<double, double>>;

const double EPSILON = 0.0001;

TRoots FindRoots(double a, double b, double c)
{
    const auto delta = b*b-4.0*a*c;

    if (delta > EPSILON)
    {
        auto p = sqrt(delta);
        double x1 = (-b + p)/(2*a);
        double x2 = (-b - p)/(2*a);
        return std::pair(x1, x2);
    }
    else if (delta < -EPSILON)
        return std::monostate();

    return -b/(2*a);
}



#include <variant>
#include <iostream>

struct S
{
    S(int i) : i(i) {}
    int i;
};

//inline variables
class MyClass {static inline const std::string s_val = "Hello";};


int main() {

    // Without the monostate type this declaration will fail.
    // This is because S is not default-constructible.

    std::variant<std::monostate, S> var;

    // var.index() is now 0 - the first element
    // std::get<S> will throw! We need to assign a value

    var = 12;

    std::cout << std::get<S>(var).i << '\n';

    double res = std::get<std::pair<double, double>>(FindRoots(0.4, 0.9, 0.4)).first;
    std::cout << res << std::endl;


    auto [ a, b, c ] = std::tuple<int,double,bool>(1,1.2,true);
    std::cout << a << " " << b << " " << c << std::endl;

    std::pair mypair{42,1.5};
    std::cout << mypair.first, mypair.second;

}