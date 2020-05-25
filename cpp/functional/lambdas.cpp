//
// Created by rostam on 25.05.20.
//

#include <algorithm>
#include <functional>
#include <vector>
#include <iostream>
#include <variant>

template<class... Ts> struct overload : Ts... { using Ts::operator()...; };
template<class... Ts> overload(Ts...) -> overload<Ts...>;

int main() {
    using std::placeholders::_1;

    const std::vector<int> v { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const auto val = std::count_if(v.begin(), v.end(),
                                   std::bind(std::logical_and<bool>(),
                                             std::bind(std::greater<int>(),_1, 2),
                                             std::bind(std::less_equal<int>(),_1,6)));

    const auto val2 = std::count_if(v.begin(), v.end(), [](int v) { return v > 2 && v <= 6;});

    std::cout << val << " " << val2;

    std::vector<int> vec { 0, 5, 2, 9, 7, 6, 1, 3, 4, 8 };

    size_t compCounter = 0;
    std::sort(vec.begin(), vec.end(), [&compCounter](int a, int b) {
    ++compCounter;
    return a < b;
    });

    std::cout << "\nnumber of comparisons: " << compCounter << '\n';

    for (auto& v : vec)
        std::cout << v << ", ";

    std::cout << std::endl;
    std::variant<int, float, std::string> intFloatString { "Hello" };
    std::visit(overload  { [](const int& i) { std::cout << "int: " << i; },
                            [](const float& f) { std::cout << "float: " << f; },
                                [](const std::string& s) { std::cout << "string: " << s; }
                        },
                                intFloatString);
    return 0;
}