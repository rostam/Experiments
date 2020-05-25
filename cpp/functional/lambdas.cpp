//
// Created by rostam on 25.05.20.
//

#include <algorithm>
#include <functional>
#include <vector>
#include <iostream>

int main() {
    using std::placeholders::_1;

    const std::vector<int> v { 1, 2, 3, 4, 5, 6, 7, 8, 9 };
    const auto val = std::count_if(v.begin(), v.end(),
                                   std::bind(std::logical_and<bool>(),
                                             std::bind(std::greater<int>(),_1, 2),
                                             std::bind(std::less_equal<int>(),_1,6)));

    const auto val2 = std::count_if(v.begin(), v.end(), [](int v) { return v > 2 && v <= 6;});

    std::cout << val << " " << val2;
    return 0;
}