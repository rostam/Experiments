#include <iostream>
#include "mybinary.h"

int main() {
    unsigned const one = mybinary<1>::value;
    unsigned const three = mybinary<11>::value;
    unsigned const five = mybinary<101>::value;
    unsigned const seven = mybinary<111>::value;
    std::cout << one << std::endl;
    std::cout << three << std::endl;
    std::cout << five << std::endl;
    return 0;
}