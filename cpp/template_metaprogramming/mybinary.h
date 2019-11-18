//
// Created by rostam on 18.11.19.
//

#ifndef TEMPLATE_METAPROGRAMMING_MYBINARY_H
#define TEMPLATE_METAPROGRAMMING_MYBINARY_H

template <unsigned long N>
class mybinary {
public:
    static unsigned const value = mybinary<N/10>::value * 2 + N%10;

};

template<>
class mybinary<0> {
public:
    static unsigned const value = 0;
};

#endif //TEMPLATE_METAPROGRAMMING_MYBINARY_H
