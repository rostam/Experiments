//
// Created by rostam on 11.06.20.
//

#include <iostream>
using namespace std;

template<typename D, typename B>
class IsDerivedFromHelper
{
    class No { };
    class Yes { No no[3]; };

    static Yes Test( B* );
    static No Test( ... );
public:
    enum { Is = sizeof(Test(static_cast<D*>(0))) == sizeof(Yes) };

};


template <class C, class P>
bool IsDerivedFrom() {
    return IsDerivedFromHelper<C, P>::Is;
}

class A {

};

class B : public A {

};

class C {

};

int main() {
    cout << IsDerivedFrom<B,A>() << endl;
    cout << IsDerivedFrom<C,A>() << endl;
    cout << std::is_base_of<A,B>();
}
