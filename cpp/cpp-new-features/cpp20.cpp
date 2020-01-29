//
// Created by rostam on 29.01.20.
//

#include <iostream>

using namespace std;

extern "C" {

static int my_callback(int a) {
    return a + 1;
}

}

template <typename> struct fnptr;
template <typename Ret, typename... Args> struct fnptr<Ret(Args...)> {
    void *obj = nullptr;
    Ret (*wrap)(void*, Args...) = nullptr;

    fnptr() = default;
    fnptr(Ret target(Args...)) : fnptr([=] (Args... args) { return target(args...); }) {}
    template <typename T> fnptr(T&& fn) : obj(&fn), wrap([](void *fn, Args... args) { return (*(T *)fn)(args...); }) {}

    Ret operator()(Args... args) const { return wrap(obj, args...); }
};

//int main() {
//    cerr << my_callback(1);
//}

 void hello() { printf("hello\n"); }

 fnptr<void()> x;

 int main() {
    int q=1;
    x = hello;
    x();
    x = [&q] { q++; printf("again\n"); };
    x();

     cerr << my_callback(1);
 }

