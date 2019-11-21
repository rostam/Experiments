//
// Created by rostam on 20.11.19.
//

template<int N>
constexpr int fibonacci2()
{
    if constexpr (N>=2)
        return fibonacci2<N-1>() + fibonacci2<N-2>();
    else
        return N;
}

int main() {
    fibonacci2<10>();
}
