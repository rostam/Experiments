//
// Created by rostam on 29.01.20.
//

#include <iostream>

using namespace std;

extern "C" {

static int my_callback(int a)
{
    return a + 1;
}

}

int main() {
    cerr << my_callback(1);
}

