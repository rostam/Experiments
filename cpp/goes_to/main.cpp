#include <cstudio>

int main() {

int x = 10;
while (x --> 0) // "goes to" trick: --> is (x--) > 0, i.e. post-decrement followed by greater-than
{
  printf("%d ", x);
}

}
