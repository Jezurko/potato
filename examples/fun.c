#include <stddef.h>

int main() {
    int a = 5;
    int b = 6;
    int *c = &a;
    if (c != NULL) {
        *c = b;
    }
    return a;
}
