int main(int argc, char **argv){
    int a = argc;
    int b = 5;
    int *c;
    if (a) {
        c = &a;
    } else {
        c = &b;
    }
    return *c;
}
