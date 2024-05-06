int main(int argc, char **argv){
    int b = 5;
    int *c = &b;
    if (argc) {
        c = &argc;
    }
    return *c;
}
