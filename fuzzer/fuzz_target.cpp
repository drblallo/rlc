#include "stdio.h"

// This is implemented by RLC.
extern "C" void RLC_simulate();

// This is called by RLC_simulate.
// It's a stand-in for the sophisticated stuff we'd like to do in C.
extern "C" void printint64_t_(const __int64_t *message) {
    printf("Message: %ld \n", *message);
}

int main() {
    RLC_simulate();
    return 0;
}