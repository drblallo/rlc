#include <cmath>
#include <cstdlib>
#include <exception>
#include "stdio.h"

// This is implemented by RLC.
extern "C" void RLC_Fuzzer_simulate();

/* --------------- These are called by RLC_Fuzzer_simulate. -------------- */
// It's a stand-in for the sophisticated stuff we'd like to do in C.
extern "C" void RLC_Fuzzer_printvoid_int64_t_(const __int64_t *message) {
    printf("Message: %ld \n", *message);
}

extern "C" void RLC_Fuzzer_getInputint64_t_int64_t_(__int64_t *result, const __int64_t *max) {
    printf("Generating input in range [0, %ld)\n", *max);
    *result = rand() % *max;
}

extern "C" void RLC_Fuzzer_pickArgumentint64_t_int64_t_(__int64_t *result, const __int64_t *size) {
    *result = abs(rand() % 10); // TODO this is temporary.
}


extern "C" void RLC_Fuzzer_skipInputvoid_() {
    printf("skipping the current fuzz input!\n");
}


int main() {
    RLC_Fuzzer_simulate();
    return 0;
}