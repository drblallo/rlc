#include <cmath>
#include <cstdlib>
#include "stdio.h"

// This is implemented by RLC.
extern "C" void RLC_Fuzzer_simulate();

/* --------------- These are is called by RLC_Fuzzer_simulate. -------------- */
// It's a stand-in for the sophisticated stuff we'd like to do in C.
extern "C" void RLC_Fuzzer_printvoid_int64_t_(const __int64_t *message) {
    printf("Message: %ld \n", *message);
}

extern "C" void RLC_Fuzzer_getInputint64_t_int64_t_(__int64_t *result, const __int64_t *max) {
    *result = rand() % *max;
}

extern "C" void RLC_Fuzzer_pickArgumentint64_t_int64_t_(__int64_t *result, const __int64_t *size) {
    *result = abs(rand() % 10); // TODO this is temporary.
}


int main() {
    RLC_Fuzzer_simulate();
    return 0;
}