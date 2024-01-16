#include <cmath>
#include <cstdlib>
#include "stdio.h"
#include <math.h>
#include <algorithm>

// This is implemented by RLC.
extern "C" void RLC_Fuzzer_simulate();


int byte_offset;
int bit_offset;
int fuzz_input_length;
const char *fuzz_input;

int64_t consume_bits(const char *Data, int num_bits, int &byte_offset, int &bit_offset) {
    int64_t result = 0;
    int remaining_bits = num_bits;

    while (true) {
        int to_consume_from_current_byte = std::min(remaining_bits, 8 - bit_offset);
        int shift_count = (8 - bit_offset - remaining_bits);
        int mask = ((1u << to_consume_from_current_byte) - 1) << shift_count;
        int data = (*(Data + byte_offset) & mask) >> shift_count;
        result = (result << to_consume_from_current_byte) | data;

        if(remaining_bits >= (8 - bit_offset)) {
            byte_offset ++;
            remaining_bits -= (8 - bit_offset);
            bit_offset = 0;
        } else {
            bit_offset = bit_offset + remaining_bits;
            return result;
        }
    }
}

// TODO this is not completely uniform since the number of possible inputs is not a power of two, think about whether or not that's a problem.
extern "C" void RLC_Fuzzer_getInputint64_t_int64_t_(__int64_t *result, const __int64_t *max) {
    printf("Generating input in range [0, %ld)\n", *max);
    int num_bits = ceil(log2(*max));
    *result = consume_bits(fuzz_input, num_bits, byte_offset, bit_offset) % *max;
}

extern "C" void RLC_Fuzzer_pickArgumentint64_t_int64_t_int64_t_(__int64_t *result, const __int64_t *min, __int64_t *max) {
    printf("Picking an integer argument in range [%ld, %ld]\n", *min, *max);
    int num_bits = ceil(log2(*max - *min));
    *result = std::abs(consume_bits(fuzz_input, num_bits, byte_offset, bit_offset)) % (*max - *min) + *min;
}

extern "C" void RLC_Fuzzer_isInputLongEnoughbool_(__int8_t *result) {
    printf("fuzz_input_length: %d, byte_offfset: %d, bit_offset: %d\n", fuzz_input_length, byte_offset, bit_offset);
    *result = (fuzz_input_length - byte_offset) > 10; // TODO handle this better.
}

extern "C" void RLC_Fuzzer_printvoid_int64_t_(const __int64_t *message) {
    printf("Message: %ld \n", *message);
}

extern "C" void RLC_Fuzzer_skipInputvoid_() {
    printf("skipping the current fuzz input!\n");
}


extern "C" int LLVMFuzzerTestOneInput(const char *Data, size_t Size) {
    byte_offset = 0;
    bit_offset = 0;
    fuzz_input = Data;
    fuzz_input_length = Size;

    
    RLC_Fuzzer_simulate();
    return 0;
}