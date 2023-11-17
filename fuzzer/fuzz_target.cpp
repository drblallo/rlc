#include <cstdint>
#include <iostream>

// This is implemented by RLC.
extern "C" void RLC_simulate();

// This is called by RLC_simulate.
// It's a stand-in for the sophisticated stuff we'd like to do in C.
extern "C" void printint64_t_(const int64_t *message) {
    std::cout << "Message: " << *message << "\n";
}

int main() {
    RLC_simulate();
    return 0;
}