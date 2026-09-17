#define GGML_COMMON_IMPL
#include "ggml-common.h"
#include <stdio.h>
int main(void) {
    printf("grid0=%llx grid1=%llx grid2=%llx\n",
        (unsigned long long)iq2xxs_grid[0], (unsigned long long)iq2xxs_grid[1],
        (unsigned long long)iq2xxs_grid[2]);
    printf("ksigns=%llx\n", (unsigned long long)keven_signs_q2xs[0]);
    return 0;
}
