#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <xmmintrin.h> 
#include <emmintrin.h>  
#include <pmmintrin.h> 
#include <tmmintrin.h> 
#include <smmintrin.h>
#include <nmmintrin.h> 
#include <ammintrin.h>
#include <immintrin.h>
#include <x86intrin.h>

#include "p2random.h"
#include "tree.h"

int main(int argc, char* argv[]) {
        // parsing arguments
        assert(argc > 3);
        size_t num_keys = strtoull(argv[1], NULL, 0);
        size_t num_probes = strtoull(argv[2], NULL, 0);
        size_t num_levels = (size_t) argc - 3;
        size_t* fanout = malloc(sizeof(size_t) * num_levels);
        assert(fanout != NULL);
        for (size_t i = 0; i < num_levels; ++i) {
                fanout[i] = strtoull(argv[i + 3], NULL, 0);
                assert(fanout[i] >= 2 && fanout[i] <= 17);
        }

        int useHardCoded = 0;
        if(num_levels == 3 && fanout[0] == 9 && fanout[1] == 5 && fanout[2] == 9)
                useHardCoded = 1;

        // building the tree index
        rand32_t* gen = rand32_init((uint32_t) time(NULL));
        assert(gen != NULL);
        int32_t* delimiter = generate_sorted_unique(num_keys, gen);
        assert(delimiter != NULL);
        Tree* tree = build_index(num_levels, fanout, num_keys, delimiter);
        free(delimiter);
        free(fanout);
        if (tree == NULL) {
                free(gen);
                exit(EXIT_FAILURE);
        }

        // generate probes
        int32_t* probe = generate(num_probes, gen);
        assert(probe != NULL);
        free(gen);
        uint32_t* result = malloc(sizeof(uint32_t) * num_probes);
        assert(result != NULL);

        uint32_t* result2 = malloc(sizeof(uint32_t) * num_probes);
        assert(result2 != NULL);

        uint32_t* result3 = malloc(sizeof(uint32_t) * num_probes);
        assert(result3 != NULL);

        // perform index probing (Phase 2)

        clock_t start1 = clock(), diff1;
        for (size_t i = 0; i < num_probes; ++i) {
                result[i] = probe_index(tree, probe[i]);
        }
        diff1 = clock() - start1;
        int time1 = diff1 * 1000 / CLOCKS_PER_SEC;

        clock_t start2 = clock(), diff2;
        //probing with SSE
        for(size_t i = 0; i < num_probes; ++i){
                result2[i] = probe_index_sse(tree, probe[i]);
        }
        diff2 = clock() - start2;
        int time2 = diff2 * 1000 / CLOCKS_PER_SEC;

        //probing with SSE hardcoded
        //explicit loading root node
        clock_t start3, diff3;
        if (useHardCoded){
                start3 = clock();
                int32_t* index = tree->key_array[0];
                register __m128i root1 = _mm_load_si128((__m128i*)&index[ 0 ]);
                register __m128i root2 = _mm_load_si128((__m128i*)&index[ 4 ]);
                for (size_t i = 0; i < num_probes; i=i+4) {
                        hardcoded_index_sse(tree, root1, root2, &probe[i], &result3[i]);
                }
                diff3 = clock() - start3;
        }
        diff3 = clock() - start3;
        int time3 = diff3 * 1000 / CLOCKS_PER_SEC;

        // output results
        if(useHardCoded){
                for (size_t i = 0; i < num_probes; ++i) {
                        fprintf(stdout, "%d %u %u %u\n", probe[i], result[i], result2[i], result3[i]);
                }
        }
        else{
                for (size_t i = 0; i < num_probes; ++i) {
                        fprintf(stdout, "%d %u %u\n", probe[i], result[i], result2[i]);
                }
        }

        //output clock results
        fprintf(stderr, "%d %d %d\n", time1, time2, time3);

        // cleanup and exit
        free(result);
        free(result2);
        free(result3);
        free(probe);
        cleanup_index(tree);
        return EXIT_SUCCESS;
}