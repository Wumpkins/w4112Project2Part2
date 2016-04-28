#include "tree.h"

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <xmmintrin.h> 
#include <emmintrin.h>  
#include <pmmintrin.h> 
#include <tmmintrin.h> 
#include <smmintrin.h>
#include <nmmintrin.h> 
#include <ammintrin.h>
#include <x86intrin.h>


extern int posix_memalign(void** memptr, size_t alignment, size_t size);
size_t alignment = 16;

Tree* build_index(size_t num_levels, size_t fanout[], size_t num_keys, int32_t key[]) {
        // return null pointer for invalid tree configuration
        size_t min_num_keys = 1;
        for (size_t i = 0; i < num_levels - 1; ++i) {
                min_num_keys *= fanout[i];
        }
        size_t max_num_keys = min_num_keys * fanout[num_levels - 1] - 1;
        if (num_keys < min_num_keys || num_keys > max_num_keys) {
                fprintf(stderr, "Error: incorrect number of keys, min %zu, max %zu\n", min_num_keys, max_num_keys);
                return NULL;
        }

        // initialize the tree index
        Tree* tree = malloc(sizeof(Tree));
        assert(tree != NULL);
        tree->num_levels = num_levels;
        tree->node_capacity = malloc(sizeof(size_t) * num_levels);
        assert(tree->node_capacity != NULL);
        for (size_t i = 0; i < num_levels; ++i) {
                tree->node_capacity[i] = fanout[i] - 1;
        }
        tree->key_array = malloc(sizeof(int32_t*) * num_levels);
        assert(tree->key_array != NULL);
        size_t* key_count = malloc(sizeof(size_t) * num_levels);
        assert(key_count != NULL);
        size_t* array_capacity = malloc(sizeof(size_t) * num_levels);
        assert(array_capacity != NULL);
        for (size_t i = 0; i < num_levels; ++i) {
                size_t size = sizeof(int32_t) * tree->node_capacity[i];         // allocate one node per level
                int error = posix_memalign((void**) &(tree->key_array[i]), alignment, size);
                assert(error == 0);
                key_count[i] = 0;
                array_capacity[i] = tree->node_capacity[i];     // array_capacity[i] is always a multiple of node_capacity[i]
        }

        // insert sorted keys into index
        for (size_t i = 1; i < num_keys; ++i) {
                assert(key[i - 1] < key[i]);
        }
        for (size_t i = 0; i < num_keys; ++i) {
                size_t level = num_levels - 1;
                while (key_count[level] == array_capacity[level])
                        level -= 1;
                tree->key_array[level][key_count[level]] = key[i];
                key_count[level] += 1;
                while (level < num_levels - 1) {
                        level += 1;
                        size_t new_capacity = array_capacity[level] + tree->node_capacity[level];
                        size_t size = sizeof(int32_t) * new_capacity;           // allocate one more node
                        int32_t* new_array = NULL;
                        int error = posix_memalign((void**) &new_array, alignment, size);
                        assert(error == 0);
                        memcpy(new_array, tree->key_array[level], sizeof(int32_t) * key_count[level]);
                        free(tree->key_array[level]);
                        tree->key_array[level] = new_array;
                        array_capacity[level] = new_capacity;
                }
        }

        // pad with INT32_MAXs
        for (size_t i = 0; i < num_levels; ++i) {
                for (size_t j = key_count[i]; j < array_capacity[i]; ++j)
                        tree->key_array[i][j] = INT32_MAX;
                key_count[i] = array_capacity[i];
        }

        // print the tree
        // for (size_t i = 0; i < num_levels; ++i) {
        //         printf("Level %zu:", i);
        //         for (size_t j = 0; j < key_count[i]; ++j)
        //                 printf(" %d", tree->key_array[i][j]);
        //         printf("\n");
        // }

        free(array_capacity);
        free(key_count);
        return tree;
}

uint32_t probe_index(Tree* tree, int32_t probe_key) {
        size_t result = 0;
        for (size_t level = 0; level < tree->num_levels; ++level) {
                size_t offset = result * tree->node_capacity[level];
                size_t low = 0;
                size_t high = tree->node_capacity[level];
                while (low != high) {
                        size_t mid = (low + high) / 2;
                        if (tree->key_array[level][mid + offset] < probe_key)
                                low = mid + 1;
                        else
                                high = mid;
                }
                size_t k = low;       // should go to child k
                result = result * (tree->node_capacity[level] + 1) + k;
        }
        return (uint32_t) result;
}

uint32_t probe_index_sse(Tree* tree, int32_t probe_key) {
        uint32_t result = 0;
        register __m128i key = _mm_cvtsi32_si128(probe_key);
        key = _mm_shuffle_epi32(key, _MM_SHUFFLE(0,0,0,0));

        for (size_t level = 0; level < tree->num_levels; ++level){
                int32_t* index = tree->key_array[level];
                if(tree->node_capacity[level] == 4){
                        register __m128i lvl = _mm_load_si128((__m128i*)&index[ result << 2 ]);
                        register __m128i cmp = _mm_cmpgt_epi32(lvl, key);
                        register __m128 cmp1 = _mm_castsi128_ps(cmp);
                        uint32_t tmp = _mm_movemask_ps(cmp1);
                        if(tmp == 0)
                                tmp = 16;
                        tmp = _bit_scan_forward(tmp);
                        result = (result << 2) + result + tmp;
                }
                else if (tree->node_capacity[level] == 8){
                        register __m128i lvla = _mm_load_si128((__m128i*)&index[ result << 3 ]);
                        register __m128i lvlb = _mm_load_si128((__m128i*)&index[ (result << 3) + 4]);
                        register __m128i cmpa = _mm_cmpgt_epi32(lvla, key);
                        register __m128i cmpb = _mm_cmpgt_epi32(lvlb, key);

                        register __m128i cmp = _mm_packs_epi32(cmpa, cmpb);
                        cmp = _mm_packs_epi16(cmp, _mm_setzero_si128());

                        uint32_t tmp = _mm_movemask_epi8(cmp);
                        if(tmp == 0)
                                tmp = 256;
                        tmp = _bit_scan_forward(tmp);
                        result = (result << 3) + result + tmp;

                }
                else if (tree->node_capacity[level] == 16){
                        register __m128i lvla = _mm_load_si128((__m128i*)&index[ result << 4 ]);
                        register __m128i lvlb = _mm_load_si128((__m128i*)&index[ (result << 4) + 4]);
                        register __m128i lvlc = _mm_load_si128((__m128i*)&index[ (result << 4) + 8 ]);
                        register __m128i lvld = _mm_load_si128((__m128i*)&index[ (result << 4) + 12]);
                        register __m128i cmpa = _mm_cmpgt_epi32(lvla, key);
                        register __m128i cmpb = _mm_cmpgt_epi32(lvlb, key);
                        register __m128i cmpc = _mm_cmpgt_epi32(lvlc, key);
                        register __m128i cmpd = _mm_cmpgt_epi32(lvld, key);

                        register __m128i cmpab = _mm_packs_epi32(cmpa, cmpb);
                        register __m128i cmpcd = _mm_packs_epi32(cmpc, cmpd);
                        register __m128i cmp = _mm_packs_epi16(cmpab, cmpcd);

                        uint32_t tmp = _mm_movemask_epi8(cmp);
                        if(tmp == 0)
                                tmp = 65536;
                        tmp = _bit_scan_forward(tmp);
                        result = (result << 4) + result + tmp;                        
                }
        }

        return result;
}

uint32_t hardcoded_index_sse(Tree* tree, int32_t probe_key) {

}

void cleanup_index(Tree* tree) {
        free(tree->node_capacity);
        for (size_t i = 0; i < tree->num_levels; ++i)
                free(tree->key_array[i]);
        free(tree->key_array);
        free(tree);
}