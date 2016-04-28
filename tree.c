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

uint32_t probe_index_sse(Tree* tree, int32_t probe_key){
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

uint32_t* hardcoded_index_sse(Tree* tree, register __m128i root1, register __m128i root2, int32_t* probe_keys) {
        uint32_t* result = malloc (sizeof(uint32_t) * 4);

        register __m128i k = _mm_load_si128((__m128i*) probe_keys);
        register __m128i k1 = _mm_shuffle_epi32(k, _MM_SHUFFLE(0,0,0,0));
        register __m128i k2 = _mm_shuffle_epi32(k, _MM_SHUFFLE(1,1,1,1));
        register __m128i k3 = _mm_shuffle_epi32(k, _MM_SHUFFLE(2,2,2,2));
        register __m128i k4 = _mm_shuffle_epi32(k, _MM_SHUFFLE(3,3,3,3));

        //root 
        //1
        register __m128i cmp01a = _mm_cmpgt_epi32(root1, k1);
        register __m128i cmp01b = _mm_cmpgt_epi32(root2, k1);

        register __m128i cmp01 = _mm_packs_epi32(cmp01a, cmp01b);
        cmp01 = _mm_packs_epi16(cmp01, _mm_setzero_si128());

        uint32_t r01 = _mm_movemask_epi8(cmp01);
        if(r01 == 0)
                r01 = 256;
        r01 = _bit_scan_forward(r01);

        //2
        register __m128i cmp02a = _mm_cmpgt_epi32(root1, k2);
        register __m128i cmp02b = _mm_cmpgt_epi32(root2, k2);

        register __m128i cmp02 = _mm_packs_epi32(cmp02a, cmp02b);
        cmp02 = _mm_packs_epi16(cmp02, _mm_setzero_si128());

        uint32_t r02 = _mm_movemask_epi8(cmp02);
        if(r02 == 0)
                r02 = 256;
        r02 = _bit_scan_forward(r02);

        //3
        register __m128i cmp03a = _mm_cmpgt_epi32(root1, k3);
        register __m128i cmp03b = _mm_cmpgt_epi32(root2, k3);

        register __m128i cmp03 = _mm_packs_epi32(cmp03a, cmp03b);
        cmp03 = _mm_packs_epi16(cmp03, _mm_setzero_si128());

        uint32_t r03 = _mm_movemask_epi8(cmp03);
        if(r03 == 0)
                r03 = 256;
        r03 = _bit_scan_forward(r03);

        //4
        register __m128i cmp04a = _mm_cmpgt_epi32(root1, k4);
        register __m128i cmp04b = _mm_cmpgt_epi32(root2, k4);

        register __m128i cmp04 = _mm_packs_epi32(cmp04a, cmp04b);
        cmp04 = _mm_packs_epi16(cmp04, _mm_setzero_si128());

        uint32_t r04 = _mm_movemask_epi8(cmp04);
        if(r04 == 0)
                r04 = 256;
        r04 = _bit_scan_forward(r04);


        //fanout 5
        int32_t* index1 = tree->key_array[1];

        //1
        register __m128i lvl11 = _mm_load_si128((__m128i*)&index1[ r01 << 2 ]);
        register __m128i cmp11 = _mm_cmpgt_epi32(lvl11, k1);
        register __m128 cmp11c = _mm_castsi128_ps(cmp11);
        uint32_t r11 = _mm_movemask_ps(cmp11c);
        if(r11 == 0)
                r11 = 16;
        r11 = _bit_scan_forward(r11);
        r11 += (r01 << 2) + r01;

        //2
        register __m128i lvl12 = _mm_load_si128((__m128i*)&index1[ r02 << 2 ]);
        register __m128i cmp12 = _mm_cmpgt_epi32(lvl12, k2);
        register __m128 cmp12c = _mm_castsi128_ps(cmp12);
        uint32_t r12 = _mm_movemask_ps(cmp12c);
        if(r12 == 0)
                r12 = 16;
        r12 = _bit_scan_forward(r12);
        r12 += (r02 << 2) + r02;

        //3
        register __m128i lvl13 = _mm_load_si128((__m128i*)&index1[ r03 << 2 ]);
        register __m128i cmp13 = _mm_cmpgt_epi32(lvl13, k3);
        register __m128 cmp13c = _mm_castsi128_ps(cmp13);
        uint32_t r13 = _mm_movemask_ps(cmp13c);
        if(r13 == 0)
                r13 = 16;
        r13 = _bit_scan_forward(r13);
        r13 += (r03 << 2) + r03;

        //4
        register __m128i lvl14 = _mm_load_si128((__m128i*)&index1[ r04 << 2 ]);
        register __m128i cmp14 = _mm_cmpgt_epi32(lvl14, k4);
        register __m128 cmp14c = _mm_castsi128_ps(cmp14);
        uint32_t r14 = _mm_movemask_ps(cmp14c);
        if(r14 == 0)
                r14 = 16;
        r14 = _bit_scan_forward(r14);
        r14 += (r04 << 2) + r04;


        //fanout 9 final
        int32_t* index2 = tree->key_array[2];

        //1
        register __m128i lvl21a = _mm_load_si128((__m128i*)&index2[ r11 << 3 ]);
        register __m128i lvl21b = _mm_load_si128((__m128i*)&index2[ (r11 << 3) + 4]);
        register __m128i cmp21a = _mm_cmpgt_epi32(lvl21a, k1);
        register __m128i cmp21b = _mm_cmpgt_epi32(lvl21b, k1);

        register __m128i cmp21 = _mm_packs_epi32(cmp21a, cmp21b);
        cmp21 = _mm_packs_epi16(cmp21, _mm_setzero_si128());

        uint32_t r21 = _mm_movemask_epi8(cmp21);
        if(r21 == 0)
                r21 = 256;
        r21 = _bit_scan_forward(r21);
        r21 += (r11 << 3) + r11;
        result[0] = r21;

        //2
        register __m128i lvl22a = _mm_load_si128((__m128i*)&index2[ r12 << 3 ]);
        register __m128i lvl22b = _mm_load_si128((__m128i*)&index2[ (r12 << 3) + 4]);
        register __m128i cmp22a = _mm_cmpgt_epi32(lvl22a, k2);
        register __m128i cmp22b = _mm_cmpgt_epi32(lvl22b, k2);

        register __m128i cmp22 = _mm_packs_epi32(cmp22a, cmp22b);
        cmp22 = _mm_packs_epi16(cmp22, _mm_setzero_si128());

        uint32_t r22 = _mm_movemask_epi8(cmp22);
        if(r22 == 0)
                r22 = 256;
        r22 = _bit_scan_forward(r22);
        r22 += (r12 << 3) + r12;
        result[1] = r22;

        //3
        register __m128i lvl23a = _mm_load_si128((__m128i*)&index2[ r13 << 3 ]);
        register __m128i lvl23b = _mm_load_si128((__m128i*)&index2[ (r13 << 3) + 4]);
        register __m128i cmp23a = _mm_cmpgt_epi32(lvl23a, k3);
        register __m128i cmp23b = _mm_cmpgt_epi32(lvl23b, k3);

        register __m128i cmp23 = _mm_packs_epi32(cmp23a, cmp23b);
        cmp23 = _mm_packs_epi16(cmp23, _mm_setzero_si128());

        uint32_t r23 = _mm_movemask_epi8(cmp23);
        if(r23 == 0)
                r23 = 256;
        r23 = _bit_scan_forward(r23);
        r23 += (r13 << 3) + r13;
        result[2] = r23;

        //4
        register __m128i lvl24a = _mm_load_si128((__m128i*)&index2[ r14 << 3 ]);
        register __m128i lvl24b = _mm_load_si128((__m128i*)&index2[ (r14 << 3) + 4]);
        register __m128i cmp24a = _mm_cmpgt_epi32(lvl24a, k4);
        register __m128i cmp24b = _mm_cmpgt_epi32(lvl24b, k4);

        register __m128i cmp24 = _mm_packs_epi32(cmp24a, cmp24b);
        cmp24 = _mm_packs_epi16(cmp24, _mm_setzero_si128());

        uint32_t r24 = _mm_movemask_epi8(cmp24);
        if(r24 == 0)
                r24 = 256;
        r24 = _bit_scan_forward(r24);
        r24 += (r14 << 3) + r14;
        result[3] = r24;

        return result;
}

void cleanup_index(Tree* tree) {
        free(tree->node_capacity);
        for (size_t i = 0; i < tree->num_levels; ++i)
                free(tree->key_array[i]);
        free(tree->key_array);
        free(tree);
}