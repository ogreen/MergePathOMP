/*
 * Authors: 
 *  Oded Green (ogreen@gatech.edu), Rob McColl (robert.c.mccoll@gmail.com)
 *  High Performance Computing Lab, Georgia Tech
 *
 * Publications (please cite):
 * GPU MergePath: A GPU Merging Algorithm
 * ACM International Conference on Supercomputing 2012
 * June 25-29 2012, San Servolo, Venice, Italy
 *
 * and
 *
 * Merge Path - Parallel Merging Made Simple
 * IEEE Workshop on Multithreaded Architectures and Applications (MTAAP)
 * 
 *
 *
 * Copyright (c) 2012 Georgia Institute of Technology
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without 
 * modification, are permitted provided that the following conditions are met:
 * 
 * - Redistributions of source code must retain the above copyright notice, 
 *   this list of conditions and the following disclaimer.
 * - Redistributions in binary form must reproduce the above copyright notice, 
 *   this list of conditions and the following disclaimer in the documentation 
 *   and/or other materials provided with the distribution.
 * - Neither the name of the Georgia Institute of Technology nor the names of 
 *   its contributors may be used to endorse or promote products derived from 
 *   this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" 
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE 
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE 
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE 
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR 
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF 
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS 
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN 
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) 
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF 
 * THE POSSIBILITY OF SUCH DAMAGE.
 */


#include <stdio.h>
#include "xmalloc.h"
#include <sys/time.h>
#include <stdint.h>
#include <float.h>
#include <getopt.h>
#include <errno.h>
#include "util.h"
#include <stdlib.h>
#include <omp.h>
#include <time.h>


#define RUNS 100

// Random Tuning Parameters
//////////////////////////////
#if 0
typedef float vec_t;
#define INFINITY_VALUE FLT_MAX
#define NEGATIVE_INFINITY_VALUE FLT_MIN
#else
typedef int32_t vec_t;
#define INFINITY_VALUE 2147483647
#define NEGATIVE_INFINITY_VALUE -2147483648
#endif

// Function Prototypes
//////////////////////////////
void MergePath(vec_t * A, uint32_t A_length, vec_t * B, uint32_t B_length, vec_t * C, uint32_t C_length, uint32_t threads);
void hostAllocateandInit(vec_t ** A, uint32_t A_length, vec_t ** B, uint32_t B_length, vec_t ** C, uint32_t C_length);
inline void serialMerge(vec_t * At, uint32_t A_length, vec_t * Bt, uint32_t B_length, vec_t * Ct, uint32_t C_length);
void hostMergePath(vec_t * A, uint32_t A_length, vec_t * B, uint32_t B_length, vec_t * C, uint32_t C_length);
void hostParseArgs(int argc, char** argv); 

// Global Host Variables
////////////////////////////
uint32_t h_ui_threads = 4; 
uint32_t h_ui_A_length = 1048576;
uint32_t h_ui_B_length = 1048576;
uint32_t h_ui_C_length = 2097152;
float h_f_serial_merge = 0;
float h_f_cuda_diagonal = 0;
float h_f_cuda_merge = 0;
uint32_t * uip_diagonal_intersections;
vec_t *globalA, *globalB, *globalC;

// Host Functions
/////////////////////////////
 int main(int argc, char** argv) {

   printf("OpenMP MergePath Implementation\n");
   hostParseArgs(argc, argv);
	printf("\nMerging: A[%d] B[%d] to C[%d] using %d threads\n", h_ui_A_length, h_ui_B_length, h_ui_C_length, h_ui_threads);
   
   tic_reset();
	hostAllocateandInit(&globalA, h_ui_A_length, &globalB, h_ui_B_length, &globalC, h_ui_C_length);

	hostMergePath(globalA, h_ui_A_length, globalB, h_ui_B_length, globalC, h_ui_C_length);


	printf("\nTotal Time %f\n", tic_total());

   return 0;
}

void hostParseArgs(int argc, char** argv) {
   static struct option long_options[] = {                                                         
      {"Alength", required_argument, 0, 'A'},
      {"Blength", required_argument, 0, 'B'},
		{"help", no_argument, 0, 'h'},
		{"threads", required_argument, 0, 't'},
      {0, 0, 0, 0}                                                                                 
   };                                                                                              
                                                                                                   
   while(1) {                                                                                      
      int option_index = 0;                                                                        
      int c = getopt_long(argc, argv, "A:B:b:t:?h", long_options, &option_index);      
      extern char * optarg;                                                                        
      extern int    optind, opterr, optopt;
		int intout = 0;                                                        
                                                                                                   
      if(-1 == c)                                                                                  
         break;                                                                                    
                                                                                                   
      switch(c) {                                                                                  
         default:                                                                                  
            printf("Unrecognized option: %c\n\n", c);                                              
         case '?':                                                                                 
         case 'h':
				printf("\nCreates two arrays A and B and merges them into array C in parallel on OpenMP.");
            printf("\n\nUsage"                                                                       
                   "\n=====" 
						 "\n\n\t-A --Alength <number>\n\t\tSpecify the length of randomly generated array A.\n"
						 "\n\t-B --Blength <number>\n\t\tSpecify the length of randomly generated array B.\n"
						 "\n\t-t --threads <number> Specify number of threads.\n");
				exit(0);
            break;
			case 'A':
            errno = 0;
            intout = strtol(optarg, NULL, 10);
            if(errno || intout < 0) {
               printf("Error - Alength %s\n", optarg);
               exit(-1);
				}
				h_ui_A_length = intout;
				break;
			case 'B':
            errno = 0;
            intout = strtol(optarg, NULL, 10);
            if(errno || intout < 0) {
               printf("Error - Blength %s\n", optarg);
               exit(-1);
				}
				h_ui_B_length = intout;
				break;
			case 't':
				errno = 0;
				intout = strtol(optarg, NULL, 10);
				if(errno || intout < 0) {
					printf("Error -threads %s\n", optarg);
					exit(-1);
				}
				h_ui_threads = intout;
				break;
      }                                                                                            
   }
	h_ui_C_length = h_ui_A_length + h_ui_B_length;
	omp_set_num_threads(h_ui_threads);
	uip_diagonal_intersections = xmalloc(2 * sizeof(uint32_t) * (h_ui_threads + 1));
}                                                                                                  

 int hostBasicCompare(const void * a, const void * b) {
 	return (int) (*(vec_t *)a - *(vec_t *)b);
}

 void hostAllocateandInit(vec_t ** A, uint32_t A_length, vec_t ** B, uint32_t B_length, vec_t ** C, uint32_t C_length) {
	if(A_length != B_length || A_length + B_length != C_length) {
		fprintf(stderr, "ERROR: |A| must equal |B| and |C| must equal |A| + |B|\n");
		return;
	}

	*A = (vec_t *) xmalloc((A_length + 1024) * (sizeof(vec_t)));
	*B = (vec_t *) xmalloc((B_length + 1024) * (sizeof(vec_t)));
	*C = (vec_t *) xmalloc((C_length + 1024) * (sizeof(vec_t)));

	srand(time(NULL));
	for(uint32_t i = 0; i < A_length; ++i) { 
		#if(DEBUG_DIAGONALS)
			(*A)[i] = rand() % 100;
		#else
			(*A)[i] = rand();
		#endif	
	}
	
	for(uint32_t i = 0; i < B_length; ++i) {
		#if(DEBUG_DIAGONALS)
			(*B)[i] = rand() % 100;
		#else
			(*B)[i] = rand();
		#endif	
	}

	qsort(*A, A_length, sizeof(vec_t), hostBasicCompare);
	qsort(*B, B_length, sizeof(vec_t), hostBasicCompare);

	for(int i = 0; i < 1024; ++i) {
		(*A)[A_length + i] = INFINITY_VALUE;
		(*B)[B_length + i] = INFINITY_VALUE;
	}  

	h_f_serial_merge = 0;
	vec_t *At = *A;
	vec_t *Bt = *B;
	vec_t *Ct = *C;

	// speed step
	for(int i = 0; i < 100; ++i) {
		uint32_t ai = 0;
		uint32_t bi = 0;
		uint32_t ci = 0;
		while(ai < A_length && bi < B_length) {
			Ct[ci++] = At[ai] < Bt[bi] ? At[ai++] : Bt[bi++];
		}
		while(ai < A_length) {
			Ct[ci++] = At[ai++];
		}
		while(bi < B_length) {
			Ct[ci++] = Bt[bi++];
		}
	}
	for(int i = 0; i < RUNS; ++i) {
		tic_reset();
		serialMerge(At, A_length, Bt, B_length, Ct, C_length);
		h_f_serial_merge += (tic_sincelast()/RUNS);
	}
	printf("\nserial merge %f\n", h_f_serial_merge);
}

inline void serialMerge(vec_t * At, uint32_t A_length, vec_t * Bt, uint32_t B_length, vec_t * Ct, uint32_t C_length) {
	uint32_t ai = 0;
	uint32_t bi = 0;
	uint32_t ci = 0;
	while(ai < A_length && bi < B_length) {
		Ct[ci++] = At[ai] < Bt[bi] ? At[ai++] : Bt[bi++];
	}
	while(ai < A_length) {
		Ct[ci++] = At[ai++];
	}
	while(bi < B_length) {
		Ct[ci++] = Bt[bi++];
	}
}

void hostMergePath(vec_t * A, uint32_t A_length, vec_t * B, uint32_t B_length, vec_t * C, uint32_t C_length) {
	vec_t *Ctest = (vec_t *) xmalloc((C_length + 1024) * (sizeof(vec_t)));

	h_f_cuda_merge = 0;
	for(int i = 0; i < RUNS; ++i) {
		tic_reset();
		MergePath(A, A_length, B, B_length, Ctest, C_length, h_ui_threads);
		h_f_cuda_merge += (tic_sincelast() / RUNS);
	}
	printf("Total OpenMP MergePath %f\nSpeedup over serial merge %f\n", h_f_cuda_merge, 
				h_f_serial_merge / (h_f_cuda_merge));

	int nogood = 0;

	for(int i = 0; i < C_length; ++i)  {
       nogood += (C[i] != Ctest[i]);
       //if(C[i] != Ctest[i])
       //printf("%d %d %d  \n",i,C[i], Ctest[i]);
	}
	if(nogood) printf("ERROR MERGING\n");
	else  printf("MERGE SUCCESSFUL\n");
}

void MergePath( vec_t * A, uint32_t A_length, vec_t * B, uint32_t B_length, vec_t * C, uint32_t C_length, uint32_t threads) {
	uip_diagonal_intersections[threads*2] = A_length;
	uip_diagonal_intersections[threads*2+1] = B_length;

	#pragma omp parallel
	{
		uint32_t thread = omp_get_thread_num();

		int32_t combinedIndex = thread * (A_length + B_length) / threads;
		int32_t x_top, y_top, x_bottom, current_x, current_y, offset;
		
		x_top = combinedIndex > A_length ? A_length : combinedIndex;
		y_top = combinedIndex > (A_length) ? combinedIndex - (A_length) : 0;
		x_bottom = y_top;

		vec_t Ai, Bi;
		while(1) {
			offset = (x_top - x_bottom) / 2;
			current_y = y_top + offset;
			current_x = x_top - offset;

			if(current_x > A_length - 1 || current_y < 1) {
				Ai = 1;
				Bi = 0;
			} else {
				Ai = A[current_x];
				Bi = B[current_y - 1];
			}
			
			if(Ai > Bi) {
				if(current_y > B_length - 1 || current_x < 1) {
					Ai = 0;
					Bi = 1;
				} else {
					Ai = A[current_x - 1];
					Bi = B[current_y];
				}

				if(Ai <= Bi) {
					//Found it
					uip_diagonal_intersections[thread*2]   = current_x;
					uip_diagonal_intersections[thread*2+1] = current_y;
					break;
				} else {
					//Both zeros
					x_top = current_x - 1;
					y_top = current_y + 1;
				}
			} else {
				// Both ones
				x_bottom = current_x + 1;
			}
		}
		
		#pragma omp barrier

		uint32_t astop = uip_diagonal_intersections[thread*2+2];
		uint32_t bstop = uip_diagonal_intersections[thread*2+3];
		uint32_t ci = current_x + current_y;

		while(current_x < astop && current_y < bstop) {
			C[ci++] = A[current_x] < B[current_y] ? A[current_x++] : B[current_y++];
		}
		while(current_x < astop) {
			C[ci++] = A[current_x++];
		}
		while(current_y < bstop) {
			C[ci++] = B[current_y++];
		}
	}
}
