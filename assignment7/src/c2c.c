#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <omp.h>

// -------------------------------------------------------------
// Util functions
// -------------------------------------------------------------

void out2file(char *filename, double *val_arr, int N){
    FILE *fptr;
    fptr = fopen(filename, "w");
    
    for (int i = 0; i < N; i++){
        for (int j = 0; j < N; j++){
            // printf("%7.3f", val_arr[i*N + j]);
            fprintf(fptr, "%7.3f", val_arr[i*N + j]);
            if (j != N-1){
                // printf(",");
                fprintf(fptr, ",");
            }
        }
        // printf("\n");
        fprintf(fptr, "\n");
    }
    fclose(fptr);
}

double wtime() {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec + 1e-6*tv.tv_usec;
}

void bench(int tc, int t1, int t2, int pings, int* addr) {
	volatile int *a = addr;
	*a = 0;

    #pragma omp parallel for
    for(int i = 0; i < tc; i++) {
        if (i == t1) {
            for(int c = 0; c < pings; c++) {
                *a = 1;
                while(*a == 1);
            }
            *a = 2;
        }
        else if (i == t2) {
            while(*a != 2) {
                    while(*a == 0);
                    if (*a == 1) *a = 0;
            }
        }
	}
}

int main(int argc, char* argv[])
{
	// parse arguments: c2c [<kilo ping-pongs> [<addresses> [<verbose>]]]
	// defaults:
	// - pings = 100 (measure 100K ping-pongs per core pair)
	// - addrs = 50 (check 50 different addresses per core pair)
	// - verb  = 0 (only print min/avg/max stat per core pair)
	//           if set to 1: show warm up for 1st addrs + per address latency
	
	int pings = -1;
	int addrs = 0;
	int verb  = 0;
	
	if (argc > 1) pings = atoi(argv[1]);
	if (argc > 2) addrs = atoi(argv[2]);
	if (argc > 3) verb = atoi(argv[3]);

	// print help if 1st arg given and cannot be parsed as int
	if (pings == 0) {
		printf( "Usage: %s [<kpps> [<addrs> [<verbose>]]]\n"
			"\n"
			"Parameters:\n"
			"  <kpps>  kilo ping-pongs to do in one measure (def: 100)\n"
			"  <addrs> number of addresses to run through per core pair (def: 50)\n"
			"  <verb>  verbosity level (def: 0), if 1 show all measurements\n", argv[0]);
		exit(1);
	}

	// set defaults if not given
	if (pings == -1) pings = 10;
	if (addrs == 0)  addrs = 30; 
	pings = pings * 1000;
    
    int *a = (int *) malloc(addrs * 64 + 64);

	// get number of threads to run ping-pongs on
	int tc = 0;
    #pragma omp parallel
    if (omp_get_thread_num() == 0)
        tc = omp_get_num_threads();

	printf("Running %.1f K ping-pongs using %d different addresses for each pair from %d threads\n", (double) pings / 1000.0, addrs, tc);
	printf("Threads: %d, Base addr: %p\n", tc, a);

    // for recording the result matrices
    double *min_arr = (double *) malloc(tc*tc * sizeof(double));
    double *max_arr = (double *) malloc(tc*tc * sizeof(double));
    double *avg_arr = (double *) malloc(tc*tc * sizeof(double));

	for(int t1 = 0; t1 < tc; t1++) {
		for(int t2 = 0; t2 < tc; t2++) {
            
			if (t1 == t2) {
                min_arr[t1*tc + t2] = 0.0;
                max_arr[t1*tc + t2] = 0.0;
                avg_arr[t1*tc + t2] = 0.0;
                continue;
            }
			printf("   Pair (%2d,%2d):\n", t1, t2);

			// Warm up of at most 20ms for core pair
			double tsum = 0.0;
			double t = wtime();
			while(tsum < .02) {
				bench(tc, t1, t2, pings, a);
				double tt = wtime();
				double tdiff = tt - t; // elapsed time of 1 bench run
				t = tt;
				tsum += tdiff;
				if (verb)
					printf("    Addr %p, warm: %.1f ns\n", a, tdiff / 2.0 / pings * 1e9);
			}

			// Measure
			double min, max, avg;
			for(int i = 0; i < addrs; i++) {
				double tt = wtime();

				// different addresses within different cache lines
				bench(tc, t1, t2, pings, a + 4*i);
				tt = (wtime() - tt) / 2.0 / pings * 1e9;
				
				// calc min, max, avg
				if (i == 0) {
					min = max = avg = tt;
				} else {
					avg += tt;
					if (tt < min) min = tt;
					if (tt > max) max = tt;
				}
				if (verb)
					printf("    Addr %p: %.1f ns\n", a + 4*i, tt);
			}
			avg = avg / addrs;
			printf("    Min: %.1f ns (max: %.1f ns, avg: %.1f ns)\n", min, max, avg);

            // recording results
            min_arr[t1*tc + t2] = min;
            max_arr[t1*tc + t2] = max;
            avg_arr[t1*tc + t2] = avg;
		}
	}

    // write output file
    // printf("------------------------------------\n");
    // char filename[] = "./avg_matrix.csv";
    // out2file(filename, avg_arr, tc);
    // printf("------------------------------------\n");

    // free memory
    free(a);
    free(min_arr);
    free(max_arr);
    free(avg_arr);

	return 0;
}

