#include <chrono>
#include <iostream>
#include <cstring>
#include <sstream>
#include <numeric>
#include <bits/stdc++.h>

typedef std::chrono::time_point<std::chrono::high_resolution_clock> TimePoint;

inline double get_duration(TimePoint t0, TimePoint t1) {
    using dsec = std::chrono::duration<double>;
    return std::chrono::duration_cast<dsec>(t1 - t0).count();
}

struct entry {
    double v;
    int64_t next;
};

/*
 * TODO@Students: Implement the list initialization
 */
void init(int64_t N, int k, struct entry* A) {
    int64_t mask = N - 1; // N is power-of-2
    for (int64_t i = 0; i < N; ++i) {
        // A[i].v = (double)i;
        A[i].v = 0.0;
        A[i].next = (k * (i + 1)) & mask;
    }
}

double sum_indexcalc_single(int64_t N, int k, int REP, double *psum, int64_t* pdummy) {
    double sum;
    int64_t mask = (N - 1); // N is power - of -2
    int64_t dummy = 0;

    struct entry* A = new struct entry[N];
    double t_res = 0;
    for (int r = 0; r < REP; ++r) {
        sum = 0.0;
        int64_t next = 0;
        for (int64_t i = 0; i < N; ++i) {
            if (i == 0) {
                auto t0 = std::chrono::high_resolution_clock::now();
                A[next].v = 1.0;
                sum += A[next].v;
                dummy |= A[next].next;
                next = (k * (i + 1)) & mask;
                auto t1 = std::chrono::high_resolution_clock::now();
                t_res += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count(); 
                // 8 - took 106 ns 
                // x  - take 1s
            } else {
                A[next].v = 1.0;
                sum += A[next].v;
                dummy |= A[next].next;
                next = (k * (i + 1)) & mask;
            }
        }
    }
    

    *psum = sum; *pdummy = dummy;

    //double duration = get_duration(t0, t1);
    // cast duration to ns
    // double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    
    return t_res / (double) (REP); 
}

double sum_indexcalc(int64_t N, int k, int REP, double *psum, int64_t* pdummy) {
    double sum;
    int64_t mask = (N - 1); // N is power - of -2
    int64_t dummy = 0;

    struct entry* A = new struct entry[N];

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < REP; ++r) {
        sum = 0.0;
        int64_t next = 0;
        for (int64_t i = 0; i < N; ++i) {
            A[next].v = 1.0;
            sum += A[next].v;
            dummy |= A[next].next;
            next = (k * (i + 1)) & mask;
            
        }
    }
    
    auto t1 = std::chrono::high_resolution_clock::now();

    *psum = sum; *pdummy = dummy;

    //double duration = get_duration(t0, t1);
    // cast duration to ns
    double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    
    return duration / (double) (N * REP); 
}

double sum_indexload_single(int64_t N, int k, int REP, double *psum, int64_t* pdummy) {
    double sum;
    int64_t mask = (N - 1); // N is power - of -2
    int64_t dummy = 0;

    struct entry* A = new struct entry[N];

    double t_res = 0;
    for (int r = 0; r < REP; ++r) {
        sum = 0.0;
        int64_t next = 0;
        for (int64_t i = 0; i < N; ++i) {
            if (i == 0) {
                auto t0 = std::chrono::high_resolution_clock::now();
                A[next].v = 1.0;
                sum += A[next].v;
                dummy |= (k * (i + 1)) & mask;
                next = A[next].next;
                auto t1 = std::chrono::high_resolution_clock::now();
                t_res += std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
            } else {
                A[next].v = 1.0;
                sum += A[next].v;
                dummy |= (k * (i + 1)) & mask;
                next = A[next].next;
            }
        }
    }
    
    *psum = sum; *pdummy = dummy;
    
    return t_res / (double) (REP);
}

double sum_indexload(int64_t N, int k, int REP, double *psum, int64_t* pdummy) {
    double sum;
    int64_t mask = (N - 1); // N is power - of -2
    int64_t dummy = 0;

    struct entry* A = new struct entry[N];

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int r = 0; r < REP; ++r) {
        sum = 0.0;
        int64_t next = 0;
        for (int64_t i = 0; i < N; ++i) {
            A[next].v = 1.0;
            sum += A[next].v;
            dummy |= (k * (i + 1)) & mask;
            next = A[next].next;
        }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    
    *psum = sum; *pdummy = dummy;
    
    double duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
    return duration / (double) (N * REP);
}

void check_result(double result, int expected_result){
    if (result != (double) expected_result){
        printf("ERROR: %f != %d\n", result, expected_result);
        exit(-1);
    }
    return;
}

int main(int argc, char **argv) {

    if (argc != 3) {
        printf("The two parameters maximum dataset size and total number of processed points need to be provided.\n");
        exit(1);
    }

    char *pEnd;
    int64_t maximumDatasetSize = strtoll(argv[1], &pEnd, 10);
    if (errno == ERANGE) {
        printf("Problem with the first number.");
        exit(2);
    }
    int64_t totalNumberProcessedPoints = strtoll(argv[2], &pEnd, 10);
    if (errno == ERANGE) {
        printf("Problem with the second number.");
        exit(3);
    }

    fprintf(
            stderr, "Maximum dataset size = %lld, total number of processed points = %lld. Performance in MFLOPS.\n",
            maximumDatasetSize, totalNumberProcessedPoints
    );

    printf("| %20s | %12s | %12s | %12s | %12s |\n", "SumType", "Datasize", "Latency[ns]", "Cycles", "Checksum");
    
    /*                                                                                     /\
                                                                                          /  \
                                                                                         |    |
                                                                                       --:'''':--
                                                                                         :'_' :
                                                                                         _:"":\___
                                                                          ' '      ____.' :::     '._  */
    static const double goldenRatio = 32951280099.0/20365011074.0; //    . *=====<<=)           \    :
    /*                                                                    .  '      '-'-'\_      /'._.'
                                                                                           \====:_ ""
                                                                                          .'     \\
                                                                                         :       :
                                                                                        /   :    \
                                                                                       :   .      '.
                                                                                       :  : :      :
                                                                                       :__:-:__.;--'
                                                                                      '-'   '-'       */
    int64_t N = 1024; // = 2^10
    int k = 0;
    double sum_indexcalc_time=0, sum_indexload_time=0 ;
    double sum_calc=0, sum_load=0;
    int64_t cycles;
    int64_t dummy;
    while (N <= maximumDatasetSize) 
    {
        cycles = std::clamp(totalNumberProcessedPoints / N, (int64_t)1, (int64_t)1000);

        sum_indexcalc_time = sum_indexcalc_single(N, 1, cycles, &sum_calc, &dummy);
        printf("| %20s | %12ld | %12.2f | %12ld | %12.1f |  \n", "sum_indexcalc k=1",N, sum_indexcalc_time, cycles, sum_calc);
        check_result(sum_calc, N); 
        sum_indexload_time = sum_indexload_single(N, 1, cycles, &sum_load, &dummy);
        printf("| %20s | %12ld | %12.2f | %12ld | %12.1f |  \n", "sum_indexload k=1",N, sum_indexload_time, cycles, sum_load );
        check_result(sum_load, N);

        // goldenRatio \approx k / N f
        k = goldenRatio * N;
        // numbers are coprime if gcd(N, k) = 1
        while(std::gcd(N, k) != 1) k++;

        sum_indexcalc_time = sum_indexcalc_single(N, k, cycles, &sum_calc, &dummy);
        printf("| %20s | %12ld | %12.2f | %12ld | %12.1f | \n", "sum_indexcalc k=gold", N, sum_indexcalc_time, cycles, sum_calc);
        check_result(sum_calc, N);
        sum_indexload_time = sum_indexload_single(N, k, cycles, &sum_load, &dummy);
        printf("| %20s | %12ld | %12.2f | %12ld | %12.1f | \n", "sum_indexload k=gold", N, sum_indexload_time, cycles, sum_load );
        check_result(sum_load, N);

        N *= 2;
    }

    return 0;
}
