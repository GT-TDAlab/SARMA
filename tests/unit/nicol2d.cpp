#include "tests.h"
#include "test_utils.hpp"

#include "algorithms/nicol1d.hpp"
#include "tools/utils.hpp"

using namespace sarma;

template<typename Ordinal, typename Value>
bool greedy_2dcuts_basic(std::vector<std::vector<Value>> & prefixes, std::vector<Ordinal> & cuts, Value B) {

    Ordinal Q = (Ordinal)prefixes.size();
    Ordinal N = (Ordinal)prefixes[0].size()-1;
    // valid prefixes
    for (Ordinal i=0; i<Q; i++) {
        if (!test_utils::is_valid(prefixes[i], prefixes[i].back())) {
            std::cout << "Invalid prefix on row : " << i << std::endl;
            return false;
        }
    }

    // valid cuts
    if (!test_utils::is_valid(cuts, N)) {
        std::cout << "In valid cuts with N=" << N << std::endl;
        for (auto cut : cuts)
            std::cout << cut << " ";
        std::cout << std::endl;
        return false;
    }

    Ordinal P = (Ordinal)cuts.size()-1;
    std::vector<Value> previous_sum(Q, 0);
    for (Ordinal i=1; i<=P; i++) {
        bool next_not_valid = false;
        for (Ordinal j=0; j<Q; j++) {
            bool valid = ((prefixes[j][cuts[i]] - previous_sum[j]) <= B);
            if (!valid) {
                std::cout << "load limit is: " << B << std::endl;
                std::cout << "but at row " << j << ": cuts[" << i << "]=" << cuts[i] << ", cuts[" << i-1 << "]=" << \
                cuts[i-1] <<  ", produces parts with weight: " << prefixes[j][cuts[i]] - previous_sum[j] << std::endl;
                return false;
            }

            if (i<P && (prefixes[j][cuts[i]+1] - previous_sum[j]) > B)
                next_not_valid = true;

            previous_sum[j] = prefixes[j][cuts[i]];
        }

        if (i<P && !next_not_valid) {
            std::cout << "load limit is: " << B << ", with total " << P+1 << " cuts"<< std::endl;
            std::cout << "but cut " << i << " can be bigger with at least one" << std::endl;
            return false;
        }
    }
    return true;
}

template<typename Ordinal, typename Value>
int probe2d_example1() {
    int pass = 0;

    Ordinal N=3;
    std::vector<Value> prefix;
    prefix.push_back(0);
    prefix.push_back(15);
    prefix.push_back(16);
    prefix.push_back(50);
    std::vector<Ordinal> ub;
    ub.push_back(N);
    ub.push_back(N);

    Ordinal Qs[] = {1, 2, 50};
    for (auto Q : Qs) {
        std::vector<std::vector<Value>> prefixes(Q, std::vector<Value>());
        for (Ordinal i=0; i<Q; i++) {
            prefixes[i] = prefix;
        }

        // L=50, 51, 100, suppose to have cuts {0, N}
        {
            Value Ls[3] = {50, 51, 100};
            std::vector<Ordinal> expected_cut;
            expected_cut.push_back(0);
            expected_cut.push_back(N);

            for (auto L : Ls) {
                // P = 1
                auto cuts = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, L, &ub);
                pass |= test_utils::is_same(cuts, expected_cut);
                cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L, &ub);
                pass |= test_utils::is_same(cuts, expected_cut);
                cuts = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, L, &expected_cut);
                pass |= test_utils::is_same(cuts, expected_cut);

                cuts = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, L);
                pass |= test_utils::is_same(cuts, expected_cut);
                cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L);
                pass |= test_utils::is_same(cuts, expected_cut);
            }
        }

        // L=0, 1, 14, 15, 30, 33, suppose to fail since L is bigger than the largest weight
        {
            Value Ls[] = {0, 1, 14, 15, 30, 33};

            for (auto L : Ls) {
                // P = 1
                auto cuts = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, L, &ub);
                EQ(cuts.back() < N, true)
                cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L, &ub);
                EQ(cuts.back() < N, true)

                cuts = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, L);
                EQ(cuts.back() < N, true)
                cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L);
                EQ(cuts.back() < N, true)
            }
        }

        // L=34, suppose to fail for ub is set, and success in the another caes.
        {
            Value Ls[] = {34};

            for (auto L : Ls) {
                // P = 1
                auto cuts = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, L, &ub);
                EQ(cuts.back() < N, true)
                cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L, &ub);
                EQ(cuts.back() < N, true)

                cuts = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, L);
                EQ(cuts.back() < N, false)
                cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L);
                EQ(cuts.back() < N, false)
            }
        }

        for (Ordinal i=0; i<Q; i++) {
            prefixes[i].clear();
        }
        prefixes.clear();
    }

    return pass;
}

template<typename Ordinal, typename Value>
int probe2d_example2() {
    int pass = 0;

    Ordinal N=5;
    Ordinal Q=3;

    std::vector<std::vector<Value>> prefixes(Q, std::vector<Value>());
    prefixes[0] = {0, 15, 16, 50, 53, 54};
    prefixes[1] = {0, 15, 20, 33, 41, 44};
    prefixes[2] = {0, 15, 17, 22, 31, 56};

    std::vector<Ordinal> ub;
    ub.push_back(N);
    ub.push_back(N);

    // L=55 suppose to fail in the case with ub set, and success in other case
    {
        Value Ls[] = {55};
        std::vector<Ordinal> expected_cut;
        expected_cut.push_back(0);
        expected_cut.push_back(4);
        expected_cut.push_back(5);

        for (auto L : Ls) {
            // P = 1
            auto cuts = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, L, &ub);
            EQ(cuts.back() < N, true)
            cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L, &ub);
            EQ(cuts.back() < N, true)

            cuts = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, L);
            pass |= test_utils::is_same(cuts, expected_cut);
            cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L);
            pass |= test_utils::is_same(cuts, expected_cut);
        }
    }

    // L=56 suppose to success with cuts as {0, 5}
    {
        Value Ls[] = {56};
        std::vector<Ordinal> expected_cut;
        expected_cut.push_back(0);
        expected_cut.push_back(5);

        for (auto L : Ls) {
            // P = 1
            auto cuts = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, L, &ub);
            pass |= test_utils::is_same(cuts, expected_cut);
            cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L, &ub);
            pass |= test_utils::is_same(cuts, expected_cut);
            cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L, &expected_cut);
            pass |= test_utils::is_same(cuts, expected_cut);

            cuts = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, L);
            pass |= test_utils::is_same(cuts, expected_cut);
            cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L);
            pass |= test_utils::is_same(cuts, expected_cut);
        }
    }

    ub.push_back(N);
    // L=34, suppose to fail in the case with given P=2 and success in the other cases such that having same cut as expected
    {
        Value Ls[] = {34};
        std::vector<Ordinal> expected_cut;
        expected_cut.push_back(0);
        expected_cut.push_back(2);
        expected_cut.push_back(3);
        expected_cut.push_back(5);

        for (auto L : Ls) {
            // P = 2
            // suppose to fail
            auto cuts = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, L, &ub);
            EQ(cuts.back() < N, true)
            cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L, &ub);
            EQ(cuts.back() < N, true)

            // P = 3
            // suppose to success with cuts as expected
            cuts = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, L);
            pass |= test_utils::is_same(cuts, expected_cut);
            cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L);
            pass |= test_utils::is_same(cuts, expected_cut);
        }

        ub.push_back(N);
        for (auto L : Ls) {
            // P = 3
            // suppose to success with cuts as expected
            auto cuts = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, L, &ub);
            pass |= test_utils::is_same(cuts, expected_cut);
            cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L, &ub);
            pass |= test_utils::is_same(cuts, expected_cut);
            cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L, &expected_cut);
            pass |= test_utils::is_same(cuts, expected_cut);

            expected_cut[1]--;
            cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L, &expected_cut);
            EQ(cuts.back() < N, true)
            expected_cut[1]++;

            expected_cut[2]--;
            cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L, &expected_cut);
            EQ(cuts.back() < N, true)
            expected_cut[2]++;

            expected_cut[3]--;
            cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L, &expected_cut);
            EQ(cuts.back() == N-1, true)
            expected_cut[3]++;
        }
    }

    // L=0, 1, 14, 15, 30, 33, suppose to fail since L is bigger than the largest weight
    {
        Value Ls[] = {0, 1, 14, 15, 30, 33};

        for (auto L : Ls) {
            // P = 3
            auto cuts = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, L, &ub);
            EQ(cuts.back() < N, true)
            cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L, &ub);
            EQ(cuts.back() < N, true)

            cuts = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, L);
            EQ(cuts.back() < N, true)
            cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L);
            EQ(cuts.back() < N, true)
        }
    }

    for (Ordinal i=0; i<Q; i++) {
        prefixes[i].clear();
    }
    prefixes.clear();

    return pass;
}

template<typename Ordinal, typename Value>
int probe2d_allzeros() {
    int pass = 0;

    Ordinal N=1000;
    Ordinal Q=50;
    std::vector<Value> prefix(N+1, 0);
    std::vector<std::vector<Value>> prefixes(Q, prefix);

    std::vector<Ordinal> ub;
    Value Ls[] = {0, 1};
    Ordinal Ps[] = {1, 2, 5};
    for (auto L : Ls) {

        std::vector<Ordinal> expected_cuts;
        expected_cuts.push_back(0);
        expected_cuts.push_back(N);

        auto cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L);
        pass |= test_utils::is_same(cuts, expected_cuts);
        cuts = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, L);
        pass |= test_utils::is_same(cuts, expected_cuts);

        for (auto P : Ps) {
            ub.clear();
            for(Ordinal i=0; i<P+1; i++)
                ub.push_back(N);

            while(expected_cuts.size() < ub.size()) {
                expected_cuts.push_back(N);
            }
            auto cuts = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, L, &ub);
            pass |= test_utils::is_same(cuts, expected_cuts);
            cuts = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, L, &ub);
            pass |= test_utils::is_same(cuts, expected_cuts);
        }
    }

    return pass;
}

template<typename Ordinal, typename Value>
int probe2d_randomnumber( unsigned int seed ) {
    int pass = 0;

    srand( seed );
    std::cout << "    random seed: " << seed << std::endl;

    Ordinal max_n = 200;
    Ordinal max_q = 50;
    Ordinal test_times_prefix = 10;
    Ordinal test_times_PL = 100;

    for (Ordinal i=0; i<test_times_prefix; i++) {
        Ordinal N = rand() % max_n + 1;
        Ordinal Q = rand() % max_q + 1;

        std::vector<std::vector<Value>> prefixes(Q, std::vector<Value>());
        Value max_sum = 0;
        Value max_element = 0;
        for (Ordinal i=0; i<Q; i++) {
            prefixes[i].push_back((Value)0);
            for (Ordinal j=1; j<=N; j++) {
                Value e = (Value)(rand()%(RAND_MAX/N)) + ((Value)rand()/RAND_MAX);
                prefixes[i].push_back(prefixes[i][j-1] + e);
                max_element = std::max(max_element, e);
            }

            max_sum = (max_sum == 0) ? prefixes[i][N] : (std::max(prefixes[i][N], max_sum));
        }

        for (Ordinal j=0; j<test_times_PL; j++) {
            Ordinal P = (rand() % N) + 1;
            P = (P>(N/20)) ? ((rand() % N) + 1) : P;
            P = (P>(N/20)) ? ((rand() % N) + 1) : P;
            Value B = (Value)(rand() % ((int)max_sum/P*2)) + ((Value)rand()/RAND_MAX);

            std::vector<Ordinal> ub;
            for (Ordinal k=0; k<=P; k++)
                ub.push_back(N);

            auto cuts_with_p = sarma::nicol1d::probe<Ordinal, Value, true >(&prefixes[0], Q, B, &ub);
            auto cuts_temp = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, B, &ub);
            pass |= test_utils::is_same(cuts_with_p, cuts_temp);
            EQ((Ordinal)cuts_with_p.size(), P+1)

            auto cuts_without_p = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, B);
            cuts_temp = sarma::nicol1d::probe<Ordinal, Value, false>(&prefixes[0], Q, B);
            pass |= test_utils::is_same(cuts_without_p, cuts_temp);

            if (cuts_without_p.back() < N) {
                EQ(B<max_sum, true)
                EQ(cuts_with_p.back() < N, true)
            } else {
                Ordinal min_cuts_size = std::min(P+1, (Ordinal)cuts_without_p.size());
                for (Ordinal k=0; k<min_cuts_size; k++)
                    EQ(cuts_without_p[k], cuts_with_p[k])
                while (min_cuts_size < P+1)
                    EQ(cuts_with_p[min_cuts_size++], N)

                while (min_cuts_size < (Ordinal)cuts_without_p.size()) {
                    EQ(cuts_without_p[min_cuts_size] > cuts_without_p[min_cuts_size-1], true)
                    min_cuts_size++;
                }
                EQ(cuts_without_p.back(), N)

                EQ(greedy_2dcuts_basic(prefixes, cuts_without_p, B), true)
            }
        }
        for (Ordinal i=0; i<Q; i++)
            prefixes[i].clear();
        prefixes.clear();
    }

    return pass;
}

template<typename Ordinal, typename Value>
int nicol2dpartition_versioncompare(std::vector<Value> * prefixes, Ordinal Q, Ordinal P, std::vector<Ordinal> & expected_cuts = std::vector<Ordinal>()) {
    int pass = 0;
    auto cuts = sarma::nicol1d::partition<Ordinal, Value, true>(prefixes, Q, P);
    if (expected_cuts.size() != 0)
        pass |= test_utils::is_same(cuts, expected_cuts);
    else
        expected_cuts = cuts;

#if defined(ENABLE_CPP_PARALLEL)
    if constexpr( std::is_integral_v<Value> ) {
#else
    if (std::is_integral<Value>::value ) {
#endif
        cuts = sarma::nicol1d::partition<Ordinal, Value, false>(prefixes, Q, P);
        pass |= test_utils::is_same(cuts, expected_cuts);
    }

    return pass;
}

template<typename Ordinal, typename Value>
int partition2d_example1() {
    int pass = 0;

    Ordinal N=3;
    std::vector<Value> prefix;
    prefix.push_back(0);
    prefix.push_back(15);
    prefix.push_back(16);
    prefix.push_back(50);

    Ordinal Qs[] = {1, 2, 50};
    for (auto Q : Qs) {
        std::vector<std::vector<Value>> prefixes(Q, prefix);

        std::vector<Ordinal> Ps({1, 2, 3, 4, 5});
        std::vector<Ordinal> expected_cuts[] = {
            {0, N},
            {0, 2, N},
            {0, 2, N, N},
            {0, 2, N, N, N},
            {0, 2, N, N, N, N},
        };
        for (Ordinal i=0; i<(Ordinal)Ps.size(); i++)
            pass |= nicol2dpartition_versioncompare<Ordinal, Value>(&prefixes[0], Q, Ps[i], expected_cuts[i]);

        for (Ordinal i=0; i<Q; i++) {
            prefixes[i].clear();
        }
        prefixes.clear();
    }

    return pass;
}

template<typename Ordinal, typename Value>
int partition2d_example2() {
    int pass = 0;

    Ordinal N=5;
    Ordinal Q=3;

    std::vector<std::vector<Value>> prefixes(Q, std::vector<Value>());
    prefixes[0] = {0, 15, 16, 50, 53, 54};
    prefixes[1] = {0, 15, 20, 33, 41, 44};
    prefixes[2] = {0, 15, 17, 22, 31, 56};

    std::vector<Ordinal> Ps({1, 2, 3, 4, 5});
    std::vector<Ordinal> expected_cuts[] = {
        {0, N},
        {0, 2, N},
        {0, 2, 3, N},
        {0, 2, 3, N, N},
        {0, 2, 3, N, N, N},
    };
    for (Ordinal i=0; i<(Ordinal)Ps.size(); i++)
        pass |= nicol2dpartition_versioncompare<Ordinal, Value>(&prefixes[0], Q, Ps[i], expected_cuts[i]);

    for (Ordinal i=0; i<Q; i++) {
        prefixes[i].clear();
    }
    prefixes.clear();

    return pass;
}

template<typename Ordinal, typename Value>
int partition2d_allzeros() {
    int pass = 0;

    Ordinal N=1000;
    Ordinal Q=50;
    std::vector<Value> prefix(N+1, 0);
    std::vector<std::vector<Value>> prefixes(Q, prefix);

    std::vector<Ordinal> Ps({1, 2, 3});
    std::vector<Ordinal> expected_cuts[] = {
        {0, N},
        {0, N, N},
        {0, N, N, N},
    };
    for (Ordinal i=0; i<(Ordinal)Ps.size(); i++)
        pass |= nicol2dpartition_versioncompare<Ordinal, Value>(&prefixes[0], Q, Ps[i], expected_cuts[i]);

    for (Ordinal i=0; i<Q; i++) {
        prefixes[i].clear();
    }
    prefixes.clear();

    return pass;
}

template<typename Ordinal, typename Value>
std::vector<Value> get_possible_loads(std::vector<std::vector<Value>> & prefixes) {
    Ordinal Q = (Ordinal)prefixes.size();
    Ordinal S = (Ordinal)prefixes[0].size();
    std::vector<Value> possible_loads;
    for (Ordinal i=0; i<S-1; i++) {
        for (Ordinal j=i+1; j<S; j++) {
            Value max_load_ij = prefixes[0][j] - prefixes[0][i];
            for (Ordinal k=1; k<Q; k++)
                max_load_ij = std::max(max_load_ij, prefixes[k][j] - prefixes[k][i]);
            possible_loads.push_back(max_load_ij);
        }
    }
    std::sort(possible_loads.begin(), possible_loads.end());
    return possible_loads;
}

template<typename Ordinal, typename Value>
Value get_max_load(std::vector<std::vector<Value>> & prefixes, std::vector<Ordinal> & cuts) {
    Ordinal Q = (Ordinal)prefixes.size();
    Ordinal P = (Ordinal)cuts.size() - 1;

    Value max_load = prefixes[0][cuts[1]] - prefixes[0][cuts[0]];
    for (Ordinal i=0; i<P; i++)
        for (Ordinal k=0; k<Q; k++)
            max_load = std::max(max_load, prefixes[k][cuts[i+1]] - prefixes[k][cuts[i]]);

    return max_load;
}

template<typename Ordinal, typename Value>
int verify_best_cuts(std::vector<std::vector<Value>> & prefixes, std::vector<Ordinal> & cuts,
        std::vector<Value> & possible_loads) {
    int pass = 0;
    Ordinal Q = (Ordinal)prefixes.size();
    Ordinal N = (Ordinal)prefixes[0].size()-1;
    Ordinal P = (Ordinal)cuts.size() - 1;

    auto it = std::find(possible_loads.begin(), possible_loads.end(), get_max_load(prefixes, cuts));
    EQ ( it != possible_loads.end(), true )
    if ( it == possible_loads.begin() )
        return pass;

    Value load_l = possible_loads[ it-possible_loads.begin()-1 ];
    std::vector<Ordinal> ub(P+1, N);
    auto cuts_l = sarma::nicol1d::probe<Ordinal, Value, true>(&prefixes[0], Q, load_l, &ub);
    EQ(cuts_l.back() < N, true )

    return pass;
}

template<typename Ordinal, typename Value>
int partition2d_randomnumber( unsigned int seed ) {
    int pass = 0;

    srand( seed );
    std::cout << "    random seed: " << seed << std::endl;

    Ordinal max_n = 100;
    Ordinal max_q = 5;
    Ordinal max_p = 100;
    Ordinal test_times_prefix = 10;
    Ordinal test_times_P = 10;

    for (Ordinal i=0; i<test_times_prefix; i++) {
        Ordinal N = rand() % max_n + 1;
        Ordinal Q = rand() % max_q + 1;

        std::vector<std::vector<Value>> prefixes(Q, std::vector<Value>());
        for (Ordinal j=0; j<Q; j++) {
            prefixes[j].push_back((Value)0);
            for (Ordinal k=1; k<=N; k++) {
                Value e = (Value)(rand()%(RAND_MAX/N)) + ((Value)rand()/RAND_MAX);
                prefixes[j].push_back(prefixes[j][k-1] + e);
            }
        }
        std::vector<Value> possible_loads = get_possible_loads<Ordinal, Value>(prefixes);

        for (Ordinal j=0; j<test_times_P; j++) {
            Ordinal P = rand() % max_p % N + 1;

            std::vector<Ordinal> cuts;
            pass |= nicol2dpartition_versioncompare(&prefixes[0], Q, P, cuts);
            pass |= verify_best_cuts<Ordinal, Value>(prefixes, cuts, possible_loads);
        }

        for (Ordinal j=0; j<Q; j++)
            prefixes[j].clear();
        prefixes.clear();
    }

    return pass;
}

int probe2d() {
    int pass = 0;

    TEST(probe2d_example1<int COMMA int>)
    TEST(probe2d_example1<int COMMA unsigned int>)
    TEST(probe2d_example1<int COMMA float>)
    TEST(probe2d_example1<int COMMA double>)

    TEST(probe2d_example2<int COMMA int>)
    TEST(probe2d_example2<int COMMA unsigned int>)
    TEST(probe2d_example2<int COMMA float>)
    TEST(probe2d_example2<int COMMA double>)

    TEST(probe2d_allzeros<int COMMA int>)
    TEST(probe2d_allzeros<int COMMA unsigned int>)
    TEST(probe2d_allzeros<int COMMA float>)
    TEST(probe2d_allzeros<int COMMA double>)

    TEST(probe2d_randomnumber<int COMMA int>, time(NULL))
    TEST(probe2d_randomnumber<int COMMA unsigned int>, time(NULL))
    TEST(probe2d_randomnumber<int COMMA float>, time(NULL))
    TEST(probe2d_randomnumber<int COMMA double>, time(NULL))

    return pass;
}

int partition2d() {
    int pass = 0;

    TEST(partition2d_example1<int COMMA int>)
    TEST(partition2d_example1<int COMMA unsigned int>)
#if defined(ENABLE_CPP_PARALLEL)
    TEST(partition2d_example1<int COMMA float>)
    TEST(partition2d_example1<int COMMA double>)
#endif

    TEST(partition2d_example2<int COMMA int>)
    TEST(partition2d_example2<int COMMA unsigned int>)
#if defined(ENABLE_CPP_PARALLEL)
    TEST(partition2d_example2<int COMMA float>)
    TEST(partition2d_example2<int COMMA double>)
#endif

    TEST(partition2d_allzeros<int COMMA int>)
    TEST(partition2d_allzeros<int COMMA unsigned int>)
#if defined(ENABLE_CPP_PARALLEL)
    TEST(partition2d_allzeros<int COMMA float>)
    TEST(partition2d_allzeros<int COMMA double>)
#endif

    TEST(partition2d_randomnumber<int COMMA int>, time(NULL))
    TEST(partition2d_randomnumber<int COMMA unsigned int>, time(NULL))
#if defined(ENABLE_CPP_PARALLEL)
    TEST(partition2d_randomnumber<int COMMA float>, time(NULL))
    TEST(partition2d_randomnumber<int COMMA double>, time(NULL))
#endif

    return pass;
}

int main() {
    int pass = 0;

    TEST(probe2d)

    TEST(partition2d)

    return pass;
}
