#include "tests.h"
#include "test_utils.hpp"

#include "algorithms/nicol1d.hpp"

using namespace sarma;

template<typename D>
int write_vector_test(const std::string file_name) {
    int pass = 0;

    // construct data
    long s = 10;
    std::vector<D> data;
    for (long i=0; i<s; i++) {
        data.push_back( (D)(i*1.5) );
    }

    // write
    EQ(sarma::test_utils::write_vector(file_name, data), true);

    data.clear();

    return pass;
}

template<typename D>
int read_vector_test(const std::string file_name) {
    int pass = 0;

    // read
    std::vector<D> r = sarma::test_utils::read_vector<D>(file_name);

    // compare
    EQ((long)r.size(), (long)10);
    for (long i=0; i<(long)r.size(); i++) {
        EQ((D)r[i], (D)(i*1.5));
    }

    return pass;
}

int write_read_vector(std::string fname) {
    int pass = 0;

    std::cout << "    " << fname << std::endl;

    TEST(write_vector_test<int>, fname)
    TEST(read_vector_test<int>, fname)
    EQ(remove((fname).c_str()), 0);

    TEST(write_vector_test<double>, fname)
    TEST(read_vector_test<double>, fname)
    EQ(remove((fname).c_str()), 0);

    return pass;
}

int write_read_vector() {
    int pass = 0;
    std::string fname {"./temp_out_nicol1d_test"};

    TEST(write_read_vector, fname + ".bin");
    TEST(write_read_vector, fname + ".txt");

    return pass;
}

template<typename Ordinal, typename Value>
bool probe1d_basic(std::vector<Value> & prefix, Ordinal P, Value B) {

    if (!test_utils::is_valid(prefix, prefix.back()))
        return false;

    if (prefix.back() / P > B)
        return false;

    if (prefix.back() <= B)
        return true;

    int s = prefix.size();
    Ordinal c = 0;
    Ordinal p = 1;
    Ordinal i = 1;

    while (i<s) {
        if (prefix[i] - prefix[c] > B) {
            if (prefix[i-1] - prefix[c] <= B) {
                c = i-1;
                p++;
            } else
                return false;
        }
        i++;
    }

    return p <= P && prefix[s-1]-prefix[c] <= B;
}

template<typename Ordinal, typename Value>
int probe1d_onenumber() {
    int pass = 0;

    bool res;

    std::vector<Value> prefix;
    prefix.push_back((Value)0);
    prefix.push_back((Value)0);
    prefix.push_back((Value)0);
    prefix.push_back(50);

    res = sarma::nicol1d::probe<Ordinal, Value>(prefix, (Ordinal)1, (Value)50);
    EQ(res, true);

    res = sarma::nicol1d::probe<Ordinal, Value>(prefix, (Ordinal)1, (Value)49);
    EQ(res, false);

    res = sarma::nicol1d::probe<Ordinal, Value>(prefix, (Ordinal)2, (Value)49);
    EQ(res, false);

    res = sarma::nicol1d::probe<Ordinal, Value>(prefix, (Ordinal)2, (Value)50);
    EQ(res, true);

    return pass;
}

template<typename Ordinal, typename Value>
int probe1d_allzeros() {
    int pass = 0;

    bool res;

    std::vector<Value> prefix;
    prefix.push_back((Value)0);
    prefix.push_back((Value)0);
    prefix.push_back((Value)0);
    prefix.push_back((Value)0);

    res = sarma::nicol1d::probe<Ordinal, Value>(prefix, (Ordinal)1, (Value)0);
    EQ(res, true);

    res = sarma::nicol1d::probe<Ordinal, Value>(prefix, (Ordinal)1, (Value)1);
    EQ(res, true);

    res = sarma::nicol1d::probe<Ordinal, Value>(prefix, (Ordinal)2, (Value)1);
    EQ(res, true);

    res = sarma::nicol1d::probe<Ordinal, Value>(prefix, (Ordinal)5, (Value)0);
    EQ(res, true);

    return pass;
}

template<typename Ordinal, typename Value>
int probe1d_example1() {
    int pass = 0;

    bool res;

    std::vector<Value> data;
    data.push_back((Value)1);
    data.push_back((Value)2);
    data.push_back((Value)3);
    data.push_back((Value)45);
    data.push_back((Value)12);
    data.push_back((Value)54);
    data.push_back((Value)13);
    data.push_back((Value)99);
    data.push_back((Value)12);

    std::vector<Value> prefix;
    prefix.push_back((Value)0);
    Value sum = 0;
    for (auto it = data.begin(); it<data.end(); it++) {
        sum += *it;
        prefix.push_back(sum);
    }

    res = sarma::nicol1d::probe<Ordinal, Value>(prefix, (Ordinal)4, (Value)98);
    EQ(res, false);

    res = sarma::nicol1d::probe<Ordinal, Value>(prefix, (Ordinal)4, (Value)99);
    EQ(res, true);

    res = sarma::nicol1d::probe<Ordinal, Value>(prefix, (Ordinal)4, (Value)100);
    EQ(res, true);

    res = sarma::nicol1d::probe<Ordinal, Value>(prefix, (Ordinal)2, (Value)124);
    EQ(res, true);

    res = sarma::nicol1d::probe<Ordinal, Value>(prefix, (Ordinal)9, (Value)99);
    EQ(res, true);

    return pass;
}

template<typename Ordinal, typename Value>
int probe1d_example2() {
    int pass = 0;

    bool res;

    std::vector<Value> prefix({
            (Value)0,   (Value) 38, (Value) 58,
            (Value)103, (Value)158, (Value)168,
            (Value)187, (Value)189, (Value)230,
            (Value)309, (Value)369, (Value)455,
            (Value)517, (Value)588});

    res = sarma::nicol1d::probe<Ordinal, Value>(prefix, (Ordinal)4, (Value)588/4);
    EQ(res, false);

    res = sarma::nicol1d::probe<Ordinal, Value>(prefix, (Ordinal)4, (Value)158);
    EQ(res, true);

    res = sarma::nicol1d::probe<Ordinal, Value>(prefix, (Ordinal)14, (Value)78);
    EQ(res, false);

    res = sarma::nicol1d::probe<Ordinal, Value>(prefix, (Ordinal)199, (Value)78);
    EQ(res, false);

    res = sarma::nicol1d::probe<Ordinal, Value>(prefix, (Ordinal)1, (Value)588);
    EQ(res, true);

    res = sarma::nicol1d::probe<Ordinal, Value>(prefix, (Ordinal)2, (Value)309);
    EQ(res, true);

    return pass;
}

template<typename Ordinal, typename Value>
int probe1d_randomnumber( unsigned int seed ) {
    int pass = 0;

    srand( seed );
    std::cout << "    random seed: " << seed << std::endl;

    Ordinal max_n = 2000;
    Ordinal test_times_prefix = 10;
    Ordinal test_times_B = 100;

    for (Ordinal i=0; i<test_times_prefix; i++) {
        Ordinal N = rand() % max_n + 1;
        Ordinal P = (rand() % N) + 1;

        std::vector<Value> prefix;

        prefix.push_back(0);
        for (Ordinal j=1; j<=N; j++) {
            prefix.push_back(prefix[j-1] + (Value)(rand()%(RAND_MAX/N) + (Value)(rand())/RAND_MAX));
        }

        for (Ordinal j=0; j<test_times_B; j++) {
            Value B = (Value)(((int)rand())%((int)(prefix[N])/P*2)) + (Value)(rand())/RAND_MAX;

            bool res;
            res = sarma::nicol1d::probe<Ordinal, Value>(prefix, P, B);
            EQ(res, probe1d_basic(prefix, P, B));
        }
        prefix.clear();
    }

    return pass;
}

template<typename Ordinal, typename Value>
auto get_cuts(std::vector<Value> &prefix, Ordinal P, Value opt_load) {
    Ordinal N = prefix.size() - 1;
    std::vector<Ordinal> cuts;
    cuts.push_back(0);
    Ordinal cut = 1;
    for (Ordinal i = 1; i < P; i++) {
        while (cut <= N && prefix[cut] - prefix[cuts[i - 1]] <= opt_load) cut++;
        cuts.push_back(cut - 1);
        if (cut > N)
            break;
    }
    while ((Ordinal)cuts.size() < P + 1)
        cuts.push_back(N);

    return cuts;
}

template<typename Ordinal, typename Value>
auto brute_force_search_opt_load(std::vector<Value> & array, Ordinal P) {
    std::vector<Ordinal> cuts;

    Ordinal N = array.size();
    if (P == 1) {
        cuts.push_back(0);
        cuts.push_back(N);
        return cuts;
    }
    std::vector<Value> prefix;
    Value sum = 0;
    prefix.push_back(sum);
    for (Ordinal i = 0; i < N; i++)
    {
        sum += array[i];
        prefix.push_back(sum);
    }

    //Note that: for double and float, array_max_element calculated below is not necessary equal to
    //*(std::max_element(array.begin(), array.end())) because of the percision of float storage
    Value array_max_element = 0;
    for (Ordinal i=1; i<= N; i++)
        if (prefix[i] - prefix[i-1] > array_max_element) array_max_element = prefix[i] - prefix[i-1];

    if (P >= N) {
        cuts = get_cuts(prefix, P, array_max_element);
    } else if (P > 1 && P < N) {

        Value low_bound = array_max_element;
        Value opt_load = prefix[N];
        low_bound = std::max(low_bound, opt_load/P);

        if (probe1d_basic(prefix, P, low_bound)) {
            return get_cuts(prefix, P, low_bound);
        }

        for (Ordinal i=0; i<N; i++) {
            for (Ordinal j=i+1; j<=N; j++) {
                Value cur_load = prefix[j] - prefix[i];
                if (cur_load > low_bound && cur_load < opt_load) {
                    if (probe1d_basic(prefix, P, cur_load)) {
                        opt_load = cur_load;
                        break;
                    } else
                        low_bound = cur_load;
                }
            }
        }
        cuts = get_cuts(prefix, P, opt_load);
    }

    return cuts;
}

template<typename Ordinal, typename Value>
int nicol1dpartition_versioncompare(std::vector<Value> & array, Ordinal P, std::vector<Ordinal> & expected_cuts) {
    int pass = 0;
    auto cuts = sarma::nicol1d::partition<Ordinal, Value>(array, P);
    pass |= test_utils::is_same(cuts, expected_cuts);

    std::vector<Value> prefix(array.size()+1, 0);
    std::inclusive_scan(std::execution::par_unseq, array.begin(), array.end(), prefix.begin() + 1);

    cuts = sarma::nicol1d::partition_prefix<Ordinal, Value>(prefix, P);

    pass |= test_utils::is_same(cuts, expected_cuts);

    cuts= sarma::nicol1d::partition<Ordinal, Value, true>(&prefix, 1, P);

    pass |= test_utils::is_same(cuts, expected_cuts);

    if constexpr ( std::is_integral_v<Value> ) {
        cuts = sarma::nicol1d::partition<Ordinal, Value, false>(&prefix, 1, P);
        pass |= test_utils::is_same(cuts, expected_cuts);
    }

    return pass;
}

template<typename Ordinal, typename Value>
int nicol1dPartition_onenumber() {
    int pass = 0;

    std::vector<Value> array({0, 0, 0, 50});
    Ordinal N = array.size();

    std::vector<Ordinal> Ps({1, 2});
    std::vector<Ordinal> expected_cuts[] = {
        {0, N},
        {0, N, N}
    };
    for (Ordinal i=0; i<(Ordinal)Ps.size(); i++)
        pass |= nicol1dpartition_versioncompare<Ordinal, Value>(array, Ps[i], expected_cuts[i]);

    return pass;
}

template<typename Ordinal, typename Value>
int nicol1dPartition_allzeros() {
    int pass = 0;

    std::vector<Value> array({0, 0, 0, 0});
    Ordinal N = array.size();

    std::vector<Ordinal> Ps({1, 2});
    std::vector<Ordinal> expected_cuts[] = {
        {0, N},
        {0, N, N}
    };
    for (Ordinal i=0; i<(Ordinal)Ps.size(); i++)
        pass |= nicol1dpartition_versioncompare(array, Ps[i], expected_cuts[i]);

    return pass;
}

template<typename Ordinal, typename Value>
int nicol1dPartition_example1()  {
    int pass = 0;

    std::vector<Value> array({1, 2, 3, 45, 12, 54, 13, 99, 12});
    Ordinal N = (Ordinal)array.size();

    std::vector<Ordinal> Ps(N);
    for (Ordinal i=1; i<=N; i++) Ps[i-1] = i;
    std::vector<Ordinal> expected_cuts[] = {
        {0, N},
        {0, 6, 9},
        {0, 5, 7, 9},
        {0, 5, 7, 8, 9},
        {0, 5, 7, 8, 9, 9},
        {0, 5, 7, 8, 9, 9, 9},
        {0, 5, 7, 8, 9, 9, 9, 9},
        {0, 5, 7, 8, 9, 9, 9, 9, 9},
        {0, 5, 7, 8, 9, 9, 9, 9, 9, 9}
    };
    for (Ordinal i=0; i<(Ordinal)Ps.size(); i++)
        pass |= nicol1dpartition_versioncompare(array, Ps[i], expected_cuts[i]);

    return pass;
}

template<typename Ordinal, typename Value>
int nicol1dPartition_example2()  {
    int pass = 0;

    std::vector<Value> array({38, 20, 45, 55, 10, 19, 2, 41, 79, 60, 55, 62, 71});
    Ordinal N = array.size();

    for (Ordinal i=1; i<=N; i++) {
        auto expected_cuts = brute_force_search_opt_load(array, i);
        pass |= nicol1dpartition_versioncompare(array, i, expected_cuts);
    }

    return pass;
}

template<typename Ordinal, typename Value>
int nicol1dPartition_example3() {
    int pass = 0;

    std::vector<Value> array({(Value)10.0001, (Value)10.0001, (Value)10.0002, (Value)10.0001});
    Ordinal N = array.size();

    std::vector<Ordinal> Ps({1, 2});
    std::vector<Ordinal> expected_cuts[] = {
        {0, N},
        {0, N/2, N}
    };
    for (Ordinal i=0; i<(Ordinal)Ps.size(); i++)
        pass |= nicol1dpartition_versioncompare(array, Ps[i], expected_cuts[i]);

    return pass;
}

template<typename Ordinal, typename Value>
int nicol1dPartition_randomnumber( unsigned int seed ) {
    int pass = 0;

    srand( seed );
    std::cout << "    random seed: " << seed << std::endl;

    int max_n = 10;
    int test_times_prefix = 10;
    int test_times_P = 10;

    for (int i=0; i<test_times_prefix; i++) {
        Ordinal N = rand() % max_n + 1;

        std::vector<Value> array;
        for (Ordinal j=0; j<N; j++)
            array.push_back((Value)(rand()%(RAND_MAX/N) + (Value)(rand())/RAND_MAX));

        for (Ordinal j=0; j<test_times_P; j++) {
            Ordinal P = (rand() % N) + 1;
            auto expected_cuts = brute_force_search_opt_load(array, P);
            pass |= nicol1dpartition_versioncompare(array,P, expected_cuts);

        }
        if (pass != 0)
            break;
    }

    return pass;
}

int probe1d() {
    int pass = 0;

    TEST(probe1d_onenumber<int COMMA int>)
    TEST(probe1d_onenumber<int COMMA unsigned int>)
    TEST(probe1d_onenumber<int COMMA float>)
    TEST(probe1d_onenumber<int COMMA double>)

    TEST(probe1d_allzeros<int COMMA int>)
    TEST(probe1d_allzeros<int COMMA unsigned int>)
    TEST(probe1d_allzeros<int COMMA float>)
    TEST(probe1d_allzeros<int COMMA double>)

    TEST(probe1d_example1<int COMMA int>)
    TEST(probe1d_example1<int COMMA unsigned int>)
    TEST(probe1d_example1<int COMMA float>)
    TEST(probe1d_example1<int COMMA double>)

    TEST(probe1d_example2<int COMMA int>)
    TEST(probe1d_example2<int COMMA unsigned int>)
    TEST(probe1d_example2<int COMMA float>)
    TEST(probe1d_example2<int COMMA double>)

    TEST(probe1d_randomnumber<int COMMA int>, time(NULL))
    TEST(probe1d_randomnumber<int COMMA unsigned int>, time(NULL))
    TEST(probe1d_randomnumber<int COMMA float>, time(NULL))
    TEST(probe1d_randomnumber<int COMMA double>, time(NULL))

    return pass;
}

int nicol1dPartition() {
    int pass = 0;

    TEST(nicol1dPartition_onenumber<int COMMA int>)
    TEST(nicol1dPartition_onenumber<int COMMA unsigned int>)
    TEST(nicol1dPartition_onenumber<int COMMA float>)
    TEST(nicol1dPartition_onenumber<int COMMA double>)

    TEST(nicol1dPartition_allzeros<int COMMA int>)
    TEST(nicol1dPartition_allzeros<int COMMA unsigned int>)
    TEST(nicol1dPartition_allzeros<int COMMA float>)
    TEST(nicol1dPartition_allzeros<int COMMA double>)

    TEST(nicol1dPartition_example1<int COMMA int>)
    TEST(nicol1dPartition_example1<int COMMA unsigned int>)
    TEST(nicol1dPartition_example1<int COMMA float>)
    TEST(nicol1dPartition_example1<int COMMA double>)

    TEST(nicol1dPartition_example2<int COMMA int>)
    TEST(nicol1dPartition_example2<int COMMA unsigned int>)
    TEST(nicol1dPartition_example2<int COMMA float>)
    TEST(nicol1dPartition_example2<int COMMA double>)

    TEST(nicol1dPartition_example3<int COMMA int>)
    TEST(nicol1dPartition_example3<int COMMA unsigned int>)
    TEST(nicol1dPartition_example3<int COMMA float>)
    TEST(nicol1dPartition_example3<int COMMA double>)

    TEST(nicol1dPartition_randomnumber<int COMMA int>, time(NULL))
    TEST(nicol1dPartition_randomnumber<int COMMA unsigned int>, time(NULL))
    TEST(nicol1dPartition_randomnumber<int COMMA float>, time(NULL))
    TEST(nicol1dPartition_randomnumber<int COMMA double>, time(NULL))

    return pass;
}

int main() {
    int pass = 0;

    TEST(write_read_vector)

    TEST(probe1d)

    TEST(nicol1dPartition)

    return pass;
}
