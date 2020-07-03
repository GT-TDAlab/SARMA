#include <iostream>
#include <mutex>
#include <thread>
#include <random>
#include <string>
#include <memory>

#include "data_structures/csr_matrix.hpp"
#include "data_structures/sparse_prefix_sum.hpp"
#include "tools/timer.hpp"
// #include "tools/utils.hpp"

using namespace std;
using namespace sarma;

template <class sps_t>
auto test(const sps_t &sps, unsigned num_queries) {
	random_device dev;
	mt19937 gen(dev());
	uniform_int_distribution<unsigned> dis(1, sps.size());
	size_t sum = 0;
	while(num_queries--) {
		pair<unsigned, unsigned> upper_left{dis(gen), dis(gen)};
		pair<unsigned, unsigned> lower_right{dis(gen), dis(gen)};
		if(upper_left.first > lower_right.first)
			swap(upper_left.first, lower_right.first);
		if(upper_left.second > lower_right.second)
			swap(upper_left.second, lower_right.second);
		sum += sps.query(upper_left, lower_right);
	}
	return sum;
}

int main(int argc, char *argv[]) {
	ios_base::sync_with_stdio(false);

	{
		vector<unsigned> xadj = {0, 2, 4, 7, 8, 8, 8};
		vector<unsigned> adj = {0, 1, 1, 3, 2, 3, 4, 5};
		vector<unsigned> data;

		Matrix A(move(xadj), move(adj), move(data), 6u);

		sparse_prefix_sum sps(A);

		for (unsigned i = 0; i <= 6; i++) {
			for (unsigned j = 0; j <= 6; j++) {
				cout << sps.query(i, j) << ' ';
			}
			cout << endl;
		}

		for (int i = 2; i <= 6; i++) {
			for (int j = 2; j <= 6; j++) {
				cout << sps.query({2, 2}, {i, j}) << ' ';
			}
			cout << endl;
		}
	}

	if (argc >= 2) {
		double keep_prob = 1;
		if (argc >= 3)
			keep_prob = stod(argv[2]);
		auto num_queries = 1 << 20;
		if (argc >= 4)
			num_queries = stoi(argv[3]);
		sparse_prefix_sum<unsigned, unsigned> sps;
		auto A = Matrix<unsigned, unsigned>(argv[1]);
		{
			timer t("sparse_prefix_sum construction");
			auto B = A.sparsify(keep_prob);
			sps = sparse_prefix_sum(B);
		}
		cerr << sps.query(6000, 6000) << endl;
		{
			timer t("sparse_prefix_sum queries, " + to_string(num_queries) + " times");
			cerr << test(sps, num_queries) << endl;
		}
	}
	
	return 0;
}