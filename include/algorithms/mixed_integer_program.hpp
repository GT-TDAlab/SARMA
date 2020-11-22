#pragma once

#include <vector>
#include <cstdlib>
#include <string>
#include <utility>
#include <cmath>

#include "gurobi_c++.h"

#include "data_structures/csr_matrix.hpp"
#include "tools/timer.hpp"

/**
* @brief To the best of our knowledge, this is the first work that tackles the symmetric
* rectilinear partitioning problem. Symmetric rectilinear partitioning is a
* restricted problem, therefore comparing our algorithms with more relaxed
* partitioning algorithms (such as jagged, rectilinear etc.) doesn't provide
* enough information about the quality of the found partition vectors. Hence,
* we implemented a mathematical model that finds the optimal solution using
* Gurobi. This namespace contains required functions.
* @note This mixed integer program implementation can also handle non-symmetric partitioning.
**/
namespace sarma::mixed_integer_program {

    /**
    * @brief Implements the mixed integer program solver for rectilinear partitioning
    * @param A the input matrix
    * @param P number of parts
    * @param Q_ number of parts in the other dimension for when template parameter symmetric is false
    * @return a cut vector if symmetric and returns a pair of vectors if symmetric is false
    */
    template <bool symmetric, typename Int, typename Value>
    auto partition(const Matrix<Int, Value> &A, const Int P, const Int Q_ = 0) {
        const auto Q = symmetric ? P : Q_;
        try {
            auto env = GRBEnv(true);
            // env.set("LogFile", "mip1.log");
            env.start();

            auto model = GRBModel(env);

            std::vector<GRBVar> p, q_original;
            auto &q = symmetric ? p : q_original;

            for (Int i = 0; i <= P; i++)
                p.emplace_back(model.addVar(0.0, A.N(), 0.0, GRB_INTEGER, "p_" + std::to_string(i)));
            
            model.addConstr(p[0] <= 0);
            
            for (Int i = 0; i < P; i++)
                model.addConstr(p[i] <= p[i + 1]);
            
            model.addConstr(p[P] >= A.N());

            if constexpr (!symmetric) {
                for(Int i = 0; i <= Q; i++)
                    q.emplace_back(model.addVar(0.0, A.M, 0.0, GRB_INTEGER, "q_" + std::to_string(i)));
                
                model.addConstr(q[0] <= 0);
            
                for (Int i = 0; i < Q; i++)
                    model.addConstr(q[i] <= q[i + 1]);
                
                model.addConstr(q[Q] >= A.M);
            }

            std::vector<std::vector<GRBVar>> I(P + 1);

            for (Int i = 0; i <= P; i++)
                for (Int u = 0; u < A.N(); u++) {
                    auto z = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);

                    model.addConstr(z * A.N() >= u - p[i] + 1);
                    model.addConstr((1 - z) * A.N() >= p[i] - u);

                    I[i].emplace_back(z);
                }
            
            std::vector<std::vector<GRBVar>> J_original(Q + 1);
            auto &J = symmetric ? I : J_original;
            
            if constexpr (!symmetric) {
                for (Int j = 0; j <= Q; j++)
                    for (Int v = 0; v < A.M; v++) {
                        auto z = model.addVar(0.0, 1.0, 0.0, GRB_BINARY);

                        model.addConstr(z * A.M >= v - q[j] + 1);
                        model.addConstr((1 - z) * A.M >= q[j] - v);

                        J[j].emplace_back(z);
                    }
            }
            
            std::vector<std::vector<GRBLinExpr>> L(P, std::vector<GRBLinExpr>(Q, 0));
            
            for (Int i = 0; i < P; i++)
                for (Int j = 0; j < Q; j++)
                    for (Int u = 0; u < A.N(); u++)
                        for (auto k = A.indptr[u]; k < A.indptr[u + 1]; k++) {
                            const auto v = A.indices[k];

                            auto x = model.addVar(0.0, 1.0, 0.0, GRB_CONTINUOUS);
                            model.addConstr(x + 1 >= I[i][u] - I[i + 1][u] + J[j][v] - J[j + 1][v]);
                            L[i][j] += x * A.data(k);
                        }

            auto L_max = model.addVar(0.0, A.total_load(), 0.0, GRB_CONTINUOUS, "L_max");

            for (Int i = 0; i < P; i++)
                for (Int j = 0; j < Q; j++)
                    model.addConstr(L_max >= L[i][j]);

            model.setObjective(L_max * 1, GRB_MINIMIZE);
            
            model.optimize();

            std::vector<Int> cuts;
            for(Int i = 0; i <= P; i++) {
                std::cerr << p[i].get(GRB_DoubleAttr_X) << ' ';
                cuts.push_back(std::lround(p[i].get(GRB_DoubleAttr_X)));
            }
            std::cerr << std::endl;

            if constexpr (symmetric)
                return cuts;
            else {
                std::vector<Int> cuts2;
                for(Int i = 0; i <= Q; i++) {
                    std::cerr << q[i].get(GRB_DoubleAttr_X) << ' ';
                    cuts2.push_back(std::lround(q[i].get(GRB_DoubleAttr_X)));
                }
                std::cerr << std::endl;

                return std::make_pair(cuts, cuts2);
            }
        } catch(GRBException e) {
            std::cerr << "Error code = " << e.getErrorCode() << std::endl;
            std::cerr << e.getMessage() << std::endl;
        } catch(...) {
            std::cerr << "Exception during optimization" << std::endl;
        }

        if constexpr (symmetric)
            return std::vector<Int>(P + 1, 0);
        else
            return std::make_pair(std::vector<Int>(P + 1, 0), std::vector<Int>(Q + 1, 0));
    }

}