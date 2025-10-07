// UTSP.cpp  — minimal, batch-friendly front-end around authors' solver
#include "include/TSP_IO.h"
#include "include/TSP_Basic_Functions.h"
#include "include/TSP_Init.h"
#include "include/TSP_2Opt.h"
#include "include/TSP_MCTS.h"
#include "include/TSP_Markov_Decision.h"
#include "include/TSP_sym.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

// CLI:
//   argv[1]  = stats_txt (output)
//   argv[2]  = input_txt (our batch file)
//   argv[3]  = n (Temp_City_Num)
//   argv[4]  = use_rec (0/1)
//   argv[5]  = rec_only (0/1)
//   argv[6]  = M  (Max_Candidate_Num and Rec_Num)
//   argv[7]  = Max_Depth
//   argv[8]  = Alpha
//   argv[9]  = Beta
//   argv[10] = Param_H
//   argv[11] = restart
//   argv[12] = restart_reconly
//   argv[13] = total_instances

static inline void log_instance(FILE* fp, int idx1, int n, double mcts_dist, double sec) {
    // Only the metrics we care about; no Concorde, no GT
    fprintf(fp, "Inst_Index:%d  City_Num:%d  MCTS:%f  Time:%.2f\n", idx1, n, mcts_dist, sec);
}

static bool Read_Instances_From_File(const char* input_path,
                                     int n, int M, int total_instances) {
    std::ifstream FIC(input_path);
    if (FIC.fail()) {
        std::cerr << "Error: cannot open input file: " << input_path << std::endl;
        return false;
    }

    // The authors' globals are *fixed-size arrays*. We just fill the first
    // [total_instances][n] slots safely; NO resize/clear on the first dim.
    // Types (from headers):
    //   Stored_Coordinates_X : double[Max_Inst][Max_City]
    //   Stored_Coordinates_Y : double[Max_Inst][Max_City]
    //   Stored_Rec           : std::vector<int>[Max_Inst][Max_City]
    //   Stored_Rec_Value     : std::vector<double>[Max_Inst][Max_City]
    //   Sparse_Stored_Rec        and _Value similarly.

    if (total_instances > Max_Inst_Num) {
        std::cerr << "Error: total_instances exceeds compiled maximum\n";
        return false;
    }
    if (n > Max_City_Num) {
        std::cerr << "Error: n exceeds compiled Max_City_Num\n";
        return false;
    }

    Temp_City_Num = n;
    Rec_Num = M;                   // how many recs per city we read from file
    Total_Instance_Num = total_instances;

    double temp_x, temp_y;
    int temp_int;
    std::string label;

    for (int i = 0; i < total_instances; ++i) {
        // coordinates
        for (int j = 0; j < n; ++j) {
            FIC >> temp_x >> temp_y;
            Stored_Coordinates_X[i][j] = temp_x;
            Stored_Coordinates_Y[i][j] = temp_y;
        }

        // consume "OPT_TOUR:" and the dummy tour line (n ints); ignore
        FIC >> label; // "OPT_TOUR:"
        for (int j = 0; j < n; ++j) FIC >> temp_int;

        // consume the extra single int (ignored by our solver)
        FIC >> temp_int;

        // REC_INDEX:
        FIC >> label; // "REC_INDEX:"
        for (int j = 0; j < n; ++j) {
            Sparse_Stored_Rec[i][j].clear();
            Sparse_Stored_Rec[i][j].reserve(M);
            for (int k = 0; k < M; ++k) {
                int idx1; FIC >> idx1;
                Sparse_Stored_Rec[i][j].push_back(idx1 - 1); // 0-based
            }
        }

        // REC_VALUE:
        FIC >> label; // "REC_VALUE:"
        for (int j = 0; j < n; ++j) {
            Sparse_Stored_Rec_Value[i][j].clear();
            Sparse_Stored_Rec_Value[i][j].reserve(M);
            for (int k = 0; k < M; ++k) {
                double v; FIC >> v;
                Sparse_Stored_Rec_Value[i][j].push_back(v);
            }
        }

        // Dense rec-value matrix: size n per row
        for (int j = 0; j < n; ++j) {
            Stored_Rec_Value[i][j].assign(n, 0.0);
        }
        for (int j = 0; j < n; ++j) {
            const auto& idxs = Sparse_Stored_Rec[i][j];
            const auto& vals = Sparse_Stored_Rec_Value[i][j];
            for (size_t k = 0; k < idxs.size(); ++k) {
                int l = idxs[k];
                if (l >= 0 && l < n) Stored_Rec_Value[i][j][l] = vals[k];
            }
        }

        // H' = H + H^T
        symmetrizeMatrix(Stored_Rec_Value[i], n);

        // Candidate set indices per city j: 0..n-1
        for (int j = 0; j < n; ++j) {
            Stored_Rec[i][j].clear();
            Stored_Rec[i][j].reserve(n);
            for (int m = 0; m < n; ++m) Stored_Rec[i][j].push_back(m);
        }
    }

    FIC.close();
    return true;
}

static void Solve_One_Instance_And_Log(int inst_idx_0b, int n, FILE* fp) {
    Current_Instance_Begin_Time = (double)clock();
    Current_Instance_Best_Distance = Inf_Cost;

    Fetch_Stored_Instance_Info(inst_idx_0b);
    Calculate_All_Pair_Distance();
    Identify_Candidate_Set();
    Markov_Decision_Process(inst_idx_0b);

    double d_mcts = Get_Current_Solution_Double_Distance() / Magnify_Rate;
    double t_sec  = ((double)clock() - Current_Instance_Begin_Time) / CLOCKS_PER_SEC;

    log_instance(fp, inst_idx_0b + 1, n, d_mcts, t_sec);

    // Tour (1-based ids)
    fprintf(fp, "Solution: ");
    int cur = Start_City;
    if (cur != Null) {
        do {
            fprintf(fp, "%d ", cur + 1);
            cur = All_Node[cur].Next_City;
        } while (cur != Null && cur != Start_City);
    }
    fprintf(fp, "\n");
}

int main(int argc, char** argv) {
    if (argc < 14) {
        std::cerr
            << "Usage:\n  " << argv[0]
            << " <stats_txt> <input_txt> <n> <use_rec> <rec_only> <M> <Max_Depth>"
               " <Alpha> <Beta> <Param_H> <restart> <restart_reconly> <total_instances>\n";
        return 1;
    }

    const char* stats_path = argv[1];
    const char* input_path = argv[2];
    int n                  = atoi(argv[3]);
    use_rec                = atoi(argv[4]);
    rec_only               = atoi(argv[5]);
    Max_Candidate_Num      = atoi(argv[6]); // M
    Max_Depth              = atoi(argv[7]);
    Alpha                  = atof(argv[8]);
    Beta                   = atof(argv[9]);
    Param_H                = atof(argv[10]);
    restart                = atoi(argv[11]);
    restart_reconly        = atoi(argv[12]);
    const int total_instances = atoi(argv[13]);

    // Keep the internal "Rec_Num" aligned with M for the reader.
    Rec_Num = Max_Candidate_Num;

    srand((unsigned)time(NULL));

    if (!Read_Instances_From_File(input_path, n, Max_Candidate_Num, total_instances)) {
        return 2;
    }

    FILE* fp = fopen(stats_path, "w+");
    if (!fp) {
        std::cerr << "Error: cannot open stats file: " << stats_path << std::endl;
        return 3;
    }

    fprintf(fp, "Number_of_Instances: %d\n", Total_Instance_Num);

    double begin_time = (double)clock();
    double sum_mcts = 0.0;

    for (int i = 0; i < Total_Instance_Num; ++i) {
        Solve_One_Instance_And_Log(i, n, fp);
        sum_mcts += Get_Current_Solution_Double_Distance() / Magnify_Rate;
        Release_Memory(n);
    }

    double total_sec = ((double)clock() - begin_time) / CLOCKS_PER_SEC;
    double avg_mcts = (Total_Instance_Num > 0) ? (sum_mcts / Total_Instance_Num) : 0.0;

    fprintf(fp, "Summary: Instances:%d  Avg_MCTS:%f  Total_Time:%.2f\n",
            Total_Instance_Num, avg_mcts, total_sec);

    fclose(fp);
    return 0;
}
