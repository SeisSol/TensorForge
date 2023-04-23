from gemmforge import DenseMatrix, GenerationError, GemmGenerator, SparseMatrix
from gemmforge.vm import vm_factory
import numpy as np
import sys
from random import randint

b_matrix_types = ["band", "one_col_b", "one_row_b", "random_entries"]


def gen_matrix_b(rowB, colB, transposed, btype):
    B = np.zeros([rowB, colB])
    coo = {"name": "B", "rows": rowB, "cols": colB, "entries": [], "coordinates": []}

    if btype == "band":
        if not transposed:
            coo["entries"].append([0, 0, 2.0])
            coo["coordinates"].append([0, 0])
            coo["entries"].append([1, 0, 1.0])
            coo["coordinates"].append([1, 0])
            for i in range(1, rowB - 1):
                coo["entries"].append([i-1, i, 3.0])
                coo["coordinates"].append([i-1, i])
                coo["entries"].append([i, i, 2.0])
                coo["coordinates"].append([i, i])
                coo["entries"].append([i+1, i, 1.0])
                coo["coordinates"].append([i+1, i])
            i = rowB - 1
            coo["entries"].append([i-1, i, 3.0])
            coo["coordinates"].append([i-1, i])
            coo["entries"].append([i, i, 2.0])
            coo["coordinates"].append([i, i])
        else:
            coo["entries"].append([0, 0, 2.0])
            coo["coordinates"].append([0, 0])
            coo["entries"].append([0, 1, 3.0])
            coo["coordinates"].append([0, 1])
            for i in range(1, rowB - 1):
                coo["entries"].append([i, i-1, 1.0])
                coo["coordinates"].append([i, i-1])
                coo["entries"].append([i, i, 2.0])
                coo["coordinates"].append([i, i])
                coo["entries"].append([i, i+1, 3.0])
                coo["coordinates"].append([i, i+1])
            i = rowB - 1
            coo["entries"].append([i, i-1, 1.0])
            coo["coordinates"].append([i, i-1])
            coo["entries"].append([i, i, 2.0])
            coo["coordinates"].append([i, i])

        for i in range(rowB):
            B[i, i] = 2.0
        for i in range(rowB - 1):
            B[i, i + 1] = 3.0
        for i in range(1, rowB):
            B[i, i - 1] = 1.0
    elif btype == "one_col_b":
        if colB < 2:
            at = 0
        else:
            at = 1
        for i in range(rowB):
            B[i, at] = 4.0
        for i in range(rowB):
            coo["entries"].append([i, at, 4.0])
            coo["coordinates"].append([i, at])
    elif btype == "one_row_b":
        if rowB < 2:
            at = 0
        else:
            at = 1
        for j in range(colB):
            B[at, j] = 5.0
        coo = {"name": "B", "rows": rowB, "cols": colB, "entries": [], "coordinates": []}
        for j in range(rowB):
            coo["entries"].append([at, j, 4.0])
            coo["coordinates"].append([at, j])
    elif btype == "random_entries":
        nonzeros = set()
        while len(nonzeros) < int(np.sqrt(rowB * colB)):
            nonzeros.add(randint(0, rowB * colB - 1))

        B_nonzeros = []
        for el in nonzeros:
            row = el // rowB
            col = el % colB
            coo["entries"].append([row, col, "9.0"])
            coo["coordinates"].append([row, col])
            B[row, col] = 9.0
            B_nonzeros.append(9.0)
    else:
        raise Exception("NO")
    if btype != "random_entries":
        if transposed:
            B = B.flatten("C")
        else:
            B = B.flatten("F")
        B_nonzeros = []
        for el in B:
            if el != 0.0:
                B_nonzeros.append(el)
    else:
        B_nonzeros = []
    return (coo, B, B_nonzeros)


try:
    for b_type in b_matrix_types:
        for tA in [False]:
            for tB in [False, True]:
                testid = ""
                if tA:
                    testid += "At_mul_"
                else:
                    testid += "A_mul_"
                if tB:
                    testid += "Bt"
                else:
                    testid += "B"
                testid += "_" + b_type

                rowA = 56
                colA = 9
                rowB = 9
                colB = 9
                rowC = 56
                colC = 9

                mat_a = DenseMatrix(num_rows=rowA,
                                    num_cols=colA,
                                    addressing="strided",
                                    bbox=[0, 0, rowA, colA])

                coo, matrix_b, matrix_b_non_zeros_flat = gen_matrix_b(rowB, colB, tB, b_type)

                mat_b_sparse = SparseMatrix(num_rows=rowB,
                                            num_cols=colB,
                                            addressing="strided",
                                            coordinates=coo["coordinates"],
                                            values=None)

                mat_b_dense = DenseMatrix(num_rows=rowB,
                                          num_cols=colB,
                                          bbox=[0, 0, rowB, colB],
                                          addressing="strided")

                mat_c = DenseMatrix(num_rows=rowC,
                                    num_cols=colC,
                                    bbox=[0, 0, rowC, colC],
                                    addressing="strided")

                vm = vm_factory(arch="sm_86", backend="cuda", fp_type="float")

                if tA:
                    transA = "Transposed"
                else:
                    transA = ""
                if tB:
                    transB = "Transposed"
                else:
                    transB = ""

                dense_gen = GemmGenerator(vm)
                dense_gen.set(tA, tB, mat_a, mat_b_dense, mat_c, alpha=1.0, beta=1.0)
                dense_gen.generate()
                # print(dense_gen.get_kernel())
                # print(dense_gen.get_launcher())
                # print(dense_gen.get_launcher_header())
                dense_header = dense_gen.get_launcher_header()
                # Get the function name without void in the beginning
                dense_function_name = dense_header.split("(")[0][4:]

                sparse_gen = GemmGenerator(vm)
                sparse_gen.set(tA, tB, mat_a, mat_b_sparse, mat_c, alpha=1.0, beta=1.0)
                sparse_gen.generate()
                # print(sparse_gen.get_kernel())
                # print(sparse_gen.get_launcher())
                # print(sparse_gen.get_launcher_header())
                sparse_header = sparse_gen.get_launcher_header()
                # Get the function name without void in the beginning
                sparse_function_name = sparse_header.split("(")[0][4:]

                # A = np.random.random({rowA} * 9)
                # B = np.random.random(9 * 9)
                C = np.empty(rowC * colC)
                C.fill(0.1)
                A = np.empty(rowA * colA)
                A.fill(1.0)
                B_dense = matrix_b
                B_sparse = matrix_b_non_zeros_flat

                np.set_printoptions(threshold=sys.maxsize)
                strA = np.array2string(A, separator=', ').replace("[", "{").replace("]", "}")
                strB_sparse = np.array2string(np.array(B_sparse), separator=', ').replace("[", "{").replace("]", "}")
                strB_dense = np.array2string(B_dense, separator=', ').replace("[", "{").replace("]", "}")
                strC = np.array2string(C, separator=', ').replace("[", "{").replace("]", "}")

                s = f"""
#include <iostream>
#include <cuda.h>
#include <cstring>

#define CHECK_ERR checkErr(__FILE__,__LINE__)

#define CHECK_CUDA_ERROR(val) check((val), #val, __FILE__, __LINE__)
template <typename T>
void check(T err, const char* const func, const char* const file,
           const int line)
{{
    if (err != cudaSuccess)
    {{
        std::cerr << "CUDA Runtime Error at: " << file << ":" << line
                  << std::endl;
        std::cerr << cudaGetErrorString(err) << " " << func << std::endl;
        // We don't exit when we encounter CUDA errors in this example.
        // std::exit(EXIT_FAILURE);
    }}
}}

std::string PrevFile = "";
int PrevLine = 0;

void checkErr(const std::string &File, int Line) {{
#ifndef NDEBUG
      cudaError_t Error = cudaGetLastError();
      if (Error != cudaSuccess) {{
        std::cout << std::endl << File
                  << ", line " << Line
                  << ": " << cudaGetErrorString(Error)
                  << " (" << Error << ")"
                  << std::endl;

        if (PrevLine > 0)
          std::cout << "Previous CUDA call:" << std::endl
                    << PrevFile << ", line " << PrevLine << std::endl;
        throw;
      }}
      PrevFile = File;
      PrevLine = Line;
#endif
}}

// Dense x Dense Kernel
{dense_gen.get_kernel()}

// Dense x Sparse Kernel
{sparse_gen.get_kernel()}

// Dense x Dense Kernel Launcher
{dense_gen.get_launcher()}

// Dense x Sparse Kernel Launcher
{sparse_gen.get_launcher()}


int main(){{
  float A[{rowA}*{colA}] = {strA};
  float B_sparse[{len(B_sparse)}] = {strB_sparse};
  float B_dense[{rowB} * {colB}] = {strB_dense};
  float C[{rowC}*{colC}] = {strC};
  float R1[{rowC}*{colC}];
  float R2[{rowC}*{colC}];

  float *A_dev = nullptr;
  float *B_sparse_dev = nullptr;
  float *B_dense_dev = nullptr;
  float *C1_dev = nullptr;
  float *C2_dev = nullptr;

  cudaMalloc((void **)&A_dev, sizeof(float) * {rowA} * {colA}); CHECK_ERR;
  cudaMalloc((void **)&B_sparse_dev, sizeof(float) * {len(B_sparse)}); CHECK_ERR;
  cudaMalloc((void **)&B_dense_dev, sizeof(float) * {rowB} * {colB}); CHECK_ERR;
  cudaMalloc((void **)&C1_dev, sizeof(float) * {rowC} * {colC}); CHECK_ERR;
  cudaMalloc((void **)&C2_dev, sizeof(float) * {rowC} * {colC}); CHECK_ERR;

  cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * {rowA} * {colA}, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_sparse_dev, (void *)B_sparse, sizeof(float) *  {len(B_sparse)}, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)B_dense_dev, (void *)B_dense, sizeof(float) *  {rowB} * {colB}, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C1_dev, (void *)C, sizeof(float) * {rowC} * {colC}, cudaMemcpyHostToDevice); CHECK_ERR;
  cudaMemcpy((void *)C2_dev, (void *)C, sizeof(float) * {rowC} * {colC}, cudaMemcpyHostToDevice); CHECK_ERR;

  // Dense x Dense Matrix Mult
  {dense_function_name}(A_dev, 0, B_dense_dev, 0, C1_dev, 0, 1, nullptr, nullptr);
  cudaDeviceSynchronize();
  cudaMemcpy(R1, C1_dev, sizeof(float)*{rowC}*{colC}, cudaMemcpyDeviceToHost);

  // Dense x Sparse Matrix Mult
  {sparse_function_name}(A_dev, 0, B_sparse_dev, 0, C2_dev, 0, 1, nullptr, nullptr);
  cudaDeviceSynchronize();
  cudaMemcpy(R2, C2_dev, sizeof(float)*{rowC}*{colC}, cudaMemcpyDeviceToHost);

  std::cout << "[";
  for (int ii = 0; ii < {rowC}*{colC} -1; ii++){{
    std::cout << R1[ii] << ", ";
  }}
  std::cout << R1[{rowC}*{colC} -1] << "]" << std::endl;
  std::cout << "[";
  for (int ii = 0; ii < {rowC}*{colC} - 1; ii++){{
    std::cout << R2[ii] << ", ";
  }}
  std::cout << R2[{rowC}*{colC} -1] << "]" << std::endl;
  for (int i = 0; i < {rowC}*{colC}; i++){{
    if (R1[i] != R2[i]) {{
    throw std::runtime_error("{transA} Dense x {transB} Dense and {transA} Dense x {transB} Sparse Matrix Mismatch in Multiplication!");
    }}
  }}
  std::cout << "{transA} Dense x {transB} Dense and {transA} Dense x {transB} Sparse Matrix Multiplications Match!" << std::endl;
}}
"""
                f = open(f"test_cuda_{testid}.cu", "w")
                f.write(s)
                f.close()
                # print(s)
except GenerationError as err:
    print(f'ERROR: {err}')
    raise err
