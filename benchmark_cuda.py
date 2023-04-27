from gemmforge import DenseMatrix, GenerationError, GemmGenerator, SparseMatrix
from gemmforge.vm import vm_factory
import numpy as np
import sys
from random import randint
from numba import cuda

b_matrix_types = ["band", "single_column_b", "single_row_b", "chequered"]

def get_available_mem_on_gpu():
    gpus = cuda.gpus.lst

    #for gpu in gpus:
    gpu = gpus[0]
    meminfo = cuda.current_context().get_memory_info()
    # print("%s, free: %s bytes, total, %s bytes" % (gpu, meminfo[0], meminfo[1]))
    return meminfo[0]

def get_suggested_num_elements(MatASize, MatBDenseSize, MatBSparseSize, MatCSize, SizeOfFloat):
    # We mul A x BD(dense) = C1, A x BS(Sparse) = C2
    # And compare C1 and C1, C1 and 2 obtained back will be R1 and R2 on host
    # On host we need A, BD, BS, C, R1, R2
    # On device we need A, BD, BS, C1, C2
    per_el_size = (MatASize + MatBDenseSize + MatBSparseSize + MatCSize*2) * SizeOfFloat

    available_mem = get_available_mem_on_gpu()
    can_fit_els = available_mem // per_el_size
    at80 = int(0.8 * can_fit_els)
    # print(f"Can fit {can_fit_els} matrices of given sizes, at 80% capacity {at80}")
    return (can_fit_els, at80)
    #return (1,1)

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
        b_el_count = 2*2 + 3*(rowB-2)
    elif btype == "single_column_b":
        at = 1
        for i in range(rowB):
            B[i, at] = i + 1.0
        for i in range(rowB):
            coo["entries"].append([i, at, i + 1.0])
            coo["coordinates"].append([i, at])
        b_el_count = rowB
    elif btype == "single_row_b":
        at = 1
        for j in range(colB):
            B[at, j] = j + 1.0
        for j in range(colB):
            coo["entries"].append([at, j, j + 1.0])
            coo["coordinates"].append([at, j])
        b_el_count = colB
    elif btype == "chequered":
        npB = np.zeros((rowB,colB))
        if transposed:
            for i in range(rowB):
                offset = i % 2
                for j in range(offset, colB, 2):
                    coo["entries"].append([i, j, i*10.0 + j +1])
                    coo["coordinates"].append([i, j])
                    npB[i, j] = i*10.0 + j +1
                    B[i, j] = i*10.0 + j +1
        else:
            for j in range(colB):
                offset = j % 2
                for i in range(offset, rowB, 2):
                    coo["entries"].append([i, j, i*10.0 + j +1])
                    coo["coordinates"].append([i, j])
                    npB[i, j] = i*10.0 + j +1 
                    B[i, j] = i*10.0 + j +1
        b_el_count = len(coo["coordinates"])
    else:
        raise Exception("NO")
    if btype != "random_entries":
        if transposed:
            Bo = B
            B = B.flatten("C")
        else:
            Bo = B
            B = B.flatten("F")
        T = "T"
        NT = ""
        print(btype, f"{T if transposed else NT}: ", coo["coordinates"])
        print(btype, f"{T if transposed else NT}: ", Bo)
        B_nonzeros = []
        for el in B:
            if el != 0.0:
                B_nonzeros.append(el)
        print(btype, f"{T if transposed else NT} sparse: ", B_nonzeros)
    else:
        B_nonzeros = []
    return (coo, B, B_nonzeros, b_el_count)


try:
    for with_compile_time_values in [True, False]:
        for b_type in b_matrix_types:
            for tA in [False]:
                for tB in [True, False]:
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
                    valid = "_compiler_time_value" if with_compile_time_values else ""
                    testid += valid

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

                    coo, matrix_b, matrix_b_non_zeros_flat, b_el_count = gen_matrix_b(rowB, colB, tB, b_type)

                    mat_b_sparse = SparseMatrix(num_rows=rowB,
                                                num_cols=colB,
                                                addressing="strided",
                                                coordinates=coo["coordinates"],
                                                values=matrix_b_non_zeros_flat)

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
                    C = np.zeros(rowC * colC)
                    C.fill(0.1)
                    for i in range(rowC * colC):
                        C[i] = i * 0.1
                    A = np.zeros(rowA * colA)
                    A.fill(1.0)
                    for i in range(rowA * colA):
                        A[i] = i * 2.0
                    B_dense = matrix_b
                    B_sparse = matrix_b_non_zeros_flat

                    np.set_printoptions(threshold=sys.maxsize)
                    strA = np.array2string(A, separator=', ').replace("[", "{").replace("]", "}")
                    strB_sparse = np.array2string(np.array(B_sparse), separator=', ').replace("[", "{").replace("]", "}")
                    strB_dense = np.array2string(B_dense, separator=', ').replace("[", "{").replace("]", "}")
                    strC = np.array2string(C, separator=', ').replace("[", "{").replace("]", "}")

                    get_available_mem_on_gpu()
                    full, at80 = get_suggested_num_elements(rowA*colA, rowB*colB, b_el_count, rowC*colC, 4)
                    num_els = at80

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
    // Element Matrices
    std::cout << "Instantiating core matrices" << std::endl;
    float CoreA[{rowA}*{colA}] = {strA};
    float CoreB_sparse[{b_el_count}] = {strB_sparse};
    float CoreB_dense[{rowB} * {colB}] = {strB_dense};
    float CoreC[{rowC}*{colC}] = {strC};
    
    // Buffers 
    std::cout << "Instantiating buffer matrices" << std::endl;
    float* A = new float[{rowA}*{colA}*{num_els}];
    float* B_dense = new float[{rowB}*{colB}*{num_els}];
    {f"float* B_sparse = new float[{b_el_count}*{num_els}];" if not with_compile_time_values else ""}
    float* C = new float[{rowC}*{colC}*{num_els}];
    float* R1 = new float[{rowC}*{colC}*{num_els}];
    float* R2 = new float[{rowC}*{colC}*{num_els}];

    // Copy the Element Matrices N times into Element Buffers
    std::cout << "Copying core matrices to buffers" << std::endl;
    for (int i = 0; i < {num_els}; i++){{
        std::memcpy(&A[{rowA} * {colA} * i], &CoreA[0], {rowA} * {colA} * sizeof(float));
        std::memcpy(&B_dense[{rowB} * {colB} * i], &CoreB_dense[0], {rowB} * {colB} * sizeof(float));
        {f"std::memcpy(&B_sparse[{b_el_count} * i], &CoreB_sparse[0], {b_el_count} * sizeof(float));" if not with_compile_time_values else ""}
        std::memcpy(&C[{rowC} * {colC} * i], &CoreC[0], {rowC} * {colC} * sizeof(float));
    }}

    float *A_dev = nullptr;
    {"float *B_sparse_dev = nullptr;" if not with_compile_time_values else ""}
    float *B_dense_dev = nullptr;
    float *C1_dev = nullptr;
    float *C2_dev = nullptr;

    std::cout << "Allocating device memory" << std::endl;
    cudaMalloc((void **)&A_dev, sizeof(float) * {rowA} * {colA} * {num_els}); CHECK_ERR;
    {f"cudaMalloc((void **)&B_sparse_dev, sizeof(float) * {b_el_count} * {num_els}); CHECK_ERR;" if not with_compile_time_values else ""}
    cudaMalloc((void **)&B_dense_dev, sizeof(float) * {rowB} * {colB} * {num_els}); CHECK_ERR;
    cudaMalloc((void **)&C1_dev, sizeof(float) * {rowC} * {colC} * {num_els}); CHECK_ERR;
    cudaMalloc((void **)&C2_dev, sizeof(float) * {rowC} * {colC} * {num_els}); CHECK_ERR;

    std::cout << "Copying buffers to device" << std::endl;
    cudaMemcpy((void *)A_dev, (void *)A, sizeof(float) * {rowA} * {colA} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    {f"cudaMemcpy((void *)B_sparse_dev, (void *)B_sparse, sizeof(float) *  {b_el_count} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;" if not with_compile_time_values else ""}
    cudaMemcpy((void *)B_dense_dev, (void *)B_dense, sizeof(float) *  {rowB} * {colB} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)C1_dev, (void *)C, sizeof(float) * {rowC} * {colC} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;
    cudaMemcpy((void *)C2_dev, (void *)C, sizeof(float) * {rowC} * {colC} * {num_els}, cudaMemcpyHostToDevice); CHECK_ERR;

    // Dense x Dense Matrix Mult
    std::cout << "Calling Dense x Dense kernel" << std::endl;
    float elapsedTime = 0.0; 
    cudaEvent_t startDD, stopDD;
    cudaEventCreate(&startDD);
    cudaEventCreate(&stopDD);
    cudaEventRecord(startDD);
    {dense_function_name}(A_dev, 0, B_dense_dev, 0, C1_dev, 0, {num_els}, nullptr, nullptr);
    cudaEventRecord(stopDD);
    cudaEventSynchronize(stopDD);
    cudaEventElapsedTime(&elapsedTime, startDD, stopDD);
    std::cout << "Dense x Dense kernel took " << elapsedTime << " ms" << std::endl; 
    cudaDeviceSynchronize();
    cudaMemcpy(R1, C1_dev, sizeof(float)*{rowC} * {colC} * {num_els}, cudaMemcpyDeviceToHost);

    // Dense x Sparse Matrix Mult
    std::cout << "Calling Dense x Sparse kernel" << std::endl;
    elapsedTime = 0.0;
    cudaEvent_t startDS, stopDS;
    cudaEventCreate(&startDS);
    cudaEventCreate(&stopDS);
    cudaEventRecord(startDS);
    {f"{sparse_function_name}(A_dev, 0, B_sparse_dev, 0, C2_dev, 0, {num_els}, nullptr, nullptr);" if not with_compile_time_values else f"{sparse_function_name}(A_dev, 0, nullptr, 0, C2_dev, 0, {num_els}, nullptr, nullptr);"}
    cudaEventRecord(stopDS);
    cudaEventSynchronize(stopDS);
    cudaEventElapsedTime(&elapsedTime, startDS, stopDS);
    std::cout << "Dense x Sparse kernel took " << elapsedTime << " ms" << std::endl; 
    cudaDeviceSynchronize();
    cudaMemcpy(R2, C2_dev, sizeof(float)*{rowC} * {colC} * {num_els}, cudaMemcpyDeviceToHost);

    std::cout << "Freeing device memory" << std::endl;
    cudaFree(A_dev);
    {f"cudaFree(B_sparse_dev);" if not with_compile_time_values else ""}
    cudaFree(B_dense_dev);
    cudaFree(C1_dev);
    cudaFree(C2_dev);

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
    for (int i = 0; i < {rowC}*{colC}*{num_els}; i++){{
        if (R1[i] != R2[i]) {{
        //throw std::runtime_error("{transA} Dense x {transB} Dense and {transA} Dense x {transB} Sparse Matrix Mismatch in Multiplication at " + std::to_string(i) + "!");
        std::cout << "RESULTS DONT MATCH" << std::endl;
        return 0;
        }}
    }}
    std::cout << "{transA} Dense x {transB} Dense and {transA} Dense x {transB} Sparse Matrix Multiplications Match!" << std::endl;
    std::cout << "Results Match!" << std::endl;
    }}
    """
                    f = open(f"benchmark_cuda_{testid}.cu", "w")
                    f.write(s)
                    f.close()
                    # print(s)
except GenerationError as err:
    print(f'ERROR: {err}')
    raise err
