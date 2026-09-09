// === base name ===
kernel_e7f2438624

// === header ===
void launcher_kernel_e7f2438624(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_e7f2438624(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_e7f2438624, block.x * block.y * block.z, 256 * sizeof(float)));
        CHECK_ERR;
        if (blocksPerSM > 0) {
          gridsize = smCount * blocksPerSM;
        }
        else {
          gridsize = smCount;
        }
      }
      
  dim3 grid (std::min(gridsize, numElements0), 1, 1);
  static bool shmemsizeset = false;
      if (!shmemsizeset) {
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_e7f2438624), hipFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_e7f2438624, grid, block, 256 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_e7f2438624(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 12×16(12×16) {0..12}×{0..16} strided
    // m1 12×20(12×20) {0..12}×{0..20} strided
    // m2 16×20(16×20) {0..16}×{0..20} strided
    // m0 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, 1] = m1 12×20(12×20) {0..12}×{0..20} strided({0..12}×{0..20})[0, -1]×m2 16×20(16×20) {0..16}×{0..20} strided({0..16}×{0..20})[1, -1]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[0];
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 192 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 240 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 320 + 0 + m2_extraOffset];
          float r0[20]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v13_lead = threadIdx.x % 16;
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v15_i1 = 0; v15_i1 < 20; ++v15_i1) {
              float v23_data = __builtin_nontemporal_load(&glb_m1[(v13_lead + (v15_i1 * 12))]);
              r0[v15_i1] = v23_data;
            }
          }
          float r1[20]{};
          // r1 = load{g>r}(glb_m2);
          #pragma unroll
          for (int32_t v29_i0 = 0; v29_i0 < 1; ++v29_i0) {
            int32_t v35_lead = v13_lead + (v29_i0 * 16);
            #pragma unroll
            for (int32_t v30_i1 = 0; v30_i1 < 20; ++v30_i1) {
              float v38_data = __builtin_nontemporal_load(&glb_m2[(v35_lead + (v30_i1 * 16))]);
              r1[(v29_i0 + v30_i1)] = v38_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 16)] [(0, 20)]
          float v41_data = r1[0];
          float v49_tp{};
          float v50_tp{};
          float v51_tp{};
          float v52_tp{};
          tensorforge::transpose4x4b32(v49_tp, v50_tp, v51_tp, v52_tp, (tensorforge::broadcast<16, 1, 0>(v41_data)), (tensorforge::broadcast<16, 1, 1>(v41_data)), (tensorforge::broadcast<16, 1, 2>(v41_data)), (tensorforge::broadcast<16, 1, 3>(v41_data)));
          float v53_data = r1[1];
          float v61_tp{};
          float v62_tp{};
          float v63_tp{};
          float v64_tp{};
          tensorforge::transpose4x4b32(v61_tp, v62_tp, v63_tp, v64_tp, (tensorforge::broadcast<16, 1, 0>(v53_data)), (tensorforge::broadcast<16, 1, 1>(v53_data)), (tensorforge::broadcast<16, 1, 2>(v53_data)), (tensorforge::broadcast<16, 1, 3>(v53_data)));
          tensorforge::VectorT<float, 4> v65_acc{};
          float v66_data = r0[0];
          float v67_data = r0[1];
          float v68_data = r0[2];
          float v69_data = r0[3];
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v66_data, v65_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v67_data, v70_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v68_data, v71_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v69_data, v72_acc, 2, 0, 0);
          float v74_data = r0[4];
          float v75_data = r0[5];
          float v76_data = r0[6];
          float v77_data = r0[7];
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v74_data, v73_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v75_data, v78_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v76_data, v79_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v77_data, v80_acc, 2, 1, 0);
          float v82_data = r0[8];
          float v83_data = r0[9];
          float v84_data = r0[10];
          float v85_data = r0[11];
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v82_data, v81_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v83_data, v86_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v84_data, v87_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v85_data, v88_acc, 2, 2, 0);
          float v90_data = r0[12];
          float v91_data = r0[13];
          float v92_data = r0[14];
          float v93_data = r0[15];
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v49_tp, v90_data, v89_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v50_tp, v91_data, v94_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v51_tp, v92_data, v95_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v52_tp, v93_data, v96_acc, 2, 3, 0);
          float v98_data = r0[16];
          float v99_data = r0[17];
          float v100_data = r0[18];
          float v101_data = r0[19];
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v61_tp, v98_data, v97_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v62_tp, v99_data, v102_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v100_data, v103_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v101_data, v104_acc, 2, 0, 0);
          r2[0] = (v105_acc[0]);
          r2[1] = (v105_acc[1]);
          r2[2] = (v105_acc[2]);
          r2[3] = (v105_acc[3]);
          float v118_tp{};
          float v119_tp{};
          float v120_tp{};
          float v121_tp{};
          tensorforge::transpose4x4b32(v118_tp, v119_tp, v120_tp, v121_tp, (tensorforge::broadcast<16, 1, 4>(v41_data)), (tensorforge::broadcast<16, 1, 5>(v41_data)), (tensorforge::broadcast<16, 1, 6>(v41_data)), (tensorforge::broadcast<16, 1, 7>(v41_data)));
          float v130_tp{};
          float v131_tp{};
          float v132_tp{};
          float v133_tp{};
          tensorforge::transpose4x4b32(v130_tp, v131_tp, v132_tp, v133_tp, (tensorforge::broadcast<16, 1, 4>(v53_data)), (tensorforge::broadcast<16, 1, 5>(v53_data)), (tensorforge::broadcast<16, 1, 6>(v53_data)), (tensorforge::broadcast<16, 1, 7>(v53_data)));
          tensorforge::VectorT<float, 4> v134_acc{};
          tensorforge::VectorT<float, 4> v139_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v66_data, v134_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v140_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v67_data, v139_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v68_data, v140_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v69_data, v141_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v74_data, v142_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v75_data, v147_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v76_data, v148_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v77_data, v149_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v82_data, v150_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v83_data, v155_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v84_data, v156_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v85_data, v157_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v90_data, v158_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v91_data, v163_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v120_tp, v92_data, v164_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v121_tp, v93_data, v165_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v130_tp, v98_data, v166_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v131_tp, v99_data, v171_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v173_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v132_tp, v100_data, v172_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v174_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v133_tp, v101_data, v173_acc, 2, 0, 0);
          r2[4] = (v174_acc[0]);
          r2[5] = (v174_acc[1]);
          r2[6] = (v174_acc[2]);
          r2[7] = (v174_acc[3]);
          float v187_tp{};
          float v188_tp{};
          float v189_tp{};
          float v190_tp{};
          tensorforge::transpose4x4b32(v187_tp, v188_tp, v189_tp, v190_tp, (tensorforge::broadcast<16, 1, 8>(v41_data)), (tensorforge::broadcast<16, 1, 9>(v41_data)), (tensorforge::broadcast<16, 1, 10>(v41_data)), (tensorforge::broadcast<16, 1, 11>(v41_data)));
          float v199_tp{};
          float v200_tp{};
          float v201_tp{};
          float v202_tp{};
          tensorforge::transpose4x4b32(v199_tp, v200_tp, v201_tp, v202_tp, (tensorforge::broadcast<16, 1, 8>(v53_data)), (tensorforge::broadcast<16, 1, 9>(v53_data)), (tensorforge::broadcast<16, 1, 10>(v53_data)), (tensorforge::broadcast<16, 1, 11>(v53_data)));
          tensorforge::VectorT<float, 4> v203_acc{};
          tensorforge::VectorT<float, 4> v208_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v66_data, v203_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v67_data, v208_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v68_data, v209_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v69_data, v210_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v74_data, v211_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v75_data, v216_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v76_data, v217_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v77_data, v218_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v82_data, v219_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v83_data, v224_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v84_data, v225_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v85_data, v226_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v90_data, v227_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v91_data, v232_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v234_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v189_tp, v92_data, v233_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v235_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v190_tp, v93_data, v234_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v199_tp, v98_data, v235_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v200_tp, v99_data, v240_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v201_tp, v100_data, v241_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v202_tp, v101_data, v242_acc, 2, 0, 0);
          r2[8] = (v243_acc[0]);
          r2[9] = (v243_acc[1]);
          r2[10] = (v243_acc[2]);
          r2[11] = (v243_acc[3]);
          float v256_tp{};
          float v257_tp{};
          float v258_tp{};
          float v259_tp{};
          tensorforge::transpose4x4b32(v256_tp, v257_tp, v258_tp, v259_tp, (tensorforge::broadcast<16, 1, 12>(v41_data)), (tensorforge::broadcast<16, 1, 13>(v41_data)), (tensorforge::broadcast<16, 1, 14>(v41_data)), (tensorforge::broadcast<16, 1, 15>(v41_data)));
          float v268_tp{};
          float v269_tp{};
          float v270_tp{};
          float v271_tp{};
          tensorforge::transpose4x4b32(v268_tp, v269_tp, v270_tp, v271_tp, (tensorforge::broadcast<16, 1, 12>(v53_data)), (tensorforge::broadcast<16, 1, 13>(v53_data)), (tensorforge::broadcast<16, 1, 14>(v53_data)), (tensorforge::broadcast<16, 1, 15>(v53_data)));
          tensorforge::VectorT<float, 4> v272_acc{};
          tensorforge::VectorT<float, 4> v277_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v66_data, v272_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v278_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v67_data, v277_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v258_tp, v68_data, v278_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v69_data, v279_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v74_data, v280_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v75_data, v285_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v258_tp, v76_data, v286_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v77_data, v287_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v82_data, v288_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v83_data, v293_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v295_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v258_tp, v84_data, v294_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v296_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v85_data, v295_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v256_tp, v90_data, v296_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v257_tp, v91_data, v301_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v303_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v258_tp, v92_data, v302_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v304_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v259_tp, v93_data, v303_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v268_tp, v98_data, v304_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v269_tp, v99_data, v309_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v311_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v270_tp, v100_data, v310_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v312_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v271_tp, v101_data, v311_acc, 2, 0, 0);
          r2[12] = (v312_acc[0]);
          r2[13] = (v312_acc[1]);
          r2[14] = (v312_acc[2]);
          r2[15] = (v312_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v321_i1 = 0; v321_i1 < 16; ++v321_i1) {
              float v323_data = r2[v321_i1];
              glb_m0[(v13_lead + (v321_i1 * 12))] = v323_data;
            }
          }
        }
      }
    }
  }
}

