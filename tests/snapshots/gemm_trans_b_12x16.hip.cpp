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
          float v26_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v26_lin;
          float v27_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v27_lin;
          float v28_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v28_lin;
          float v29_lin = glb_m2[48 + threadIdx.x * 1];
          r1[3] = v29_lin;
          float v30_lin = glb_m2[64 + threadIdx.x * 1];
          r1[4] = v30_lin;
          float v31_lin = glb_m2[80 + threadIdx.x * 1];
          r1[5] = v31_lin;
          float v32_lin = glb_m2[96 + threadIdx.x * 1];
          r1[6] = v32_lin;
          float v33_lin = glb_m2[112 + threadIdx.x * 1];
          r1[7] = v33_lin;
          float v34_lin = glb_m2[128 + threadIdx.x * 1];
          r1[8] = v34_lin;
          float v35_lin = glb_m2[144 + threadIdx.x * 1];
          r1[9] = v35_lin;
          float v36_lin = glb_m2[160 + threadIdx.x * 1];
          r1[10] = v36_lin;
          float v37_lin = glb_m2[176 + threadIdx.x * 1];
          r1[11] = v37_lin;
          float v38_lin = glb_m2[192 + threadIdx.x * 1];
          r1[12] = v38_lin;
          float v39_lin = glb_m2[208 + threadIdx.x * 1];
          r1[13] = v39_lin;
          float v40_lin = glb_m2[224 + threadIdx.x * 1];
          r1[14] = v40_lin;
          float v41_lin = glb_m2[240 + threadIdx.x * 1];
          r1[15] = v41_lin;
          float v42_lin = glb_m2[256 + threadIdx.x * 1];
          r1[16] = v42_lin;
          float v43_lin = glb_m2[272 + threadIdx.x * 1];
          r1[17] = v43_lin;
          float v44_lin = glb_m2[288 + threadIdx.x * 1];
          r1[18] = v44_lin;
          float v45_lin = glb_m2[304 + threadIdx.x * 1];
          r1[19] = v45_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 16)] [(0, 20)]
          float v47_data = r1[0];
          float v55_tp{};
          float v56_tp{};
          float v57_tp{};
          float v58_tp{};
          tensorforge::transpose4x4b32(v55_tp, v56_tp, v57_tp, v58_tp, (tensorforge::broadcast<16, 1, 0>(v47_data)), (tensorforge::broadcast<16, 1, 1>(v47_data)), (tensorforge::broadcast<16, 1, 2>(v47_data)), (tensorforge::broadcast<16, 1, 3>(v47_data)));
          float v59_data = r1[1];
          float v67_tp{};
          float v68_tp{};
          float v69_tp{};
          float v70_tp{};
          tensorforge::transpose4x4b32(v67_tp, v68_tp, v69_tp, v70_tp, (tensorforge::broadcast<16, 1, 0>(v59_data)), (tensorforge::broadcast<16, 1, 1>(v59_data)), (tensorforge::broadcast<16, 1, 2>(v59_data)), (tensorforge::broadcast<16, 1, 3>(v59_data)));
          tensorforge::VectorT<float, 4> v71_acc{};
          float v72_data = r0[0];
          float v73_data = r0[1];
          float v74_data = r0[2];
          float v75_data = r0[3];
          tensorforge::VectorT<float, 4> v76_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v72_data, v71_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v77_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v73_data, v76_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v78_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v74_data, v77_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v79_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v75_data, v78_acc, 2, 0, 0);
          float v80_data = r0[4];
          float v81_data = r0[5];
          float v82_data = r0[6];
          float v83_data = r0[7];
          tensorforge::VectorT<float, 4> v84_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v80_data, v79_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v85_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v81_data, v84_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v86_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v82_data, v85_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v87_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v83_data, v86_acc, 2, 1, 0);
          float v88_data = r0[8];
          float v89_data = r0[9];
          float v90_data = r0[10];
          float v91_data = r0[11];
          tensorforge::VectorT<float, 4> v92_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v88_data, v87_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v93_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v89_data, v92_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v94_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v90_data, v93_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v95_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v91_data, v94_acc, 2, 2, 0);
          float v96_data = r0[12];
          float v97_data = r0[13];
          float v98_data = r0[14];
          float v99_data = r0[15];
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v96_data, v95_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v97_data, v100_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v98_data, v101_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v99_data, v102_acc, 2, 3, 0);
          float v104_data = r0[16];
          float v105_data = r0[17];
          float v106_data = r0[18];
          float v107_data = r0[19];
          tensorforge::VectorT<float, 4> v108_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v67_tp, v104_data, v103_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v109_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v68_tp, v105_data, v108_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v110_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v69_tp, v106_data, v109_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v111_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v70_tp, v107_data, v110_acc, 2, 0, 0);
          r2[0] = (v111_acc[0]);
          r2[1] = (v111_acc[1]);
          r2[2] = (v111_acc[2]);
          r2[3] = (v111_acc[3]);
          float v124_tp{};
          float v125_tp{};
          float v126_tp{};
          float v127_tp{};
          tensorforge::transpose4x4b32(v124_tp, v125_tp, v126_tp, v127_tp, (tensorforge::broadcast<16, 1, 4>(v47_data)), (tensorforge::broadcast<16, 1, 5>(v47_data)), (tensorforge::broadcast<16, 1, 6>(v47_data)), (tensorforge::broadcast<16, 1, 7>(v47_data)));
          float v136_tp{};
          float v137_tp{};
          float v138_tp{};
          float v139_tp{};
          tensorforge::transpose4x4b32(v136_tp, v137_tp, v138_tp, v139_tp, (tensorforge::broadcast<16, 1, 4>(v59_data)), (tensorforge::broadcast<16, 1, 5>(v59_data)), (tensorforge::broadcast<16, 1, 6>(v59_data)), (tensorforge::broadcast<16, 1, 7>(v59_data)));
          tensorforge::VectorT<float, 4> v140_acc{};
          tensorforge::VectorT<float, 4> v145_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v72_data, v140_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v146_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v73_data, v145_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v147_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v74_data, v146_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v148_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v75_data, v147_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v153_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v80_data, v148_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v154_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v81_data, v153_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v155_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v82_data, v154_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v156_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v83_data, v155_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v161_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v88_data, v156_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v162_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v89_data, v161_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v163_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v90_data, v162_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v91_data, v163_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v169_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v96_data, v164_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v170_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v97_data, v169_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v171_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v98_data, v170_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v172_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v99_data, v171_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v177_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v136_tp, v104_data, v172_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v178_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v137_tp, v105_data, v177_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v179_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v138_tp, v106_data, v178_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v180_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v139_tp, v107_data, v179_acc, 2, 0, 0);
          r2[4] = (v180_acc[0]);
          r2[5] = (v180_acc[1]);
          r2[6] = (v180_acc[2]);
          r2[7] = (v180_acc[3]);
          float v193_tp{};
          float v194_tp{};
          float v195_tp{};
          float v196_tp{};
          tensorforge::transpose4x4b32(v193_tp, v194_tp, v195_tp, v196_tp, (tensorforge::broadcast<16, 1, 8>(v47_data)), (tensorforge::broadcast<16, 1, 9>(v47_data)), (tensorforge::broadcast<16, 1, 10>(v47_data)), (tensorforge::broadcast<16, 1, 11>(v47_data)));
          float v205_tp{};
          float v206_tp{};
          float v207_tp{};
          float v208_tp{};
          tensorforge::transpose4x4b32(v205_tp, v206_tp, v207_tp, v208_tp, (tensorforge::broadcast<16, 1, 8>(v59_data)), (tensorforge::broadcast<16, 1, 9>(v59_data)), (tensorforge::broadcast<16, 1, 10>(v59_data)), (tensorforge::broadcast<16, 1, 11>(v59_data)));
          tensorforge::VectorT<float, 4> v209_acc{};
          tensorforge::VectorT<float, 4> v214_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v72_data, v209_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v215_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v73_data, v214_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v216_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, v74_data, v215_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v217_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v75_data, v216_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v222_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v80_data, v217_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v223_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v81_data, v222_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v224_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, v82_data, v223_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v225_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v83_data, v224_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v230_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v88_data, v225_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v231_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v89_data, v230_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v232_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, v90_data, v231_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v233_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v91_data, v232_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v238_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v193_tp, v96_data, v233_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v239_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v194_tp, v97_data, v238_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v240_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v195_tp, v98_data, v239_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v196_tp, v99_data, v240_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v246_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v205_tp, v104_data, v241_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v247_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v206_tp, v105_data, v246_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v248_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v207_tp, v106_data, v247_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v249_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v208_tp, v107_data, v248_acc, 2, 0, 0);
          r2[8] = (v249_acc[0]);
          r2[9] = (v249_acc[1]);
          r2[10] = (v249_acc[2]);
          r2[11] = (v249_acc[3]);
          float v262_tp{};
          float v263_tp{};
          float v264_tp{};
          float v265_tp{};
          tensorforge::transpose4x4b32(v262_tp, v263_tp, v264_tp, v265_tp, (tensorforge::broadcast<16, 1, 12>(v47_data)), (tensorforge::broadcast<16, 1, 13>(v47_data)), (tensorforge::broadcast<16, 1, 14>(v47_data)), (tensorforge::broadcast<16, 1, 15>(v47_data)));
          float v274_tp{};
          float v275_tp{};
          float v276_tp{};
          float v277_tp{};
          tensorforge::transpose4x4b32(v274_tp, v275_tp, v276_tp, v277_tp, (tensorforge::broadcast<16, 1, 12>(v59_data)), (tensorforge::broadcast<16, 1, 13>(v59_data)), (tensorforge::broadcast<16, 1, 14>(v59_data)), (tensorforge::broadcast<16, 1, 15>(v59_data)));
          tensorforge::VectorT<float, 4> v278_acc{};
          tensorforge::VectorT<float, 4> v283_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v72_data, v278_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v284_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v73_data, v283_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v285_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v74_data, v284_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v286_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v75_data, v285_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v291_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v80_data, v286_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v292_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v81_data, v291_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v293_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v82_data, v292_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v294_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v83_data, v293_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v299_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v88_data, v294_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v300_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v89_data, v299_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v301_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v90_data, v300_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v302_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v91_data, v301_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v262_tp, v96_data, v302_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v263_tp, v97_data, v307_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v309_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v264_tp, v98_data, v308_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v310_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v265_tp, v99_data, v309_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v315_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v274_tp, v104_data, v310_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v316_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v275_tp, v105_data, v315_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v317_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v276_tp, v106_data, v316_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v318_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v277_tp, v107_data, v317_acc, 2, 0, 0);
          r2[12] = (v318_acc[0]);
          r2[13] = (v318_acc[1]);
          r2[14] = (v318_acc[2]);
          r2[15] = (v318_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v13_lead < 12) {
            #pragma unroll
            for (int32_t v327_i1 = 0; v327_i1 < 16; ++v327_i1) {
              float v329_data = r2[v327_i1];
              glb_m0[(v13_lead + (v327_i1 * 12))] = v329_data;
            }
          }
        }
      }
    }
  }
}

