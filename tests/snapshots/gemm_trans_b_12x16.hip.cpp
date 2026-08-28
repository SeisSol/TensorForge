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
          int32_t v10_lead = threadIdx.x % 16;
          if (v10_lead < 12) {
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 20; ++v12_i1) {
              int32_t v18_a = v12_i1 * 12;
              int32_t v19_a = v10_lead + v18_a;
              float v27_data = __builtin_nontemporal_load(&glb_m1[(v10_lead + v18_a)]);
              r0[v12_i1] = v27_data;
            }
          }
          float r1[20]{};
          // r1 = load{g>r}(glb_m2);
          float v30_lin = glb_m2[0 + threadIdx.x * 1];
          r1[0] = v30_lin;
          float v31_lin = glb_m2[16 + threadIdx.x * 1];
          r1[1] = v31_lin;
          float v32_lin = glb_m2[32 + threadIdx.x * 1];
          r1[2] = v32_lin;
          float v33_lin = glb_m2[48 + threadIdx.x * 1];
          r1[3] = v33_lin;
          float v34_lin = glb_m2[64 + threadIdx.x * 1];
          r1[4] = v34_lin;
          float v35_lin = glb_m2[80 + threadIdx.x * 1];
          r1[5] = v35_lin;
          float v36_lin = glb_m2[96 + threadIdx.x * 1];
          r1[6] = v36_lin;
          float v37_lin = glb_m2[112 + threadIdx.x * 1];
          r1[7] = v37_lin;
          float v38_lin = glb_m2[128 + threadIdx.x * 1];
          r1[8] = v38_lin;
          float v39_lin = glb_m2[144 + threadIdx.x * 1];
          r1[9] = v39_lin;
          float v40_lin = glb_m2[160 + threadIdx.x * 1];
          r1[10] = v40_lin;
          float v41_lin = glb_m2[176 + threadIdx.x * 1];
          r1[11] = v41_lin;
          float v42_lin = glb_m2[192 + threadIdx.x * 1];
          r1[12] = v42_lin;
          float v43_lin = glb_m2[208 + threadIdx.x * 1];
          r1[13] = v43_lin;
          float v44_lin = glb_m2[224 + threadIdx.x * 1];
          r1[14] = v44_lin;
          float v45_lin = glb_m2[240 + threadIdx.x * 1];
          r1[15] = v45_lin;
          float v46_lin = glb_m2[256 + threadIdx.x * 1];
          r1[16] = v46_lin;
          float v47_lin = glb_m2[272 + threadIdx.x * 1];
          r1[17] = v47_lin;
          float v48_lin = glb_m2[288 + threadIdx.x * 1];
          r1[18] = v48_lin;
          float v49_lin = glb_m2[304 + threadIdx.x * 1];
          r1[19] = v49_lin;
          // wait(r0 = load{g>r}(glb_m1););
          // wait(r1 = load{g>r}(glb_m2););
          float r2[16]{};
          // r2 = +(r0 * r1) + None
          // [(0, 12), (0, 16)] [(0, 20)]
          float v51_data = tensorforge::broadcast<16, 1, 0>(r1[0]);
          float v52_data = tensorforge::broadcast<16, 1, 1>(r1[0]);
          float v53_data = tensorforge::broadcast<16, 1, 2>(r1[0]);
          float v54_data = tensorforge::broadcast<16, 1, 3>(r1[0]);
          float v55_tp{};
          float v56_tp{};
          float v57_tp{};
          float v58_tp{};
          tensorforge::transpose4x4b32(v55_tp, v56_tp, v57_tp, v58_tp, v51_data, v52_data, v53_data, v54_data);
          float v59_data = tensorforge::broadcast<16, 1, 0>(r1[1]);
          float v60_data = tensorforge::broadcast<16, 1, 1>(r1[1]);
          float v61_data = tensorforge::broadcast<16, 1, 2>(r1[1]);
          float v62_data = tensorforge::broadcast<16, 1, 3>(r1[1]);
          float v63_tp{};
          float v64_tp{};
          float v65_tp{};
          float v66_tp{};
          tensorforge::transpose4x4b32(v63_tp, v64_tp, v65_tp, v66_tp, v59_data, v60_data, v61_data, v62_data);
          tensorforge::VectorT<float, 4> v67_acc{};
          float v68_data = r0[0];
          float v69_data = r0[1];
          float v70_data = r0[2];
          float v71_data = r0[3];
          tensorforge::VectorT<float, 4> v72_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v68_data, v67_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v73_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v69_data, v72_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v74_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v70_data, v73_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v75_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v71_data, v74_acc, 2, 0, 0);
          float v76_data = r0[4];
          float v77_data = r0[5];
          float v78_data = r0[6];
          float v79_data = r0[7];
          tensorforge::VectorT<float, 4> v80_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v76_data, v75_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v81_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v77_data, v80_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v82_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v78_data, v81_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v83_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v79_data, v82_acc, 2, 1, 0);
          float v84_data = r0[8];
          float v85_data = r0[9];
          float v86_data = r0[10];
          float v87_data = r0[11];
          tensorforge::VectorT<float, 4> v88_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v84_data, v83_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v89_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v85_data, v88_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v90_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v86_data, v89_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v91_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v87_data, v90_acc, 2, 2, 0);
          float v92_data = r0[12];
          float v93_data = r0[13];
          float v94_data = r0[14];
          float v95_data = r0[15];
          tensorforge::VectorT<float, 4> v96_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v55_tp, v92_data, v91_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v97_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v56_tp, v93_data, v96_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v98_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v57_tp, v94_data, v97_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v99_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v58_tp, v95_data, v98_acc, 2, 3, 0);
          float v100_data = r0[16];
          float v101_data = r0[17];
          float v102_data = r0[18];
          float v103_data = r0[19];
          tensorforge::VectorT<float, 4> v104_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v63_tp, v100_data, v99_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v105_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v64_tp, v101_data, v104_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v106_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v65_tp, v102_data, v105_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v107_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v66_tp, v103_data, v106_acc, 2, 0, 0);
          r2[0] = (v107_acc[0]);
          r2[1] = (v107_acc[1]);
          r2[2] = (v107_acc[2]);
          r2[3] = (v107_acc[3]);
          float v112_data = tensorforge::broadcast<16, 1, 4>(r1[0]);
          float v113_data = tensorforge::broadcast<16, 1, 5>(r1[0]);
          float v114_data = tensorforge::broadcast<16, 1, 6>(r1[0]);
          float v115_data = tensorforge::broadcast<16, 1, 7>(r1[0]);
          float v116_tp{};
          float v117_tp{};
          float v118_tp{};
          float v119_tp{};
          tensorforge::transpose4x4b32(v116_tp, v117_tp, v118_tp, v119_tp, v112_data, v113_data, v114_data, v115_data);
          float v120_data = tensorforge::broadcast<16, 1, 4>(r1[1]);
          float v121_data = tensorforge::broadcast<16, 1, 5>(r1[1]);
          float v122_data = tensorforge::broadcast<16, 1, 6>(r1[1]);
          float v123_data = tensorforge::broadcast<16, 1, 7>(r1[1]);
          float v124_tp{};
          float v125_tp{};
          float v126_tp{};
          float v127_tp{};
          tensorforge::transpose4x4b32(v124_tp, v125_tp, v126_tp, v127_tp, v120_data, v121_data, v122_data, v123_data);
          tensorforge::VectorT<float, 4> v128_acc{};
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v68_data, v128_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v69_data, v133_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v70_data, v134_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v136_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v71_data, v135_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v141_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v76_data, v136_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v142_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v77_data, v141_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v143_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v78_data, v142_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v144_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v79_data, v143_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v149_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v84_data, v144_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v150_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v85_data, v149_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v151_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v86_data, v150_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v152_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v87_data, v151_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v157_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v116_tp, v92_data, v152_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v158_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v117_tp, v93_data, v157_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v159_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v118_tp, v94_data, v158_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v160_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v119_tp, v95_data, v159_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v124_tp, v100_data, v160_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v125_tp, v101_data, v165_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v126_tp, v102_data, v166_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v168_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v127_tp, v103_data, v167_acc, 2, 0, 0);
          r2[4] = (v168_acc[0]);
          r2[5] = (v168_acc[1]);
          r2[6] = (v168_acc[2]);
          r2[7] = (v168_acc[3]);
          float v173_data = tensorforge::broadcast<16, 1, 8>(r1[0]);
          float v174_data = tensorforge::broadcast<16, 1, 9>(r1[0]);
          float v175_data = tensorforge::broadcast<16, 1, 10>(r1[0]);
          float v176_data = tensorforge::broadcast<16, 1, 11>(r1[0]);
          float v177_tp{};
          float v178_tp{};
          float v179_tp{};
          float v180_tp{};
          tensorforge::transpose4x4b32(v177_tp, v178_tp, v179_tp, v180_tp, v173_data, v174_data, v175_data, v176_data);
          float v181_data = tensorforge::broadcast<16, 1, 8>(r1[1]);
          float v182_data = tensorforge::broadcast<16, 1, 9>(r1[1]);
          float v183_data = tensorforge::broadcast<16, 1, 10>(r1[1]);
          float v184_data = tensorforge::broadcast<16, 1, 11>(r1[1]);
          float v185_tp{};
          float v186_tp{};
          float v187_tp{};
          float v188_tp{};
          tensorforge::transpose4x4b32(v185_tp, v186_tp, v187_tp, v188_tp, v181_data, v182_data, v183_data, v184_data);
          tensorforge::VectorT<float, 4> v189_acc{};
          tensorforge::VectorT<float, 4> v194_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v68_data, v189_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v195_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v69_data, v194_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v196_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v70_data, v195_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v197_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v71_data, v196_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v202_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v76_data, v197_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v203_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v77_data, v202_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v204_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v78_data, v203_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v205_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v79_data, v204_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v84_data, v205_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v85_data, v210_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v86_data, v211_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v213_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v87_data, v212_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v218_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v92_data, v213_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v219_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v93_data, v218_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v220_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v94_data, v219_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v221_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v180_tp, v95_data, v220_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v226_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v185_tp, v100_data, v221_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v227_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v186_tp, v101_data, v226_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v228_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v187_tp, v102_data, v227_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v229_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v188_tp, v103_data, v228_acc, 2, 0, 0);
          r2[8] = (v229_acc[0]);
          r2[9] = (v229_acc[1]);
          r2[10] = (v229_acc[2]);
          r2[11] = (v229_acc[3]);
          float v234_data = tensorforge::broadcast<16, 1, 12>(r1[0]);
          float v235_data = tensorforge::broadcast<16, 1, 13>(r1[0]);
          float v236_data = tensorforge::broadcast<16, 1, 14>(r1[0]);
          float v237_data = tensorforge::broadcast<16, 1, 15>(r1[0]);
          float v238_tp{};
          float v239_tp{};
          float v240_tp{};
          float v241_tp{};
          tensorforge::transpose4x4b32(v238_tp, v239_tp, v240_tp, v241_tp, v234_data, v235_data, v236_data, v237_data);
          float v242_data = tensorforge::broadcast<16, 1, 12>(r1[1]);
          float v243_data = tensorforge::broadcast<16, 1, 13>(r1[1]);
          float v244_data = tensorforge::broadcast<16, 1, 14>(r1[1]);
          float v245_data = tensorforge::broadcast<16, 1, 15>(r1[1]);
          float v246_tp{};
          float v247_tp{};
          float v248_tp{};
          float v249_tp{};
          tensorforge::transpose4x4b32(v246_tp, v247_tp, v248_tp, v249_tp, v242_data, v243_data, v244_data, v245_data);
          tensorforge::VectorT<float, 4> v250_acc{};
          tensorforge::VectorT<float, 4> v255_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v68_data, v250_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v256_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v69_data, v255_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v257_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v70_data, v256_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v258_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v71_data, v257_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v263_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v76_data, v258_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v264_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v77_data, v263_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v265_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v78_data, v264_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v266_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v79_data, v265_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v271_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v84_data, v266_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v272_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v85_data, v271_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v86_data, v272_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v87_data, v273_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v279_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v238_tp, v92_data, v274_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v280_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v239_tp, v93_data, v279_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v281_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v240_tp, v94_data, v280_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v282_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v241_tp, v95_data, v281_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v287_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v246_tp, v100_data, v282_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v288_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v247_tp, v101_data, v287_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v289_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v248_tp, v102_data, v288_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v290_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v249_tp, v103_data, v289_acc, 2, 0, 0);
          r2[12] = (v290_acc[0]);
          r2[13] = (v290_acc[1]);
          r2[14] = (v290_acc[2]);
          r2[15] = (v290_acc[3]);
          // glb_m0 = store{r>g}(r2);
          if (v10_lead < 12) {
            #pragma unroll
            for (int32_t v299_i1 = 0; v299_i1 < 16; ++v299_i1) {
              int32_t v300_a = 0 + v299_i1;
              float v302_data = r2[v299_i1];
              glb_m0[(v10_lead + (v299_i1 * 12))] = v302_data;
            }
          }
        }
      }
    }
  }
}

