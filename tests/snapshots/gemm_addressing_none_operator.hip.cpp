// === base name ===
kernel_4838a21a5e7c92cf

// === header ===
void launcher_kernel_4838a21a5e7c92cf(float* m0, size_t m0_extraOffset, const float* m1, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_4838a21a5e7c92cf(float* m0, size_t m0_extraOffset, const float* m1, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_4838a21a5e7c92cf, block.x * block.y * block.z, 512 * sizeof(float)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (512 * sizeof(float)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_4838a21a5e7c92cf, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (512 * sizeof(float)));
          blocksPerSM = std::max(blocksPerSM, std::min(blocksNoLds, blocksByLds));
        }
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_4838a21a5e7c92cf), hipFuncAttributeMaxDynamicSharedMemorySize, 512 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_4838a21a5e7c92cf, grid, block, 512 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_4838a21a5e7c92cf(float* m0, size_t m0_extraOffset, const float* m1, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
    // meta data:
    // m0 16×16(16×16) {0..16}×{0..16} strided
    // m1 16×16(16×16) {0..16}×{0..16} none
    // m2 16×16(16×16) {0..16}×{0..16} strided
    // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} none({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
    {
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[16 * threadIdx.y + 256];
      float* tempShrMem = &localShrMem0[0];
      tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[0];
      float * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      __syncthreads();
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m0[v5_batchId0 * 256 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[v5_batchId0 * 256 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m2);
          int32_t v18_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
            int32_t v25_lead = v18_lead + (v19_i0 * 16);
            #pragma unroll
            for (int32_t v20_i1 = 0; v20_i1 < 16; ++v20_i1) {
              float v28_data = __builtin_nontemporal_load(&glb_m2[(v25_lead + (v20_i1 * 16))]);
              r0[(v19_i0 + v20_i1)] = v28_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m2););
          float r1[16]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float v31_data = r0[0];
          float v32_data = r0[1];
          float v33_data = r0[2];
          float v34_data = r0[3];
          float v35_tp{};
          float v36_tp{};
          float v37_tp{};
          float v38_tp{};
          tensorforge::transpose4x4b32(v35_tp, v36_tp, v37_tp, v38_tp, v31_data, v32_data, v33_data, v34_data);
          tensorforge::VectorT<float, 4> v39_acc{};
          float v46_data = glb_m1[v18_lead];
          float v53_data = glb_m1[(v18_lead + 16)];
          float v60_data = glb_m1[(v18_lead + 32)];
          float v67_data = glb_m1[(v18_lead + 48)];
          tensorforge::VectorT<float, 4> v68_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v46_data, v39_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v69_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v53_data, v68_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v70_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v60_data, v69_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v71_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v67_data, v70_acc, 2, 0, 0);
          float v78_data = glb_m1[(v18_lead + 64)];
          float v85_data = glb_m1[(v18_lead + 80)];
          float v92_data = glb_m1[(v18_lead + 96)];
          float v99_data = glb_m1[(v18_lead + 112)];
          tensorforge::VectorT<float, 4> v100_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v78_data, v71_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v101_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v85_data, v100_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v102_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v92_data, v101_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v103_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v99_data, v102_acc, 2, 1, 0);
          float v110_data = glb_m1[(v18_lead + 128)];
          float v117_data = glb_m1[(v18_lead + 144)];
          float v124_data = glb_m1[(v18_lead + 160)];
          float v131_data = glb_m1[(v18_lead + 176)];
          tensorforge::VectorT<float, 4> v132_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v110_data, v103_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v133_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v117_data, v132_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v134_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v124_data, v133_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v135_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v131_data, v134_acc, 2, 2, 0);
          float v142_data = glb_m1[(v18_lead + 192)];
          float v149_data = glb_m1[(v18_lead + 208)];
          float v156_data = glb_m1[(v18_lead + 224)];
          float v163_data = glb_m1[(v18_lead + 240)];
          tensorforge::VectorT<float, 4> v164_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v35_tp, v142_data, v135_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v165_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v36_tp, v149_data, v164_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v166_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v37_tp, v156_data, v165_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v167_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v38_tp, v163_data, v166_acc, 2, 3, 0);
          r1[0] = (v167_acc[0]);
          r1[1] = (v167_acc[1]);
          r1[2] = (v167_acc[2]);
          r1[3] = (v167_acc[3]);
          float v172_data = r0[4];
          float v173_data = r0[5];
          float v174_data = r0[6];
          float v175_data = r0[7];
          float v176_tp{};
          float v177_tp{};
          float v178_tp{};
          float v179_tp{};
          tensorforge::transpose4x4b32(v176_tp, v177_tp, v178_tp, v179_tp, v172_data, v173_data, v174_data, v175_data);
          tensorforge::VectorT<float, 4> v180_acc{};
          float v187_data = glb_m1[v18_lead];
          float v194_data = glb_m1[(v18_lead + 16)];
          float v201_data = glb_m1[(v18_lead + 32)];
          float v208_data = glb_m1[(v18_lead + 48)];
          tensorforge::VectorT<float, 4> v209_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v187_data, v180_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v210_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v194_data, v209_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v211_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v201_data, v210_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v212_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v208_data, v211_acc, 2, 0, 0);
          float v219_data = glb_m1[(v18_lead + 64)];
          float v226_data = glb_m1[(v18_lead + 80)];
          float v233_data = glb_m1[(v18_lead + 96)];
          float v240_data = glb_m1[(v18_lead + 112)];
          tensorforge::VectorT<float, 4> v241_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v219_data, v212_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v242_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v226_data, v241_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v243_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v233_data, v242_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v244_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v240_data, v243_acc, 2, 1, 0);
          float v251_data = glb_m1[(v18_lead + 128)];
          float v258_data = glb_m1[(v18_lead + 144)];
          float v265_data = glb_m1[(v18_lead + 160)];
          float v272_data = glb_m1[(v18_lead + 176)];
          tensorforge::VectorT<float, 4> v273_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v251_data, v244_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v274_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v258_data, v273_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v275_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v265_data, v274_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v276_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v272_data, v275_acc, 2, 2, 0);
          float v283_data = glb_m1[(v18_lead + 192)];
          float v290_data = glb_m1[(v18_lead + 208)];
          float v297_data = glb_m1[(v18_lead + 224)];
          float v304_data = glb_m1[(v18_lead + 240)];
          tensorforge::VectorT<float, 4> v305_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v176_tp, v283_data, v276_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v306_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v177_tp, v290_data, v305_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v307_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v178_tp, v297_data, v306_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v308_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v179_tp, v304_data, v307_acc, 2, 3, 0);
          r1[4] = (v308_acc[0]);
          r1[5] = (v308_acc[1]);
          r1[6] = (v308_acc[2]);
          r1[7] = (v308_acc[3]);
          float v313_data = r0[8];
          float v314_data = r0[9];
          float v315_data = r0[10];
          float v316_data = r0[11];
          float v317_tp{};
          float v318_tp{};
          float v319_tp{};
          float v320_tp{};
          tensorforge::transpose4x4b32(v317_tp, v318_tp, v319_tp, v320_tp, v313_data, v314_data, v315_data, v316_data);
          tensorforge::VectorT<float, 4> v321_acc{};
          float v328_data = glb_m1[v18_lead];
          float v335_data = glb_m1[(v18_lead + 16)];
          float v342_data = glb_m1[(v18_lead + 32)];
          float v349_data = glb_m1[(v18_lead + 48)];
          tensorforge::VectorT<float, 4> v350_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v317_tp, v328_data, v321_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v351_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v335_data, v350_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v352_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v319_tp, v342_data, v351_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v353_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v349_data, v352_acc, 2, 0, 0);
          float v360_data = glb_m1[(v18_lead + 64)];
          float v367_data = glb_m1[(v18_lead + 80)];
          float v374_data = glb_m1[(v18_lead + 96)];
          float v381_data = glb_m1[(v18_lead + 112)];
          tensorforge::VectorT<float, 4> v382_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v317_tp, v360_data, v353_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v383_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v367_data, v382_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v384_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v319_tp, v374_data, v383_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v385_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v381_data, v384_acc, 2, 1, 0);
          float v392_data = glb_m1[(v18_lead + 128)];
          float v399_data = glb_m1[(v18_lead + 144)];
          float v406_data = glb_m1[(v18_lead + 160)];
          float v413_data = glb_m1[(v18_lead + 176)];
          tensorforge::VectorT<float, 4> v414_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v317_tp, v392_data, v385_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v415_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v399_data, v414_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v416_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v319_tp, v406_data, v415_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v417_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v413_data, v416_acc, 2, 2, 0);
          float v424_data = glb_m1[(v18_lead + 192)];
          float v431_data = glb_m1[(v18_lead + 208)];
          float v438_data = glb_m1[(v18_lead + 224)];
          float v445_data = glb_m1[(v18_lead + 240)];
          tensorforge::VectorT<float, 4> v446_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v317_tp, v424_data, v417_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v447_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v318_tp, v431_data, v446_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v448_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v319_tp, v438_data, v447_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v449_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v320_tp, v445_data, v448_acc, 2, 3, 0);
          r1[8] = (v449_acc[0]);
          r1[9] = (v449_acc[1]);
          r1[10] = (v449_acc[2]);
          r1[11] = (v449_acc[3]);
          float v454_data = r0[12];
          float v455_data = r0[13];
          float v456_data = r0[14];
          float v457_data = r0[15];
          float v458_tp{};
          float v459_tp{};
          float v460_tp{};
          float v461_tp{};
          tensorforge::transpose4x4b32(v458_tp, v459_tp, v460_tp, v461_tp, v454_data, v455_data, v456_data, v457_data);
          tensorforge::VectorT<float, 4> v462_acc{};
          float v469_data = glb_m1[v18_lead];
          float v476_data = glb_m1[(v18_lead + 16)];
          float v483_data = glb_m1[(v18_lead + 32)];
          float v490_data = glb_m1[(v18_lead + 48)];
          tensorforge::VectorT<float, 4> v491_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v469_data, v462_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v492_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v476_data, v491_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v493_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v483_data, v492_acc, 2, 0, 0);
          tensorforge::VectorT<float, 4> v494_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v490_data, v493_acc, 2, 0, 0);
          float v501_data = glb_m1[(v18_lead + 64)];
          float v508_data = glb_m1[(v18_lead + 80)];
          float v515_data = glb_m1[(v18_lead + 96)];
          float v522_data = glb_m1[(v18_lead + 112)];
          tensorforge::VectorT<float, 4> v523_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v501_data, v494_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v524_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v508_data, v523_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v525_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v515_data, v524_acc, 2, 1, 0);
          tensorforge::VectorT<float, 4> v526_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v522_data, v525_acc, 2, 1, 0);
          float v533_data = glb_m1[(v18_lead + 128)];
          float v540_data = glb_m1[(v18_lead + 144)];
          float v547_data = glb_m1[(v18_lead + 160)];
          float v554_data = glb_m1[(v18_lead + 176)];
          tensorforge::VectorT<float, 4> v555_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v533_data, v526_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v556_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v540_data, v555_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v557_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v547_data, v556_acc, 2, 2, 0);
          tensorforge::VectorT<float, 4> v558_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v554_data, v557_acc, 2, 2, 0);
          float v565_data = glb_m1[(v18_lead + 192)];
          float v572_data = glb_m1[(v18_lead + 208)];
          float v579_data = glb_m1[(v18_lead + 224)];
          float v586_data = glb_m1[(v18_lead + 240)];
          tensorforge::VectorT<float, 4> v587_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v458_tp, v565_data, v558_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v588_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v459_tp, v572_data, v587_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v589_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v460_tp, v579_data, v588_acc, 2, 3, 0);
          tensorforge::VectorT<float, 4> v590_acc = __builtin_amdgcn_mfma_f32_4x4x1f32(v461_tp, v586_data, v589_acc, 2, 3, 0);
          r1[12] = (v590_acc[0]);
          r1[13] = (v590_acc[1]);
          r1[14] = (v590_acc[2]);
          r1[15] = (v590_acc[3]);
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v598_i0 = 0; v598_i0 < 1; ++v598_i0) {
            int32_t v606_lead = v18_lead + (v598_i0 * 16);
            #pragma unroll
            for (int32_t v599_i1 = 0; v599_i1 < 16; ++v599_i1) {
              float v601_data = r1[(v598_i0 + v599_i1)];
              glb_m0[(v606_lead + (v599_i1 * 16))] = v601_data;
            }
          }
        }
      }
    }
  }
}

