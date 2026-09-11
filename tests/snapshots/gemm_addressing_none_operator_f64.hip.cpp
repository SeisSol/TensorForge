// === base name ===
kernel_b34edf45ce21aa11

// === header ===
void launcher_kernel_b34edf45ce21aa11(double* m0, size_t m0_extraOffset, const double* m1, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_b34edf45ce21aa11(double* m0, size_t m0_extraOffset, const double* m1, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_b34edf45ce21aa11, block.x * block.y * block.z, 512 * sizeof(double)));
        CHECK_ERR;
        int gfxMajor = 0;
        CHECK_RES(hipDeviceGetAttribute(&gfxMajor, hipDeviceAttributeComputeCapabilityMajor, device));
        if (gfxMajor >= 10 && (512 * sizeof(double)) > 0) {
          int ldsPerMP = 0, blocksNoLds = 0;
          CHECK_RES(hipDeviceGetAttribute(&ldsPerMP, hipDeviceAttributeMaxSharedMemoryPerMultiprocessor, device));
          CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksNoLds, kernel_kernel_b34edf45ce21aa11, block.x * block.y * block.z, 0));
          const int blocksByLds = static_cast<int>((2 * static_cast<std::size_t>(ldsPerMP)) / (512 * sizeof(double)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_b34edf45ce21aa11), hipFuncAttributeMaxDynamicSharedMemorySize, 512 * sizeof(double)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_b34edf45ce21aa11, grid, block, 512 * sizeof(double), stream,  m0,  m0_extraOffset,  m1,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_b34edf45ce21aa11(double* m0, size_t m0_extraOffset, const double* m1, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[16 * threadIdx.y + 256];
      double* tempShrMem = &localShrMem0[0];
      tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const ptr_glb_m1 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m1[0];
      double * __restrict__ glb_m1 = &totalShrMem[0];
      // glb_m1 = load{g>s}(ptr_glb_m1[0, 1])
      glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0] = __builtin_nontemporal_load(&ptr_glb_m1[0 + 0 + 1 * (threadIdx.x + threadIdx.y * blockDim.x) + 0]);
      __syncthreads();
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace> const glb_m0 = (tensorforge::SpacePtrRestrict<double, tensorforge::GlobalMemspace>)&m0[v5_batchId0 * 256 + 0 + m0_extraOffset];
          tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace> const glb_m2 = (tensorforge::SpacePtrRestrict<const double, tensorforge::GlobalMemspace>)&m2[v5_batchId0 * 256 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m2);
          int32_t v18_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
            int32_t v25_lead = v18_lead + (v19_i0 * 16);
            #pragma unroll
            for (int32_t v20_i1 = 0; v20_i1 < 16; ++v20_i1) {
              double v28_data = __builtin_nontemporal_load(&glb_m2[(v25_lead + (v20_i1 * 16))]);
              r0[(v19_i0 + v20_i1)] = v28_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m2););
          double r1[16]{};
          // r1 = +(glb_m1 * r0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          double v37_data = glb_m1[v18_lead];
          double v44_data = glb_m1[(v18_lead + 16)];
          double v51_data = glb_m1[(v18_lead + 32)];
          double v58_data = glb_m1[(v18_lead + 48)];
          double v65_data = glb_m1[(v18_lead + 64)];
          double v72_data = glb_m1[(v18_lead + 80)];
          double v79_data = glb_m1[(v18_lead + 96)];
          double v86_data = glb_m1[(v18_lead + 112)];
          double v93_data = glb_m1[(v18_lead + 128)];
          double v100_data = glb_m1[(v18_lead + 144)];
          double v107_data = glb_m1[(v18_lead + 160)];
          double v114_data = glb_m1[(v18_lead + 176)];
          double v121_data = glb_m1[(v18_lead + 192)];
          double v128_data = glb_m1[(v18_lead + 208)];
          double v135_data = glb_m1[(v18_lead + 224)];
          double v142_data = glb_m1[(v18_lead + 240)];
          double v143_acc{};
          double v144_acc{};
          double v145_acc{};
          double v146_acc{};
          double v147_acc{};
          double v148_acc{};
          double v149_acc{};
          double v150_acc{};
          double v151_acc{};
          double v152_acc{};
          double v153_acc{};
          double v154_acc{};
          double v155_acc{};
          double v156_acc{};
          double v157_acc{};
          double v158_acc{};
          double v159_data = r0[0];
          double v160_data = r0[1];
          double v161_data = r0[2];
          double v162_data = r0[3];
          double v163_data = r0[4];
          double v164_data = r0[5];
          double v165_data = r0[6];
          double v166_data = r0[7];
          double v167_data = r0[8];
          double v168_data = r0[9];
          double v169_data = r0[10];
          double v170_data = r0[11];
          double v171_data = r0[12];
          double v172_data = r0[13];
          double v173_data = r0[14];
          double v174_data = r0[15];
          tensorforge::fmacdpp16<0>(v143_acc, v159_data, v37_data);
          tensorforge::fmacdpp16<1>(v143_acc, v159_data, v44_data);
          tensorforge::fmacdpp16<2>(v143_acc, v159_data, v51_data);
          tensorforge::fmacdpp16<3>(v143_acc, v159_data, v58_data);
          tensorforge::fmacdpp16<4>(v143_acc, v159_data, v65_data);
          tensorforge::fmacdpp16<5>(v143_acc, v159_data, v72_data);
          tensorforge::fmacdpp16<6>(v143_acc, v159_data, v79_data);
          tensorforge::fmacdpp16<7>(v143_acc, v159_data, v86_data);
          tensorforge::fmacdpp16<8>(v143_acc, v159_data, v93_data);
          tensorforge::fmacdpp16<9>(v143_acc, v159_data, v100_data);
          tensorforge::fmacdpp16<10>(v143_acc, v159_data, v107_data);
          tensorforge::fmacdpp16<11>(v143_acc, v159_data, v114_data);
          tensorforge::fmacdpp16<12>(v143_acc, v159_data, v121_data);
          tensorforge::fmacdpp16<13>(v143_acc, v159_data, v128_data);
          tensorforge::fmacdpp16<14>(v143_acc, v159_data, v135_data);
          tensorforge::fmacdpp16<15>(v143_acc, v159_data, v142_data);
          tensorforge::fmacdpp16<0>(v144_acc, v160_data, v37_data);
          tensorforge::fmacdpp16<1>(v144_acc, v160_data, v44_data);
          tensorforge::fmacdpp16<2>(v144_acc, v160_data, v51_data);
          tensorforge::fmacdpp16<3>(v144_acc, v160_data, v58_data);
          tensorforge::fmacdpp16<4>(v144_acc, v160_data, v65_data);
          tensorforge::fmacdpp16<5>(v144_acc, v160_data, v72_data);
          tensorforge::fmacdpp16<6>(v144_acc, v160_data, v79_data);
          tensorforge::fmacdpp16<7>(v144_acc, v160_data, v86_data);
          tensorforge::fmacdpp16<8>(v144_acc, v160_data, v93_data);
          tensorforge::fmacdpp16<9>(v144_acc, v160_data, v100_data);
          tensorforge::fmacdpp16<10>(v144_acc, v160_data, v107_data);
          tensorforge::fmacdpp16<11>(v144_acc, v160_data, v114_data);
          tensorforge::fmacdpp16<12>(v144_acc, v160_data, v121_data);
          tensorforge::fmacdpp16<13>(v144_acc, v160_data, v128_data);
          tensorforge::fmacdpp16<14>(v144_acc, v160_data, v135_data);
          tensorforge::fmacdpp16<15>(v144_acc, v160_data, v142_data);
          tensorforge::fmacdpp16<0>(v145_acc, v161_data, v37_data);
          tensorforge::fmacdpp16<1>(v145_acc, v161_data, v44_data);
          tensorforge::fmacdpp16<2>(v145_acc, v161_data, v51_data);
          tensorforge::fmacdpp16<3>(v145_acc, v161_data, v58_data);
          tensorforge::fmacdpp16<4>(v145_acc, v161_data, v65_data);
          tensorforge::fmacdpp16<5>(v145_acc, v161_data, v72_data);
          tensorforge::fmacdpp16<6>(v145_acc, v161_data, v79_data);
          tensorforge::fmacdpp16<7>(v145_acc, v161_data, v86_data);
          tensorforge::fmacdpp16<8>(v145_acc, v161_data, v93_data);
          tensorforge::fmacdpp16<9>(v145_acc, v161_data, v100_data);
          tensorforge::fmacdpp16<10>(v145_acc, v161_data, v107_data);
          tensorforge::fmacdpp16<11>(v145_acc, v161_data, v114_data);
          tensorforge::fmacdpp16<12>(v145_acc, v161_data, v121_data);
          tensorforge::fmacdpp16<13>(v145_acc, v161_data, v128_data);
          tensorforge::fmacdpp16<14>(v145_acc, v161_data, v135_data);
          tensorforge::fmacdpp16<15>(v145_acc, v161_data, v142_data);
          tensorforge::fmacdpp16<0>(v146_acc, v162_data, v37_data);
          tensorforge::fmacdpp16<1>(v146_acc, v162_data, v44_data);
          tensorforge::fmacdpp16<2>(v146_acc, v162_data, v51_data);
          tensorforge::fmacdpp16<3>(v146_acc, v162_data, v58_data);
          tensorforge::fmacdpp16<4>(v146_acc, v162_data, v65_data);
          tensorforge::fmacdpp16<5>(v146_acc, v162_data, v72_data);
          tensorforge::fmacdpp16<6>(v146_acc, v162_data, v79_data);
          tensorforge::fmacdpp16<7>(v146_acc, v162_data, v86_data);
          tensorforge::fmacdpp16<8>(v146_acc, v162_data, v93_data);
          tensorforge::fmacdpp16<9>(v146_acc, v162_data, v100_data);
          tensorforge::fmacdpp16<10>(v146_acc, v162_data, v107_data);
          tensorforge::fmacdpp16<11>(v146_acc, v162_data, v114_data);
          tensorforge::fmacdpp16<12>(v146_acc, v162_data, v121_data);
          tensorforge::fmacdpp16<13>(v146_acc, v162_data, v128_data);
          tensorforge::fmacdpp16<14>(v146_acc, v162_data, v135_data);
          tensorforge::fmacdpp16<15>(v146_acc, v162_data, v142_data);
          tensorforge::fmacdpp16<0>(v147_acc, v163_data, v37_data);
          tensorforge::fmacdpp16<1>(v147_acc, v163_data, v44_data);
          tensorforge::fmacdpp16<2>(v147_acc, v163_data, v51_data);
          tensorforge::fmacdpp16<3>(v147_acc, v163_data, v58_data);
          tensorforge::fmacdpp16<4>(v147_acc, v163_data, v65_data);
          tensorforge::fmacdpp16<5>(v147_acc, v163_data, v72_data);
          tensorforge::fmacdpp16<6>(v147_acc, v163_data, v79_data);
          tensorforge::fmacdpp16<7>(v147_acc, v163_data, v86_data);
          tensorforge::fmacdpp16<8>(v147_acc, v163_data, v93_data);
          tensorforge::fmacdpp16<9>(v147_acc, v163_data, v100_data);
          tensorforge::fmacdpp16<10>(v147_acc, v163_data, v107_data);
          tensorforge::fmacdpp16<11>(v147_acc, v163_data, v114_data);
          tensorforge::fmacdpp16<12>(v147_acc, v163_data, v121_data);
          tensorforge::fmacdpp16<13>(v147_acc, v163_data, v128_data);
          tensorforge::fmacdpp16<14>(v147_acc, v163_data, v135_data);
          tensorforge::fmacdpp16<15>(v147_acc, v163_data, v142_data);
          tensorforge::fmacdpp16<0>(v148_acc, v164_data, v37_data);
          tensorforge::fmacdpp16<1>(v148_acc, v164_data, v44_data);
          tensorforge::fmacdpp16<2>(v148_acc, v164_data, v51_data);
          tensorforge::fmacdpp16<3>(v148_acc, v164_data, v58_data);
          tensorforge::fmacdpp16<4>(v148_acc, v164_data, v65_data);
          tensorforge::fmacdpp16<5>(v148_acc, v164_data, v72_data);
          tensorforge::fmacdpp16<6>(v148_acc, v164_data, v79_data);
          tensorforge::fmacdpp16<7>(v148_acc, v164_data, v86_data);
          tensorforge::fmacdpp16<8>(v148_acc, v164_data, v93_data);
          tensorforge::fmacdpp16<9>(v148_acc, v164_data, v100_data);
          tensorforge::fmacdpp16<10>(v148_acc, v164_data, v107_data);
          tensorforge::fmacdpp16<11>(v148_acc, v164_data, v114_data);
          tensorforge::fmacdpp16<12>(v148_acc, v164_data, v121_data);
          tensorforge::fmacdpp16<13>(v148_acc, v164_data, v128_data);
          tensorforge::fmacdpp16<14>(v148_acc, v164_data, v135_data);
          tensorforge::fmacdpp16<15>(v148_acc, v164_data, v142_data);
          tensorforge::fmacdpp16<0>(v149_acc, v165_data, v37_data);
          tensorforge::fmacdpp16<1>(v149_acc, v165_data, v44_data);
          tensorforge::fmacdpp16<2>(v149_acc, v165_data, v51_data);
          tensorforge::fmacdpp16<3>(v149_acc, v165_data, v58_data);
          tensorforge::fmacdpp16<4>(v149_acc, v165_data, v65_data);
          tensorforge::fmacdpp16<5>(v149_acc, v165_data, v72_data);
          tensorforge::fmacdpp16<6>(v149_acc, v165_data, v79_data);
          tensorforge::fmacdpp16<7>(v149_acc, v165_data, v86_data);
          tensorforge::fmacdpp16<8>(v149_acc, v165_data, v93_data);
          tensorforge::fmacdpp16<9>(v149_acc, v165_data, v100_data);
          tensorforge::fmacdpp16<10>(v149_acc, v165_data, v107_data);
          tensorforge::fmacdpp16<11>(v149_acc, v165_data, v114_data);
          tensorforge::fmacdpp16<12>(v149_acc, v165_data, v121_data);
          tensorforge::fmacdpp16<13>(v149_acc, v165_data, v128_data);
          tensorforge::fmacdpp16<14>(v149_acc, v165_data, v135_data);
          tensorforge::fmacdpp16<15>(v149_acc, v165_data, v142_data);
          tensorforge::fmacdpp16<0>(v150_acc, v166_data, v37_data);
          tensorforge::fmacdpp16<1>(v150_acc, v166_data, v44_data);
          tensorforge::fmacdpp16<2>(v150_acc, v166_data, v51_data);
          tensorforge::fmacdpp16<3>(v150_acc, v166_data, v58_data);
          tensorforge::fmacdpp16<4>(v150_acc, v166_data, v65_data);
          tensorforge::fmacdpp16<5>(v150_acc, v166_data, v72_data);
          tensorforge::fmacdpp16<6>(v150_acc, v166_data, v79_data);
          tensorforge::fmacdpp16<7>(v150_acc, v166_data, v86_data);
          tensorforge::fmacdpp16<8>(v150_acc, v166_data, v93_data);
          tensorforge::fmacdpp16<9>(v150_acc, v166_data, v100_data);
          tensorforge::fmacdpp16<10>(v150_acc, v166_data, v107_data);
          tensorforge::fmacdpp16<11>(v150_acc, v166_data, v114_data);
          tensorforge::fmacdpp16<12>(v150_acc, v166_data, v121_data);
          tensorforge::fmacdpp16<13>(v150_acc, v166_data, v128_data);
          tensorforge::fmacdpp16<14>(v150_acc, v166_data, v135_data);
          tensorforge::fmacdpp16<15>(v150_acc, v166_data, v142_data);
          tensorforge::fmacdpp16<0>(v151_acc, v167_data, v37_data);
          tensorforge::fmacdpp16<1>(v151_acc, v167_data, v44_data);
          tensorforge::fmacdpp16<2>(v151_acc, v167_data, v51_data);
          tensorforge::fmacdpp16<3>(v151_acc, v167_data, v58_data);
          tensorforge::fmacdpp16<4>(v151_acc, v167_data, v65_data);
          tensorforge::fmacdpp16<5>(v151_acc, v167_data, v72_data);
          tensorforge::fmacdpp16<6>(v151_acc, v167_data, v79_data);
          tensorforge::fmacdpp16<7>(v151_acc, v167_data, v86_data);
          tensorforge::fmacdpp16<8>(v151_acc, v167_data, v93_data);
          tensorforge::fmacdpp16<9>(v151_acc, v167_data, v100_data);
          tensorforge::fmacdpp16<10>(v151_acc, v167_data, v107_data);
          tensorforge::fmacdpp16<11>(v151_acc, v167_data, v114_data);
          tensorforge::fmacdpp16<12>(v151_acc, v167_data, v121_data);
          tensorforge::fmacdpp16<13>(v151_acc, v167_data, v128_data);
          tensorforge::fmacdpp16<14>(v151_acc, v167_data, v135_data);
          tensorforge::fmacdpp16<15>(v151_acc, v167_data, v142_data);
          tensorforge::fmacdpp16<0>(v152_acc, v168_data, v37_data);
          tensorforge::fmacdpp16<1>(v152_acc, v168_data, v44_data);
          tensorforge::fmacdpp16<2>(v152_acc, v168_data, v51_data);
          tensorforge::fmacdpp16<3>(v152_acc, v168_data, v58_data);
          tensorforge::fmacdpp16<4>(v152_acc, v168_data, v65_data);
          tensorforge::fmacdpp16<5>(v152_acc, v168_data, v72_data);
          tensorforge::fmacdpp16<6>(v152_acc, v168_data, v79_data);
          tensorforge::fmacdpp16<7>(v152_acc, v168_data, v86_data);
          tensorforge::fmacdpp16<8>(v152_acc, v168_data, v93_data);
          tensorforge::fmacdpp16<9>(v152_acc, v168_data, v100_data);
          tensorforge::fmacdpp16<10>(v152_acc, v168_data, v107_data);
          tensorforge::fmacdpp16<11>(v152_acc, v168_data, v114_data);
          tensorforge::fmacdpp16<12>(v152_acc, v168_data, v121_data);
          tensorforge::fmacdpp16<13>(v152_acc, v168_data, v128_data);
          tensorforge::fmacdpp16<14>(v152_acc, v168_data, v135_data);
          tensorforge::fmacdpp16<15>(v152_acc, v168_data, v142_data);
          tensorforge::fmacdpp16<0>(v153_acc, v169_data, v37_data);
          tensorforge::fmacdpp16<1>(v153_acc, v169_data, v44_data);
          tensorforge::fmacdpp16<2>(v153_acc, v169_data, v51_data);
          tensorforge::fmacdpp16<3>(v153_acc, v169_data, v58_data);
          tensorforge::fmacdpp16<4>(v153_acc, v169_data, v65_data);
          tensorforge::fmacdpp16<5>(v153_acc, v169_data, v72_data);
          tensorforge::fmacdpp16<6>(v153_acc, v169_data, v79_data);
          tensorforge::fmacdpp16<7>(v153_acc, v169_data, v86_data);
          tensorforge::fmacdpp16<8>(v153_acc, v169_data, v93_data);
          tensorforge::fmacdpp16<9>(v153_acc, v169_data, v100_data);
          tensorforge::fmacdpp16<10>(v153_acc, v169_data, v107_data);
          tensorforge::fmacdpp16<11>(v153_acc, v169_data, v114_data);
          tensorforge::fmacdpp16<12>(v153_acc, v169_data, v121_data);
          tensorforge::fmacdpp16<13>(v153_acc, v169_data, v128_data);
          tensorforge::fmacdpp16<14>(v153_acc, v169_data, v135_data);
          tensorforge::fmacdpp16<15>(v153_acc, v169_data, v142_data);
          tensorforge::fmacdpp16<0>(v154_acc, v170_data, v37_data);
          tensorforge::fmacdpp16<1>(v154_acc, v170_data, v44_data);
          tensorforge::fmacdpp16<2>(v154_acc, v170_data, v51_data);
          tensorforge::fmacdpp16<3>(v154_acc, v170_data, v58_data);
          tensorforge::fmacdpp16<4>(v154_acc, v170_data, v65_data);
          tensorforge::fmacdpp16<5>(v154_acc, v170_data, v72_data);
          tensorforge::fmacdpp16<6>(v154_acc, v170_data, v79_data);
          tensorforge::fmacdpp16<7>(v154_acc, v170_data, v86_data);
          tensorforge::fmacdpp16<8>(v154_acc, v170_data, v93_data);
          tensorforge::fmacdpp16<9>(v154_acc, v170_data, v100_data);
          tensorforge::fmacdpp16<10>(v154_acc, v170_data, v107_data);
          tensorforge::fmacdpp16<11>(v154_acc, v170_data, v114_data);
          tensorforge::fmacdpp16<12>(v154_acc, v170_data, v121_data);
          tensorforge::fmacdpp16<13>(v154_acc, v170_data, v128_data);
          tensorforge::fmacdpp16<14>(v154_acc, v170_data, v135_data);
          tensorforge::fmacdpp16<15>(v154_acc, v170_data, v142_data);
          tensorforge::fmacdpp16<0>(v155_acc, v171_data, v37_data);
          tensorforge::fmacdpp16<1>(v155_acc, v171_data, v44_data);
          tensorforge::fmacdpp16<2>(v155_acc, v171_data, v51_data);
          tensorforge::fmacdpp16<3>(v155_acc, v171_data, v58_data);
          tensorforge::fmacdpp16<4>(v155_acc, v171_data, v65_data);
          tensorforge::fmacdpp16<5>(v155_acc, v171_data, v72_data);
          tensorforge::fmacdpp16<6>(v155_acc, v171_data, v79_data);
          tensorforge::fmacdpp16<7>(v155_acc, v171_data, v86_data);
          tensorforge::fmacdpp16<8>(v155_acc, v171_data, v93_data);
          tensorforge::fmacdpp16<9>(v155_acc, v171_data, v100_data);
          tensorforge::fmacdpp16<10>(v155_acc, v171_data, v107_data);
          tensorforge::fmacdpp16<11>(v155_acc, v171_data, v114_data);
          tensorforge::fmacdpp16<12>(v155_acc, v171_data, v121_data);
          tensorforge::fmacdpp16<13>(v155_acc, v171_data, v128_data);
          tensorforge::fmacdpp16<14>(v155_acc, v171_data, v135_data);
          tensorforge::fmacdpp16<15>(v155_acc, v171_data, v142_data);
          tensorforge::fmacdpp16<0>(v156_acc, v172_data, v37_data);
          tensorforge::fmacdpp16<1>(v156_acc, v172_data, v44_data);
          tensorforge::fmacdpp16<2>(v156_acc, v172_data, v51_data);
          tensorforge::fmacdpp16<3>(v156_acc, v172_data, v58_data);
          tensorforge::fmacdpp16<4>(v156_acc, v172_data, v65_data);
          tensorforge::fmacdpp16<5>(v156_acc, v172_data, v72_data);
          tensorforge::fmacdpp16<6>(v156_acc, v172_data, v79_data);
          tensorforge::fmacdpp16<7>(v156_acc, v172_data, v86_data);
          tensorforge::fmacdpp16<8>(v156_acc, v172_data, v93_data);
          tensorforge::fmacdpp16<9>(v156_acc, v172_data, v100_data);
          tensorforge::fmacdpp16<10>(v156_acc, v172_data, v107_data);
          tensorforge::fmacdpp16<11>(v156_acc, v172_data, v114_data);
          tensorforge::fmacdpp16<12>(v156_acc, v172_data, v121_data);
          tensorforge::fmacdpp16<13>(v156_acc, v172_data, v128_data);
          tensorforge::fmacdpp16<14>(v156_acc, v172_data, v135_data);
          tensorforge::fmacdpp16<15>(v156_acc, v172_data, v142_data);
          tensorforge::fmacdpp16<0>(v157_acc, v173_data, v37_data);
          tensorforge::fmacdpp16<1>(v157_acc, v173_data, v44_data);
          tensorforge::fmacdpp16<2>(v157_acc, v173_data, v51_data);
          tensorforge::fmacdpp16<3>(v157_acc, v173_data, v58_data);
          tensorforge::fmacdpp16<4>(v157_acc, v173_data, v65_data);
          tensorforge::fmacdpp16<5>(v157_acc, v173_data, v72_data);
          tensorforge::fmacdpp16<6>(v157_acc, v173_data, v79_data);
          tensorforge::fmacdpp16<7>(v157_acc, v173_data, v86_data);
          tensorforge::fmacdpp16<8>(v157_acc, v173_data, v93_data);
          tensorforge::fmacdpp16<9>(v157_acc, v173_data, v100_data);
          tensorforge::fmacdpp16<10>(v157_acc, v173_data, v107_data);
          tensorforge::fmacdpp16<11>(v157_acc, v173_data, v114_data);
          tensorforge::fmacdpp16<12>(v157_acc, v173_data, v121_data);
          tensorforge::fmacdpp16<13>(v157_acc, v173_data, v128_data);
          tensorforge::fmacdpp16<14>(v157_acc, v173_data, v135_data);
          tensorforge::fmacdpp16<15>(v157_acc, v173_data, v142_data);
          tensorforge::fmacdpp16<0>(v158_acc, v174_data, v37_data);
          tensorforge::fmacdpp16<1>(v158_acc, v174_data, v44_data);
          tensorforge::fmacdpp16<2>(v158_acc, v174_data, v51_data);
          tensorforge::fmacdpp16<3>(v158_acc, v174_data, v58_data);
          tensorforge::fmacdpp16<4>(v158_acc, v174_data, v65_data);
          tensorforge::fmacdpp16<5>(v158_acc, v174_data, v72_data);
          tensorforge::fmacdpp16<6>(v158_acc, v174_data, v79_data);
          tensorforge::fmacdpp16<7>(v158_acc, v174_data, v86_data);
          tensorforge::fmacdpp16<8>(v158_acc, v174_data, v93_data);
          tensorforge::fmacdpp16<9>(v158_acc, v174_data, v100_data);
          tensorforge::fmacdpp16<10>(v158_acc, v174_data, v107_data);
          tensorforge::fmacdpp16<11>(v158_acc, v174_data, v114_data);
          tensorforge::fmacdpp16<12>(v158_acc, v174_data, v121_data);
          tensorforge::fmacdpp16<13>(v158_acc, v174_data, v128_data);
          tensorforge::fmacdpp16<14>(v158_acc, v174_data, v135_data);
          tensorforge::fmacdpp16<15>(v158_acc, v174_data, v142_data);
          r1[0] = v143_acc;
          r1[1] = v144_acc;
          r1[2] = v145_acc;
          r1[3] = v146_acc;
          r1[4] = v147_acc;
          r1[5] = v148_acc;
          r1[6] = v149_acc;
          r1[7] = v150_acc;
          r1[8] = v151_acc;
          r1[9] = v152_acc;
          r1[10] = v153_acc;
          r1[11] = v154_acc;
          r1[12] = v155_acc;
          r1[13] = v156_acc;
          r1[14] = v157_acc;
          r1[15] = v158_acc;
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v178_i0 = 0; v178_i0 < 1; ++v178_i0) {
            int32_t v186_lead = v18_lead + (v178_i0 * 16);
            #pragma unroll
            for (int32_t v179_i1 = 0; v179_i1 < 16; ++v179_i1) {
              double v181_data = r1[(v178_i0 + v179_i1)];
              glb_m0[(v186_lead + (v179_i1 * 16))] = v181_data;
            }
          }
        }
      }
    }
  }
}

