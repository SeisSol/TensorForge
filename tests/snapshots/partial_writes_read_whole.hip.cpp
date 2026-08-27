// === base name ===
kernel_7ab185b978

// === header ===
void launcher_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        CHECK_RES(hipGetDevice(&device));
        CHECK_RES(hipDeviceGetAttribute(&smCount, hipDeviceAttributeMultiprocessorCount, device));
        CHECK_RES(hipOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_7ab185b978, block.x * block.y * block.z, 2560 * sizeof(float)));
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
        CHECK_RES(hipFuncSetAttribute(reinterpret_cast<const void*>(&kernel_kernel_7ab185b978), hipFuncAttributeMaxDynamicSharedMemorySize, 2560 * sizeof(float)));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  hipStream_t stream = (streamPtr != nullptr) ? static_cast<hipStream_t>(streamPtr) : 0;
  hipLaunchKernelGGL(kernel_kernel_7ab185b978, grid, block, 2560 * sizeof(float), stream,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_7ab185b978(const float** m0, unsigned m0_extraOffset, const float** m1, unsigned m1_extraOffset, const float** m2, unsigned m2_extraOffset, float** m3, unsigned m3_extraOffset, const float** m4, unsigned m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 32×9(32×9) {0..32}×{0..9} pointer_based
    // m1 16×9(16×9) {0..16}×{0..9} pointer_based
    // m2 16×9(16×9) {0..16}×{0..9} pointer_based
    // m3 32×9(32×9) {0..32}×{0..9} pointer_based
    // m4 9×9(9×9) {0..9}×{0..9} pointer_based
    // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] = m0 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1]
    // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m1 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
    // t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, 1] += m2 16×9(16×9) {0..16}×{0..9} pointer_based({0..16}×{0..9})[0, 1]
    // m3 32×9(32×9) {0..32}×{0..9} pointer_based({0..32}×{0..9})[0, 1] = t0 32×9(32×9) {0..32}×{0..9} strided({0..32}×{0..9})[0, -1]×m4 9×9(9×9) {0..9}×{0..9} pointer_based({0..9}×{0..9})[-1, 1]
    {
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[320 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[320];
      __syncthreads();
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        bool allowed = true;
        if (flags0 != nullptr) {
          allowed = static_cast<bool>(flags0[batchId0]);
        }
        if (allowed) {
          auto glb_m0 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m0[batchId0][0 + m0_extraOffset];
          auto glb_m1 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m1[batchId0][0 + m1_extraOffset];
          auto glb_m2 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m2[batchId0][0 + m2_extraOffset];
          auto glb_m3 = (tensorforge::SpacePtrRestrict<float, tensorforge::GlobalMemspace>)&m3[batchId0][0 + m3_extraOffset];
          auto glb_m4 = (tensorforge::SpacePtrRestrict<const float, tensorforge::GlobalMemspace>)&m4[batchId0][0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v8_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
            int32_t v14_lead = v9_i0 * 32;
            int32_t v15_lead = v8_lead + v14_lead;
            int32_t v22_lead = v8_lead + v14_lead;
            #pragma unroll
            for (int32_t v10_i1 = 0; v10_i1 < 9; ++v10_i1) {
              int32_t v16_a = v10_i1 * 32;
              int32_t v17_a = v15_lead + v16_a;
              float v25_data = __builtin_nontemporal_load(&glb_m0[(v22_lead + v16_a)]);
              int32_t v26_a = v9_i0 + v10_i1;
              r0[v26_a] = v25_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          if (v8_lead < 16) {
            #pragma unroll
            for (int32_t v32_i1 = 0; v32_i1 < 9; ++v32_i1) {
              int32_t v38_a = v32_i1 * 16;
              int32_t v39_a = v8_lead + v38_a;
              float v47_data = __builtin_nontemporal_load(&glb_m1[(v8_lead + v38_a)]);
              int32_t v48_a = 0 + v32_i1;
              r2[v48_a] = v47_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          auto& ir1 = r1;
          float v53_data = r0[0];
          float v54_data = ir1[0];
          ir1[0] = (v54_data + v53_data);
          float v56_data = r0[1];
          float v57_data = ir1[1];
          ir1[1] = (v57_data + v56_data);
          float v59_data = r0[2];
          float v60_data = ir1[2];
          ir1[2] = (v60_data + v59_data);
          float v62_data = r0[3];
          float v63_data = ir1[3];
          ir1[3] = (v63_data + v62_data);
          float v65_data = r0[4];
          float v66_data = ir1[4];
          ir1[4] = (v66_data + v65_data);
          float v68_data = r0[5];
          float v69_data = ir1[5];
          ir1[5] = (v69_data + v68_data);
          float v71_data = r0[6];
          float v72_data = ir1[6];
          ir1[6] = (v72_data + v71_data);
          float v74_data = r0[7];
          float v75_data = ir1[7];
          ir1[7] = (v75_data + v74_data);
          float v77_data = r0[8];
          float v78_data = ir1[8];
          ir1[8] = (v78_data + v77_data);
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v84_i0 = 0; v84_i0 < 1; ++v84_i0) {
            int32_t v93_lead = v8_lead + (v84_i0 * 32);
            #pragma unroll
            for (int32_t v85_i1 = 0; v85_i1 < 9; ++v85_i1) {
              int32_t v86_a = v84_i0 + v85_i1;
              float v88_data = r1[(v84_i0 + v85_i1)];
              int32_t v95_a = v93_lead + (v85_i1 * 32);
              s0[v95_a] = v88_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v8_lead < 16) {
            #pragma unroll
            for (int32_t v101_i1 = 0; v101_i1 < 9; ++v101_i1) {
              int32_t v107_a = v101_i1 * 16;
              int32_t v108_a = v8_lead + v107_a;
              float v116_data = __builtin_nontemporal_load(&glb_m2[(v8_lead + v107_a)]);
              int32_t v117_a = 0 + v101_i1;
              r4[v117_a] = v116_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          // r3 = +(r2) + None
          // [(0, 16), (0, 9)] []
          auto& ir3 = r3;
          if (v8_lead < 16) {
            float v123_data = r2[0];
            float v124_data = ir3[0];
            ir3[0] = (v124_data + v123_data);
            float v126_data = r2[1];
            float v127_data = ir3[1];
            ir3[1] = (v127_data + v126_data);
            float v129_data = r2[2];
            float v130_data = ir3[2];
            ir3[2] = (v130_data + v129_data);
            float v132_data = r2[3];
            float v133_data = ir3[3];
            ir3[3] = (v133_data + v132_data);
            float v135_data = r2[4];
            float v136_data = ir3[4];
            ir3[4] = (v136_data + v135_data);
            float v138_data = r2[5];
            float v139_data = ir3[5];
            ir3[5] = (v139_data + v138_data);
            float v141_data = r2[6];
            float v142_data = ir3[6];
            ir3[6] = (v142_data + v141_data);
            float v144_data = r2[7];
            float v145_data = ir3[7];
            ir3[7] = (v145_data + v144_data);
            float v147_data = r2[8];
            float v148_data = ir3[8];
            ir3[8] = (v148_data + v147_data);
          }
          // s0 = store{r>s}(localShrMem0, r3);
          if (v8_lead < 16) {
            #pragma unroll
            for (int32_t v154_i1 = 0; v154_i1 < 9; ++v154_i1) {
              int32_t v155_a = 0 + v154_i1;
              float v157_data = r3[v154_i1];
              int32_t v164_a = v8_lead + (v154_i1 * 32);
              s0[v164_a] = v157_data;
            }
          }
          float r6[9]{};
          // r6 = load{g>r}(glb_m4);
          float v166_lin = glb_m4[0 + threadIdx.x * 1];
          r6[0] = v166_lin;
          float v167_lin = glb_m4[32 + threadIdx.x * 1];
          r6[1] = v167_lin;
          float v168_lin = glb_m4[64 + threadIdx.x * 1];
          r6[2] = v168_lin;
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          // r5 = +(r4) + None
          // [(0, 16), (0, 9)] []
          auto& ir5 = r5;
          if (v8_lead < 16) {
            float v174_data = r4[0];
            float v175_data = ir5[0];
            ir5[0] = (v175_data + v174_data);
            float v177_data = r4[1];
            float v178_data = ir5[1];
            ir5[1] = (v178_data + v177_data);
            float v180_data = r4[2];
            float v181_data = ir5[2];
            ir5[2] = (v181_data + v180_data);
            float v183_data = r4[3];
            float v184_data = ir5[3];
            ir5[3] = (v184_data + v183_data);
            float v186_data = r4[4];
            float v187_data = ir5[4];
            ir5[4] = (v187_data + v186_data);
            float v189_data = r4[5];
            float v190_data = ir5[5];
            ir5[5] = (v190_data + v189_data);
            float v192_data = r4[6];
            float v193_data = ir5[6];
            ir5[6] = (v193_data + v192_data);
            float v195_data = r4[7];
            float v196_data = ir5[7];
            ir5[7] = (v196_data + v195_data);
            float v198_data = r4[8];
            float v199_data = ir5[8];
            ir5[8] = (v199_data + v198_data);
          }
          // s0 = store{r>s}(localShrMem0, r5);
          if (v8_lead < 16) {
            #pragma unroll
            for (int32_t v205_i1 = 0; v205_i1 < 9; ++v205_i1) {
              int32_t v206_a = 0 + v205_i1;
              float v208_data = r5[v205_i1];
              int32_t v215_a = v8_lead + (v205_i1 * 32);
              s0[v215_a] = v208_data;
            }
          }
          // wait(r6 = load{g>r}(glb_m4););
          float r7[9]{};
          ;
          // r7 = +(s0 * r6) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          auto& ir7 = r7;
          int32_t v222_a = v8_lead + 0;
          float v229_data = s0[v8_lead];
          int32_t v235_a = v8_lead + 32;
          float v242_data = s0[(v8_lead + 32)];
          int32_t v248_a = v8_lead + 64;
          float v255_data = s0[(v8_lead + 64)];
          int32_t v261_a = v8_lead + 96;
          float v268_data = s0[(v8_lead + 96)];
          int32_t v274_a = v8_lead + 128;
          float v281_data = s0[(v8_lead + 128)];
          int32_t v287_a = v8_lead + 160;
          float v294_data = s0[(v8_lead + 160)];
          int32_t v300_a = v8_lead + 192;
          float v307_data = s0[(v8_lead + 192)];
          int32_t v313_a = v8_lead + 224;
          float v320_data = s0[(v8_lead + 224)];
          int32_t v326_a = v8_lead + 256;
          float v333_data = s0[(v8_lead + 256)];
          float v334_acc{};
          float v335_acc{};
          float v336_acc{};
          float v337_acc{};
          float v338_acc{};
          float v339_acc{};
          float v340_acc{};
          float v341_acc{};
          float v342_acc{};
          float v343_lin = r6[0];
          float v344_bc = tensorforge::broadcast<32, 16, 0>(v343_lin);
          tensorforge::fmacdpp16<0>(v334_acc, v344_bc, v229_data);
          tensorforge::fmacdpp16<1>(v334_acc, v344_bc, v242_data);
          tensorforge::fmacdpp16<2>(v334_acc, v344_bc, v255_data);
          tensorforge::fmacdpp16<3>(v334_acc, v344_bc, v268_data);
          tensorforge::fmacdpp16<4>(v334_acc, v344_bc, v281_data);
          tensorforge::fmacdpp16<5>(v334_acc, v344_bc, v294_data);
          tensorforge::fmacdpp16<6>(v334_acc, v344_bc, v307_data);
          tensorforge::fmacdpp16<7>(v334_acc, v344_bc, v320_data);
          tensorforge::fmacdpp16<8>(v334_acc, v344_bc, v333_data);
          tensorforge::fmacdpp16<9>(v335_acc, v344_bc, v229_data);
          tensorforge::fmacdpp16<10>(v335_acc, v344_bc, v242_data);
          tensorforge::fmacdpp16<11>(v335_acc, v344_bc, v255_data);
          tensorforge::fmacdpp16<12>(v335_acc, v344_bc, v268_data);
          tensorforge::fmacdpp16<13>(v335_acc, v344_bc, v281_data);
          tensorforge::fmacdpp16<14>(v335_acc, v344_bc, v294_data);
          tensorforge::fmacdpp16<15>(v335_acc, v344_bc, v307_data);
          float v345_bc = tensorforge::broadcast<32, 16, 1>(v343_lin);
          tensorforge::fmacdpp16<0>(v335_acc, v345_bc, v320_data);
          tensorforge::fmacdpp16<1>(v335_acc, v345_bc, v333_data);
          tensorforge::fmacdpp16<2>(v336_acc, v345_bc, v229_data);
          tensorforge::fmacdpp16<3>(v336_acc, v345_bc, v242_data);
          tensorforge::fmacdpp16<4>(v336_acc, v345_bc, v255_data);
          tensorforge::fmacdpp16<5>(v336_acc, v345_bc, v268_data);
          tensorforge::fmacdpp16<6>(v336_acc, v345_bc, v281_data);
          tensorforge::fmacdpp16<7>(v336_acc, v345_bc, v294_data);
          tensorforge::fmacdpp16<8>(v336_acc, v345_bc, v307_data);
          tensorforge::fmacdpp16<9>(v336_acc, v345_bc, v320_data);
          tensorforge::fmacdpp16<10>(v336_acc, v345_bc, v333_data);
          tensorforge::fmacdpp16<11>(v337_acc, v345_bc, v229_data);
          tensorforge::fmacdpp16<12>(v337_acc, v345_bc, v242_data);
          tensorforge::fmacdpp16<13>(v337_acc, v345_bc, v255_data);
          tensorforge::fmacdpp16<14>(v337_acc, v345_bc, v268_data);
          tensorforge::fmacdpp16<15>(v337_acc, v345_bc, v281_data);
          float v346_lin = r6[1];
          float v347_bc = tensorforge::broadcast<32, 16, 0>(v346_lin);
          tensorforge::fmacdpp16<0>(v337_acc, v347_bc, v294_data);
          tensorforge::fmacdpp16<1>(v337_acc, v347_bc, v307_data);
          tensorforge::fmacdpp16<2>(v337_acc, v347_bc, v320_data);
          tensorforge::fmacdpp16<3>(v337_acc, v347_bc, v333_data);
          tensorforge::fmacdpp16<4>(v338_acc, v347_bc, v229_data);
          tensorforge::fmacdpp16<5>(v338_acc, v347_bc, v242_data);
          tensorforge::fmacdpp16<6>(v338_acc, v347_bc, v255_data);
          tensorforge::fmacdpp16<7>(v338_acc, v347_bc, v268_data);
          tensorforge::fmacdpp16<8>(v338_acc, v347_bc, v281_data);
          tensorforge::fmacdpp16<9>(v338_acc, v347_bc, v294_data);
          tensorforge::fmacdpp16<10>(v338_acc, v347_bc, v307_data);
          tensorforge::fmacdpp16<11>(v338_acc, v347_bc, v320_data);
          tensorforge::fmacdpp16<12>(v338_acc, v347_bc, v333_data);
          tensorforge::fmacdpp16<13>(v339_acc, v347_bc, v229_data);
          tensorforge::fmacdpp16<14>(v339_acc, v347_bc, v242_data);
          tensorforge::fmacdpp16<15>(v339_acc, v347_bc, v255_data);
          float v348_bc = tensorforge::broadcast<32, 16, 1>(v346_lin);
          tensorforge::fmacdpp16<0>(v339_acc, v348_bc, v268_data);
          tensorforge::fmacdpp16<1>(v339_acc, v348_bc, v281_data);
          tensorforge::fmacdpp16<2>(v339_acc, v348_bc, v294_data);
          tensorforge::fmacdpp16<3>(v339_acc, v348_bc, v307_data);
          tensorforge::fmacdpp16<4>(v339_acc, v348_bc, v320_data);
          tensorforge::fmacdpp16<5>(v339_acc, v348_bc, v333_data);
          tensorforge::fmacdpp16<6>(v340_acc, v348_bc, v229_data);
          tensorforge::fmacdpp16<7>(v340_acc, v348_bc, v242_data);
          tensorforge::fmacdpp16<8>(v340_acc, v348_bc, v255_data);
          tensorforge::fmacdpp16<9>(v340_acc, v348_bc, v268_data);
          tensorforge::fmacdpp16<10>(v340_acc, v348_bc, v281_data);
          tensorforge::fmacdpp16<11>(v340_acc, v348_bc, v294_data);
          tensorforge::fmacdpp16<12>(v340_acc, v348_bc, v307_data);
          tensorforge::fmacdpp16<13>(v340_acc, v348_bc, v320_data);
          tensorforge::fmacdpp16<14>(v340_acc, v348_bc, v333_data);
          tensorforge::fmacdpp16<15>(v341_acc, v348_bc, v229_data);
          float v349_lin = r6[2];
          float v350_bc = tensorforge::broadcast<32, 16, 0>(v349_lin);
          tensorforge::fmacdpp16<0>(v341_acc, v350_bc, v242_data);
          tensorforge::fmacdpp16<1>(v341_acc, v350_bc, v255_data);
          tensorforge::fmacdpp16<2>(v341_acc, v350_bc, v268_data);
          tensorforge::fmacdpp16<3>(v341_acc, v350_bc, v281_data);
          tensorforge::fmacdpp16<4>(v341_acc, v350_bc, v294_data);
          tensorforge::fmacdpp16<5>(v341_acc, v350_bc, v307_data);
          tensorforge::fmacdpp16<6>(v341_acc, v350_bc, v320_data);
          tensorforge::fmacdpp16<7>(v341_acc, v350_bc, v333_data);
          tensorforge::fmacdpp16<8>(v342_acc, v350_bc, v229_data);
          tensorforge::fmacdpp16<9>(v342_acc, v350_bc, v242_data);
          tensorforge::fmacdpp16<10>(v342_acc, v350_bc, v255_data);
          tensorforge::fmacdpp16<11>(v342_acc, v350_bc, v268_data);
          tensorforge::fmacdpp16<12>(v342_acc, v350_bc, v281_data);
          tensorforge::fmacdpp16<13>(v342_acc, v350_bc, v294_data);
          tensorforge::fmacdpp16<14>(v342_acc, v350_bc, v307_data);
          tensorforge::fmacdpp16<15>(v342_acc, v350_bc, v320_data);
          tensorforge::fmacdpp16<0>(v342_acc, (tensorforge::broadcast<32, 16, 1>(v349_lin)), v333_data);
          ir7[0] = v334_acc;
          ir7[1] = v335_acc;
          ir7[2] = v336_acc;
          ir7[3] = v337_acc;
          ir7[4] = v338_acc;
          ir7[5] = v339_acc;
          ir7[6] = v340_acc;
          ir7[7] = v341_acc;
          ir7[8] = v342_acc;
          // glb_m3 = store{r>g}(r7);
          #pragma unroll
          for (int32_t v355_i0 = 0; v355_i0 < 1; ++v355_i0) {
            int32_t v364_lead = v8_lead + (v355_i0 * 32);
            #pragma unroll
            for (int32_t v356_i1 = 0; v356_i1 < 9; ++v356_i1) {
              int32_t v357_a = v355_i0 + v356_i1;
              float v359_data = r7[(v355_i0 + v356_i1)];
              int32_t v366_a = v364_lead + (v356_i1 * 32);
              glb_m3[v366_a] = v359_data;
            }
          }
          ;
        }
      }
    }
  }
}

