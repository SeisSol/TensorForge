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
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_7ab185b978, block.x * block.y * block.z, 3072 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_7ab185b978, cudaFuncAttributeMaxDynamicSharedMemorySize, 3072 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_7ab185b978<<<grid,block,3072 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
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
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[384 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[384];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[batchId0][0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0][0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0][0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[batchId0][0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[batchId0][0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v12_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
            int32_t v19_lead = v12_lead + (v13_i0 * 32);
            #pragma unroll
            for (int32_t v14_i1 = 0; v14_i1 < 9; ++v14_i1) {
              float v22_data = __ldcg(&glb_m0[(v19_lead + (v14_i1 * 32))]);
              r0[(v13_i0 + v14_i1)] = v22_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v29_i1 = 0; v29_i1 < 9; ++v29_i1) {
              float v37_data = __ldcg(&glb_m1[(v12_lead + (v29_i1 * 16))]);
              r2[v29_i1] = v37_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          float v43_data = r0[0];
          float v44_data = r1[0];
          r1[0] = (v44_data + v43_data);
          float v46_data = r0[1];
          float v47_data = r1[1];
          r1[1] = (v47_data + v46_data);
          float v49_data = r0[2];
          float v50_data = r1[2];
          r1[2] = (v50_data + v49_data);
          float v52_data = r0[3];
          float v53_data = r1[3];
          r1[3] = (v53_data + v52_data);
          float v55_data = r0[4];
          float v56_data = r1[4];
          r1[4] = (v56_data + v55_data);
          float v58_data = r0[5];
          float v59_data = r1[5];
          r1[5] = (v59_data + v58_data);
          float v61_data = r0[6];
          float v62_data = r1[6];
          r1[6] = (v62_data + v61_data);
          float v64_data = r0[7];
          float v65_data = r1[7];
          r1[7] = (v65_data + v64_data);
          float v67_data = r0[8];
          float v68_data = r1[8];
          r1[8] = (v68_data + v67_data);
          float* __restrict__ s0 = &localShrMem0[96];
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v74_i0 = 0; v74_i0 < 1; ++v74_i0) {
            int32_t v82_lead = v12_lead + (v74_i0 * 32);
            #pragma unroll
            for (int32_t v75_i1 = 0; v75_i1 < 9; ++v75_i1) {
              float v77_data = r1[(v74_i0 + v75_i1)];
              int32_t v84_a = v82_lead + (v75_i1 * 32);
              s0[(v84_a ^ ((v84_a >> 5) & 31))] = v77_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v93_i1 = 0; v93_i1 < 9; ++v93_i1) {
              float v101_data = __ldcg(&glb_m2[(v12_lead + (v93_i1 * 16))]);
              r4[v93_i1] = v101_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          __syncwarp();
          // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir3[9]{};
          if (v12_lead < 16) {
            float v109_data = r2[0];
            float v110_data = ir3[0];
            ir3[0] = (v110_data + v109_data);
            float v112_data = r2[1];
            float v113_data = ir3[1];
            ir3[1] = (v113_data + v112_data);
            float v115_data = r2[2];
            float v116_data = ir3[2];
            ir3[2] = (v116_data + v115_data);
            float v118_data = r2[3];
            float v119_data = ir3[3];
            ir3[3] = (v119_data + v118_data);
            float v121_data = r2[4];
            float v122_data = ir3[4];
            ir3[4] = (v122_data + v121_data);
            float v124_data = r2[5];
            float v125_data = ir3[5];
            ir3[5] = (v125_data + v124_data);
            float v127_data = r2[6];
            float v128_data = ir3[6];
            ir3[6] = (v128_data + v127_data);
            float v130_data = r2[7];
            float v131_data = ir3[7];
            ir3[7] = (v131_data + v130_data);
            float v133_data = r2[8];
            float v134_data = ir3[8];
            ir3[8] = (v134_data + v133_data);
          }
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v140_n1 = 0; v140_n1 < 9; ++v140_n1) {
              float v142_data = ir3[v140_n1];
              int32_t v149_a = v12_lead + (v140_n1 * 32);
              float v153_data = s0[(v149_a ^ ((v149_a >> 5) & 31))];
              r3[v140_n1] = (v153_data + v142_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r3);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v160_i1 = 0; v160_i1 < 9; ++v160_i1) {
              float v162_data = r3[v160_i1];
              int32_t v169_a = v12_lead + (v160_i1 * 32);
              s0[(v169_a ^ ((v169_a >> 5) & 31))] = v162_data;
            }
          }
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          __syncwarp();
          // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir5[9]{};
          if (v12_lead < 16) {
            float v179_data = r4[0];
            float v180_data = ir5[0];
            ir5[0] = (v180_data + v179_data);
            float v182_data = r4[1];
            float v183_data = ir5[1];
            ir5[1] = (v183_data + v182_data);
            float v185_data = r4[2];
            float v186_data = ir5[2];
            ir5[2] = (v186_data + v185_data);
            float v188_data = r4[3];
            float v189_data = ir5[3];
            ir5[3] = (v189_data + v188_data);
            float v191_data = r4[4];
            float v192_data = ir5[4];
            ir5[4] = (v192_data + v191_data);
            float v194_data = r4[5];
            float v195_data = ir5[5];
            ir5[5] = (v195_data + v194_data);
            float v197_data = r4[6];
            float v198_data = ir5[6];
            ir5[6] = (v198_data + v197_data);
            float v200_data = r4[7];
            float v201_data = ir5[7];
            ir5[7] = (v201_data + v200_data);
            float v203_data = r4[8];
            float v204_data = ir5[8];
            ir5[8] = (v204_data + v203_data);
          }
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v210_n1 = 0; v210_n1 < 9; ++v210_n1) {
              float v212_data = ir5[v210_n1];
              int32_t v219_a = v12_lead + (v210_n1 * 32);
              float v223_data = s0[(v219_a ^ ((v219_a >> 5) & 31))];
              r5[v210_n1] = (v223_data + v212_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r5);
          if (v12_lead < 16) {
            #pragma unroll
            for (int32_t v230_i1 = 0; v230_i1 < 9; ++v230_i1) {
              float v232_data = r5[v230_i1];
              int32_t v239_a = v12_lead + (v230_i1 * 32);
              s0[(v239_a ^ ((v239_a >> 5) & 31))] = v232_data;
            }
          }
          float* __restrict__ s1 = &localShrMem0[0];
          // s1 = load{g>s}(glb_m4[0, 1])
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 0], &glb_m4[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 32], &glb_m4[0 + 0 + 1 * threadIdx.x + 32], 4);
          __pipeline_commit();
          if (threadIdx.x < 17) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 64], &glb_m4[0 + 0 + 1 * threadIdx.x + 64], 4);
            __pipeline_commit();
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r6[9]{};
          __syncwarp();
          // r6 = +(s0 * s1) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          float ir6[9]{};
          float v261_data = s0[(v12_lead ^ ((v12_lead >> 5) & 31))];
          float v262_data = s1[0];
          float v264_data = ir6[0];
          ir6[0] = (v264_data + (v261_data * v262_data));
          float v275_data = s0[(v12_lead ^ ((v12_lead >> 5) & 31))];
          float v276_data = s1[9];
          float v278_data = ir6[1];
          ir6[1] = (v278_data + (v275_data * v276_data));
          float v289_data = s0[(v12_lead ^ ((v12_lead >> 5) & 31))];
          float v290_data = s1[18];
          float v292_data = ir6[2];
          ir6[2] = (v292_data + (v289_data * v290_data));
          float v303_data = s0[(v12_lead ^ ((v12_lead >> 5) & 31))];
          float v304_data = s1[27];
          float v306_data = ir6[3];
          ir6[3] = (v306_data + (v303_data * v304_data));
          float v317_data = s0[(v12_lead ^ ((v12_lead >> 5) & 31))];
          float v318_data = s1[36];
          float v320_data = ir6[4];
          ir6[4] = (v320_data + (v317_data * v318_data));
          float v331_data = s0[(v12_lead ^ ((v12_lead >> 5) & 31))];
          float v332_data = s1[45];
          float v334_data = ir6[5];
          ir6[5] = (v334_data + (v331_data * v332_data));
          float v345_data = s0[(v12_lead ^ ((v12_lead >> 5) & 31))];
          float v346_data = s1[54];
          float v348_data = ir6[6];
          ir6[6] = (v348_data + (v345_data * v346_data));
          float v359_data = s0[(v12_lead ^ ((v12_lead >> 5) & 31))];
          float v360_data = s1[63];
          float v362_data = ir6[7];
          ir6[7] = (v362_data + (v359_data * v360_data));
          float v373_data = s0[(v12_lead ^ ((v12_lead >> 5) & 31))];
          float v374_data = s1[72];
          float v376_data = ir6[8];
          ir6[8] = (v376_data + (v373_data * v374_data));
          int32_t v386_a = v12_lead + 32;
          float v390_data = s0[(v386_a ^ ((v386_a >> 5) & 31))];
          float v391_data = s1[1];
          float v393_data = ir6[0];
          ir6[0] = (v393_data + (v390_data * v391_data));
          int32_t v400_a = v12_lead + 32;
          float v404_data = s0[(v400_a ^ ((v400_a >> 5) & 31))];
          float v405_data = s1[10];
          float v407_data = ir6[1];
          ir6[1] = (v407_data + (v404_data * v405_data));
          int32_t v414_a = v12_lead + 32;
          float v418_data = s0[(v414_a ^ ((v414_a >> 5) & 31))];
          float v419_data = s1[19];
          float v421_data = ir6[2];
          ir6[2] = (v421_data + (v418_data * v419_data));
          int32_t v428_a = v12_lead + 32;
          float v432_data = s0[(v428_a ^ ((v428_a >> 5) & 31))];
          float v433_data = s1[28];
          float v435_data = ir6[3];
          ir6[3] = (v435_data + (v432_data * v433_data));
          int32_t v442_a = v12_lead + 32;
          float v446_data = s0[(v442_a ^ ((v442_a >> 5) & 31))];
          float v447_data = s1[37];
          float v449_data = ir6[4];
          ir6[4] = (v449_data + (v446_data * v447_data));
          int32_t v456_a = v12_lead + 32;
          float v460_data = s0[(v456_a ^ ((v456_a >> 5) & 31))];
          float v461_data = s1[46];
          float v463_data = ir6[5];
          ir6[5] = (v463_data + (v460_data * v461_data));
          int32_t v470_a = v12_lead + 32;
          float v474_data = s0[(v470_a ^ ((v470_a >> 5) & 31))];
          float v475_data = s1[55];
          float v477_data = ir6[6];
          ir6[6] = (v477_data + (v474_data * v475_data));
          int32_t v484_a = v12_lead + 32;
          float v488_data = s0[(v484_a ^ ((v484_a >> 5) & 31))];
          float v489_data = s1[64];
          float v491_data = ir6[7];
          ir6[7] = (v491_data + (v488_data * v489_data));
          int32_t v498_a = v12_lead + 32;
          float v502_data = s0[(v498_a ^ ((v498_a >> 5) & 31))];
          float v503_data = s1[73];
          float v505_data = ir6[8];
          ir6[8] = (v505_data + (v502_data * v503_data));
          int32_t v515_a = v12_lead + 64;
          float v519_data = s0[(v515_a ^ ((v515_a >> 5) & 31))];
          float v520_data = s1[2];
          float v522_data = ir6[0];
          ir6[0] = (v522_data + (v519_data * v520_data));
          int32_t v529_a = v12_lead + 64;
          float v533_data = s0[(v529_a ^ ((v529_a >> 5) & 31))];
          float v534_data = s1[11];
          float v536_data = ir6[1];
          ir6[1] = (v536_data + (v533_data * v534_data));
          int32_t v543_a = v12_lead + 64;
          float v547_data = s0[(v543_a ^ ((v543_a >> 5) & 31))];
          float v548_data = s1[20];
          float v550_data = ir6[2];
          ir6[2] = (v550_data + (v547_data * v548_data));
          int32_t v557_a = v12_lead + 64;
          float v561_data = s0[(v557_a ^ ((v557_a >> 5) & 31))];
          float v562_data = s1[29];
          float v564_data = ir6[3];
          ir6[3] = (v564_data + (v561_data * v562_data));
          int32_t v571_a = v12_lead + 64;
          float v575_data = s0[(v571_a ^ ((v571_a >> 5) & 31))];
          float v576_data = s1[38];
          float v578_data = ir6[4];
          ir6[4] = (v578_data + (v575_data * v576_data));
          int32_t v585_a = v12_lead + 64;
          float v589_data = s0[(v585_a ^ ((v585_a >> 5) & 31))];
          float v590_data = s1[47];
          float v592_data = ir6[5];
          ir6[5] = (v592_data + (v589_data * v590_data));
          int32_t v599_a = v12_lead + 64;
          float v603_data = s0[(v599_a ^ ((v599_a >> 5) & 31))];
          float v604_data = s1[56];
          float v606_data = ir6[6];
          ir6[6] = (v606_data + (v603_data * v604_data));
          int32_t v613_a = v12_lead + 64;
          float v617_data = s0[(v613_a ^ ((v613_a >> 5) & 31))];
          float v618_data = s1[65];
          float v620_data = ir6[7];
          ir6[7] = (v620_data + (v617_data * v618_data));
          int32_t v627_a = v12_lead + 64;
          float v631_data = s0[(v627_a ^ ((v627_a >> 5) & 31))];
          float v632_data = s1[74];
          float v634_data = ir6[8];
          ir6[8] = (v634_data + (v631_data * v632_data));
          int32_t v644_a = v12_lead + 96;
          float v648_data = s0[(v644_a ^ ((v644_a >> 5) & 31))];
          float v649_data = s1[3];
          float v651_data = ir6[0];
          ir6[0] = (v651_data + (v648_data * v649_data));
          int32_t v658_a = v12_lead + 96;
          float v662_data = s0[(v658_a ^ ((v658_a >> 5) & 31))];
          float v663_data = s1[12];
          float v665_data = ir6[1];
          ir6[1] = (v665_data + (v662_data * v663_data));
          int32_t v672_a = v12_lead + 96;
          float v676_data = s0[(v672_a ^ ((v672_a >> 5) & 31))];
          float v677_data = s1[21];
          float v679_data = ir6[2];
          ir6[2] = (v679_data + (v676_data * v677_data));
          int32_t v686_a = v12_lead + 96;
          float v690_data = s0[(v686_a ^ ((v686_a >> 5) & 31))];
          float v691_data = s1[30];
          float v693_data = ir6[3];
          ir6[3] = (v693_data + (v690_data * v691_data));
          int32_t v700_a = v12_lead + 96;
          float v704_data = s0[(v700_a ^ ((v700_a >> 5) & 31))];
          float v705_data = s1[39];
          float v707_data = ir6[4];
          ir6[4] = (v707_data + (v704_data * v705_data));
          int32_t v714_a = v12_lead + 96;
          float v718_data = s0[(v714_a ^ ((v714_a >> 5) & 31))];
          float v719_data = s1[48];
          float v721_data = ir6[5];
          ir6[5] = (v721_data + (v718_data * v719_data));
          int32_t v728_a = v12_lead + 96;
          float v732_data = s0[(v728_a ^ ((v728_a >> 5) & 31))];
          float v733_data = s1[57];
          float v735_data = ir6[6];
          ir6[6] = (v735_data + (v732_data * v733_data));
          int32_t v742_a = v12_lead + 96;
          float v746_data = s0[(v742_a ^ ((v742_a >> 5) & 31))];
          float v747_data = s1[66];
          float v749_data = ir6[7];
          ir6[7] = (v749_data + (v746_data * v747_data));
          int32_t v756_a = v12_lead + 96;
          float v760_data = s0[(v756_a ^ ((v756_a >> 5) & 31))];
          float v761_data = s1[75];
          float v763_data = ir6[8];
          ir6[8] = (v763_data + (v760_data * v761_data));
          int32_t v773_a = v12_lead + 128;
          float v777_data = s0[(v773_a ^ ((v773_a >> 5) & 31))];
          float v778_data = s1[4];
          float v780_data = ir6[0];
          ir6[0] = (v780_data + (v777_data * v778_data));
          int32_t v787_a = v12_lead + 128;
          float v791_data = s0[(v787_a ^ ((v787_a >> 5) & 31))];
          float v792_data = s1[13];
          float v794_data = ir6[1];
          ir6[1] = (v794_data + (v791_data * v792_data));
          int32_t v801_a = v12_lead + 128;
          float v805_data = s0[(v801_a ^ ((v801_a >> 5) & 31))];
          float v806_data = s1[22];
          float v808_data = ir6[2];
          ir6[2] = (v808_data + (v805_data * v806_data));
          int32_t v815_a = v12_lead + 128;
          float v819_data = s0[(v815_a ^ ((v815_a >> 5) & 31))];
          float v820_data = s1[31];
          float v822_data = ir6[3];
          ir6[3] = (v822_data + (v819_data * v820_data));
          int32_t v829_a = v12_lead + 128;
          float v833_data = s0[(v829_a ^ ((v829_a >> 5) & 31))];
          float v834_data = s1[40];
          float v836_data = ir6[4];
          ir6[4] = (v836_data + (v833_data * v834_data));
          int32_t v843_a = v12_lead + 128;
          float v847_data = s0[(v843_a ^ ((v843_a >> 5) & 31))];
          float v848_data = s1[49];
          float v850_data = ir6[5];
          ir6[5] = (v850_data + (v847_data * v848_data));
          int32_t v857_a = v12_lead + 128;
          float v861_data = s0[(v857_a ^ ((v857_a >> 5) & 31))];
          float v862_data = s1[58];
          float v864_data = ir6[6];
          ir6[6] = (v864_data + (v861_data * v862_data));
          int32_t v871_a = v12_lead + 128;
          float v875_data = s0[(v871_a ^ ((v871_a >> 5) & 31))];
          float v876_data = s1[67];
          float v878_data = ir6[7];
          ir6[7] = (v878_data + (v875_data * v876_data));
          int32_t v885_a = v12_lead + 128;
          float v889_data = s0[(v885_a ^ ((v885_a >> 5) & 31))];
          float v890_data = s1[76];
          float v892_data = ir6[8];
          ir6[8] = (v892_data + (v889_data * v890_data));
          int32_t v902_a = v12_lead + 160;
          float v906_data = s0[(v902_a ^ ((v902_a >> 5) & 31))];
          float v907_data = s1[5];
          float v909_data = ir6[0];
          ir6[0] = (v909_data + (v906_data * v907_data));
          int32_t v916_a = v12_lead + 160;
          float v920_data = s0[(v916_a ^ ((v916_a >> 5) & 31))];
          float v921_data = s1[14];
          float v923_data = ir6[1];
          ir6[1] = (v923_data + (v920_data * v921_data));
          int32_t v930_a = v12_lead + 160;
          float v934_data = s0[(v930_a ^ ((v930_a >> 5) & 31))];
          float v935_data = s1[23];
          float v937_data = ir6[2];
          ir6[2] = (v937_data + (v934_data * v935_data));
          int32_t v944_a = v12_lead + 160;
          float v948_data = s0[(v944_a ^ ((v944_a >> 5) & 31))];
          float v949_data = s1[32];
          float v951_data = ir6[3];
          ir6[3] = (v951_data + (v948_data * v949_data));
          int32_t v958_a = v12_lead + 160;
          float v962_data = s0[(v958_a ^ ((v958_a >> 5) & 31))];
          float v963_data = s1[41];
          float v965_data = ir6[4];
          ir6[4] = (v965_data + (v962_data * v963_data));
          int32_t v972_a = v12_lead + 160;
          float v976_data = s0[(v972_a ^ ((v972_a >> 5) & 31))];
          float v977_data = s1[50];
          float v979_data = ir6[5];
          ir6[5] = (v979_data + (v976_data * v977_data));
          int32_t v986_a = v12_lead + 160;
          float v990_data = s0[(v986_a ^ ((v986_a >> 5) & 31))];
          float v991_data = s1[59];
          float v993_data = ir6[6];
          ir6[6] = (v993_data + (v990_data * v991_data));
          int32_t v1000_a = v12_lead + 160;
          float v1004_data = s0[(v1000_a ^ ((v1000_a >> 5) & 31))];
          float v1005_data = s1[68];
          float v1007_data = ir6[7];
          ir6[7] = (v1007_data + (v1004_data * v1005_data));
          int32_t v1014_a = v12_lead + 160;
          float v1018_data = s0[(v1014_a ^ ((v1014_a >> 5) & 31))];
          float v1019_data = s1[77];
          float v1021_data = ir6[8];
          ir6[8] = (v1021_data + (v1018_data * v1019_data));
          int32_t v1031_a = v12_lead + 192;
          float v1035_data = s0[(v1031_a ^ ((v1031_a >> 5) & 31))];
          float v1036_data = s1[6];
          float v1038_data = ir6[0];
          ir6[0] = (v1038_data + (v1035_data * v1036_data));
          int32_t v1045_a = v12_lead + 192;
          float v1049_data = s0[(v1045_a ^ ((v1045_a >> 5) & 31))];
          float v1050_data = s1[15];
          float v1052_data = ir6[1];
          ir6[1] = (v1052_data + (v1049_data * v1050_data));
          int32_t v1059_a = v12_lead + 192;
          float v1063_data = s0[(v1059_a ^ ((v1059_a >> 5) & 31))];
          float v1064_data = s1[24];
          float v1066_data = ir6[2];
          ir6[2] = (v1066_data + (v1063_data * v1064_data));
          int32_t v1073_a = v12_lead + 192;
          float v1077_data = s0[(v1073_a ^ ((v1073_a >> 5) & 31))];
          float v1078_data = s1[33];
          float v1080_data = ir6[3];
          ir6[3] = (v1080_data + (v1077_data * v1078_data));
          int32_t v1087_a = v12_lead + 192;
          float v1091_data = s0[(v1087_a ^ ((v1087_a >> 5) & 31))];
          float v1092_data = s1[42];
          float v1094_data = ir6[4];
          ir6[4] = (v1094_data + (v1091_data * v1092_data));
          int32_t v1101_a = v12_lead + 192;
          float v1105_data = s0[(v1101_a ^ ((v1101_a >> 5) & 31))];
          float v1106_data = s1[51];
          float v1108_data = ir6[5];
          ir6[5] = (v1108_data + (v1105_data * v1106_data));
          int32_t v1115_a = v12_lead + 192;
          float v1119_data = s0[(v1115_a ^ ((v1115_a >> 5) & 31))];
          float v1120_data = s1[60];
          float v1122_data = ir6[6];
          ir6[6] = (v1122_data + (v1119_data * v1120_data));
          int32_t v1129_a = v12_lead + 192;
          float v1133_data = s0[(v1129_a ^ ((v1129_a >> 5) & 31))];
          float v1134_data = s1[69];
          float v1136_data = ir6[7];
          ir6[7] = (v1136_data + (v1133_data * v1134_data));
          int32_t v1143_a = v12_lead + 192;
          float v1147_data = s0[(v1143_a ^ ((v1143_a >> 5) & 31))];
          float v1148_data = s1[78];
          float v1150_data = ir6[8];
          ir6[8] = (v1150_data + (v1147_data * v1148_data));
          int32_t v1160_a = v12_lead + 224;
          float v1164_data = s0[(v1160_a ^ ((v1160_a >> 5) & 31))];
          float v1165_data = s1[7];
          float v1167_data = ir6[0];
          ir6[0] = (v1167_data + (v1164_data * v1165_data));
          int32_t v1174_a = v12_lead + 224;
          float v1178_data = s0[(v1174_a ^ ((v1174_a >> 5) & 31))];
          float v1179_data = s1[16];
          float v1181_data = ir6[1];
          ir6[1] = (v1181_data + (v1178_data * v1179_data));
          int32_t v1188_a = v12_lead + 224;
          float v1192_data = s0[(v1188_a ^ ((v1188_a >> 5) & 31))];
          float v1193_data = s1[25];
          float v1195_data = ir6[2];
          ir6[2] = (v1195_data + (v1192_data * v1193_data));
          int32_t v1202_a = v12_lead + 224;
          float v1206_data = s0[(v1202_a ^ ((v1202_a >> 5) & 31))];
          float v1207_data = s1[34];
          float v1209_data = ir6[3];
          ir6[3] = (v1209_data + (v1206_data * v1207_data));
          int32_t v1216_a = v12_lead + 224;
          float v1220_data = s0[(v1216_a ^ ((v1216_a >> 5) & 31))];
          float v1221_data = s1[43];
          float v1223_data = ir6[4];
          ir6[4] = (v1223_data + (v1220_data * v1221_data));
          int32_t v1230_a = v12_lead + 224;
          float v1234_data = s0[(v1230_a ^ ((v1230_a >> 5) & 31))];
          float v1235_data = s1[52];
          float v1237_data = ir6[5];
          ir6[5] = (v1237_data + (v1234_data * v1235_data));
          int32_t v1244_a = v12_lead + 224;
          float v1248_data = s0[(v1244_a ^ ((v1244_a >> 5) & 31))];
          float v1249_data = s1[61];
          float v1251_data = ir6[6];
          ir6[6] = (v1251_data + (v1248_data * v1249_data));
          int32_t v1258_a = v12_lead + 224;
          float v1262_data = s0[(v1258_a ^ ((v1258_a >> 5) & 31))];
          float v1263_data = s1[70];
          float v1265_data = ir6[7];
          ir6[7] = (v1265_data + (v1262_data * v1263_data));
          int32_t v1272_a = v12_lead + 224;
          float v1276_data = s0[(v1272_a ^ ((v1272_a >> 5) & 31))];
          float v1277_data = s1[79];
          float v1279_data = ir6[8];
          ir6[8] = (v1279_data + (v1276_data * v1277_data));
          int32_t v1289_a = v12_lead + 256;
          float v1293_data = s0[(v1289_a ^ ((v1289_a >> 5) & 31))];
          float v1294_data = s1[8];
          float v1296_data = ir6[0];
          ir6[0] = (v1296_data + (v1293_data * v1294_data));
          int32_t v1303_a = v12_lead + 256;
          float v1307_data = s0[(v1303_a ^ ((v1303_a >> 5) & 31))];
          float v1308_data = s1[17];
          float v1310_data = ir6[1];
          ir6[1] = (v1310_data + (v1307_data * v1308_data));
          int32_t v1317_a = v12_lead + 256;
          float v1321_data = s0[(v1317_a ^ ((v1317_a >> 5) & 31))];
          float v1322_data = s1[26];
          float v1324_data = ir6[2];
          ir6[2] = (v1324_data + (v1321_data * v1322_data));
          int32_t v1331_a = v12_lead + 256;
          float v1335_data = s0[(v1331_a ^ ((v1331_a >> 5) & 31))];
          float v1336_data = s1[35];
          float v1338_data = ir6[3];
          ir6[3] = (v1338_data + (v1335_data * v1336_data));
          int32_t v1345_a = v12_lead + 256;
          float v1349_data = s0[(v1345_a ^ ((v1345_a >> 5) & 31))];
          float v1350_data = s1[44];
          float v1352_data = ir6[4];
          ir6[4] = (v1352_data + (v1349_data * v1350_data));
          int32_t v1359_a = v12_lead + 256;
          float v1363_data = s0[(v1359_a ^ ((v1359_a >> 5) & 31))];
          float v1364_data = s1[53];
          float v1366_data = ir6[5];
          ir6[5] = (v1366_data + (v1363_data * v1364_data));
          int32_t v1373_a = v12_lead + 256;
          float v1377_data = s0[(v1373_a ^ ((v1373_a >> 5) & 31))];
          float v1378_data = s1[62];
          float v1380_data = ir6[6];
          ir6[6] = (v1380_data + (v1377_data * v1378_data));
          int32_t v1387_a = v12_lead + 256;
          float v1391_data = s0[(v1387_a ^ ((v1387_a >> 5) & 31))];
          float v1392_data = s1[71];
          float v1394_data = ir6[7];
          ir6[7] = (v1394_data + (v1391_data * v1392_data));
          int32_t v1401_a = v12_lead + 256;
          float v1405_data = s0[(v1401_a ^ ((v1401_a >> 5) & 31))];
          float v1406_data = s1[80];
          float v1408_data = ir6[8];
          ir6[8] = (v1408_data + (v1405_data * v1406_data));
          #pragma unroll
          for (int32_t v1413_n0 = 0; v1413_n0 < 1; ++v1413_n0) {
            #pragma unroll
            for (int32_t v1414_n1 = 0; v1414_n1 < 9; ++v1414_n1) {
              int32_t v1415_a = v1413_n0 + v1414_n1;
              float v1416_data = ir6[v1415_a];
              r6[v1415_a] = v1416_data;
            }
          }
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v1421_i0 = 0; v1421_i0 < 1; ++v1421_i0) {
            int32_t v1429_lead = v12_lead + (v1421_i0 * 32);
            #pragma unroll
            for (int32_t v1422_i1 = 0; v1422_i1 < 9; ++v1422_i1) {
              float v1424_data = r6[(v1421_i0 + v1422_i1)];
              glb_m3[(v1429_lead + (v1422_i1 * 32))] = v1424_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

