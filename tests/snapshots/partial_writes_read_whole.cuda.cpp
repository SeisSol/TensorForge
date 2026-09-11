// === base name ===
kernel_dc02485e895da10b

// === header ===
void launcher_kernel_dc02485e895da10b(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, const float** m2, size_t m2_extraOffset, float** m3, size_t m3_extraOffset, const float** m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_dc02485e895da10b(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, const float** m2, size_t m2_extraOffset, float** m3, size_t m3_extraOffset, const float** m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 4, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_dc02485e895da10b, block.x * block.y * block.z, 1536 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_dc02485e895da10b, cudaFuncAttributeMaxDynamicSharedMemorySize, 1536 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_dc02485e895da10b<<<grid,block,1536 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(128)
 kernel_kernel_dc02485e895da10b(const float** m0, size_t m0_extraOffset, const float** m1, size_t m1_extraOffset, const float** m2, size_t m2_extraOffset, float** m3, size_t m3_extraOffset, const float** m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // options: default
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
      const auto batchId_start = (threadIdx.y + blockDim.y * (blockIdx.x));
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[384 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[384];
      float * __restrict__ s0 = &localShrMem0[96];
      float * __restrict__ s1 = &localShrMem0[0];
      for (size_t v5_batchId0 = (threadIdx.y + blockDim.y * (blockIdx.x)); v5_batchId0 < numElements0; v5_batchId0 += (gridDim.x * blockDim.y)) {
        size_t v6_ahead1 = v5_batchId0 + (gridDim.x * blockDim.y);
        size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
        if (allowed) {
          const float *const __restrict__ glb_m0 = &m0[v5_batchId0][0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[v5_batchId0][0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[v5_batchId0][0 + m2_extraOffset];
          float *const __restrict__ glb_m3 = &m3[v5_batchId0][0 + m3_extraOffset];
          const float *const __restrict__ glb_m4 = &m4[v5_batchId0][0 + m4_extraOffset];
          float r0[9]{};
          // r0 = load{g>r}(glb_m0);
          int32_t v21_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v22_i0 = 0; v22_i0 < 1; ++v22_i0) {
            int32_t v28_lead = v21_lead + (v22_i0 * 32);
            #pragma unroll
            for (int32_t v23_i1 = 0; v23_i1 < 9; ++v23_i1) {
              float v31_data = __ldcg(&glb_m0[(v28_lead + (v23_i1 * 32))]);
              r0[(v22_i0 + v23_i1)] = v31_data;
            }
          }
          float r2[9]{};
          // r2 = load{g>r}(glb_m1);
          if (v21_lead < 16) {
            #pragma unroll
            for (int32_t v38_i1 = 0; v38_i1 < 9; ++v38_i1) {
              float v46_data = __ldcg(&glb_m1[(v21_lead + (v38_i1 * 16))]);
              r2[v38_i1] = v46_data;
            }
          }
          // wait(r0 = load{g>r}(glb_m0););
          float r1[9]{};
          // r1 = +(r0) + None
          // [(0, 32), (0, 9)] []
          float v52_data = r0[0];
          float v53_data = r1[0];
          r1[0] = (v53_data + v52_data);
          float v55_data = r0[1];
          float v56_data = r1[1];
          r1[1] = (v56_data + v55_data);
          float v58_data = r0[2];
          float v59_data = r1[2];
          r1[2] = (v59_data + v58_data);
          float v61_data = r0[3];
          float v62_data = r1[3];
          r1[3] = (v62_data + v61_data);
          float v64_data = r0[4];
          float v65_data = r1[4];
          r1[4] = (v65_data + v64_data);
          float v67_data = r0[5];
          float v68_data = r1[5];
          r1[5] = (v68_data + v67_data);
          float v70_data = r0[6];
          float v71_data = r1[6];
          r1[6] = (v71_data + v70_data);
          float v73_data = r0[7];
          float v74_data = r1[7];
          r1[7] = (v74_data + v73_data);
          float v76_data = r0[8];
          float v77_data = r1[8];
          r1[8] = (v77_data + v76_data);
          // s0 = store{r>s}(localShrMem0, r1);
          #pragma unroll
          for (int32_t v82_i0 = 0; v82_i0 < 1; ++v82_i0) {
            int32_t v90_lead = v21_lead + (v82_i0 * 32);
            #pragma unroll
            for (int32_t v83_i1 = 0; v83_i1 < 9; ++v83_i1) {
              float v85_data = r1[(v82_i0 + v83_i1)];
              int32_t v92_a = v90_lead + (v83_i1 * 32);
              s0[(v92_a ^ ((v92_a >> 5) & 31))] = v85_data;
            }
          }
          float r4[9]{};
          // r4 = load{g>r}(glb_m2);
          if (v21_lead < 16) {
            #pragma unroll
            for (int32_t v101_i1 = 0; v101_i1 < 9; ++v101_i1) {
              float v109_data = __ldcg(&glb_m2[(v21_lead + (v101_i1 * 16))]);
              r4[v101_i1] = v109_data;
            }
          }
          // wait(r2 = load{g>r}(glb_m1););
          float r3[9]{};
          __syncwarp();
          // r3 = +(r2) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir3[9]{};
          if (v21_lead < 16) {
            float v117_data = r2[0];
            float v118_data = ir3[0];
            ir3[0] = (v118_data + v117_data);
            float v120_data = r2[1];
            float v121_data = ir3[1];
            ir3[1] = (v121_data + v120_data);
            float v123_data = r2[2];
            float v124_data = ir3[2];
            ir3[2] = (v124_data + v123_data);
            float v126_data = r2[3];
            float v127_data = ir3[3];
            ir3[3] = (v127_data + v126_data);
            float v129_data = r2[4];
            float v130_data = ir3[4];
            ir3[4] = (v130_data + v129_data);
            float v132_data = r2[5];
            float v133_data = ir3[5];
            ir3[5] = (v133_data + v132_data);
            float v135_data = r2[6];
            float v136_data = ir3[6];
            ir3[6] = (v136_data + v135_data);
            float v138_data = r2[7];
            float v139_data = ir3[7];
            ir3[7] = (v139_data + v138_data);
            float v141_data = r2[8];
            float v142_data = ir3[8];
            ir3[8] = (v142_data + v141_data);
          }
          if (v21_lead < 16) {
            #pragma unroll
            for (int32_t v148_n1 = 0; v148_n1 < 9; ++v148_n1) {
              float v150_data = ir3[v148_n1];
              int32_t v157_a = v21_lead + (v148_n1 * 32);
              float v161_data = s0[(v157_a ^ ((v157_a >> 5) & 31))];
              r3[v148_n1] = (v161_data + v150_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r3);
          if (v21_lead < 16) {
            #pragma unroll
            for (int32_t v168_i1 = 0; v168_i1 < 9; ++v168_i1) {
              float v170_data = r3[v168_i1];
              int32_t v177_a = v21_lead + (v168_i1 * 32);
              s0[(v177_a ^ ((v177_a >> 5) & 31))] = v170_data;
            }
          }
          // s1 = load{g>s}(glb_m4[0, 1])
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 0], &glb_m4[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 32], &glb_m4[0 + 0 + 1 * threadIdx.x + 32], 4);
          if (threadIdx.x < 17) {
            __pipeline_memcpy_async(&s1[0 + 0 + 1 * threadIdx.x + 64], &glb_m4[0 + 0 + 1 * threadIdx.x + 64], 4);
          }
          __pipeline_commit();
          // wait(r4 = load{g>r}(glb_m2););
          float r5[9]{};
          __syncwarp();
          // r5 = +(r4) + name: s0, type: SymbolType.SharedMem, lead: [0]
          // [(0, 16), (0, 9)] []
          float ir5[9]{};
          if (v21_lead < 16) {
            float v190_data = r4[0];
            float v191_data = ir5[0];
            ir5[0] = (v191_data + v190_data);
            float v193_data = r4[1];
            float v194_data = ir5[1];
            ir5[1] = (v194_data + v193_data);
            float v196_data = r4[2];
            float v197_data = ir5[2];
            ir5[2] = (v197_data + v196_data);
            float v199_data = r4[3];
            float v200_data = ir5[3];
            ir5[3] = (v200_data + v199_data);
            float v202_data = r4[4];
            float v203_data = ir5[4];
            ir5[4] = (v203_data + v202_data);
            float v205_data = r4[5];
            float v206_data = ir5[5];
            ir5[5] = (v206_data + v205_data);
            float v208_data = r4[6];
            float v209_data = ir5[6];
            ir5[6] = (v209_data + v208_data);
            float v211_data = r4[7];
            float v212_data = ir5[7];
            ir5[7] = (v212_data + v211_data);
            float v214_data = r4[8];
            float v215_data = ir5[8];
            ir5[8] = (v215_data + v214_data);
          }
          if (v21_lead < 16) {
            #pragma unroll
            for (int32_t v221_n1 = 0; v221_n1 < 9; ++v221_n1) {
              float v223_data = ir5[v221_n1];
              int32_t v230_a = v21_lead + (v221_n1 * 32);
              float v234_data = s0[(v230_a ^ ((v230_a >> 5) & 31))];
              r5[v221_n1] = (v234_data + v223_data);
            }
          }
          __syncwarp();
          // s0 = store{r>s}(localShrMem0, r5);
          if (v21_lead < 16) {
            #pragma unroll
            for (int32_t v241_i1 = 0; v241_i1 < 9; ++v241_i1) {
              float v243_data = r5[v241_i1];
              int32_t v250_a = v21_lead + (v241_i1 * 32);
              s0[(v250_a ^ ((v250_a >> 5) & 31))] = v243_data;
            }
          }
          // wait(s1 = load{g>s}(glb_m4[0, 1]));
          __pipeline_wait_prior(0);
          float r6[9]{};
          __syncwarp();
          // r6 = +(s0 * s1) + None
          // [(0, 32), (0, 9)] [(0, 9)]
          float ir6[9]{};
          float v268_data = s0[(v21_lead ^ ((v21_lead >> 5) & 31))];
          float v269_data = s1[0];
          float v271_data = ir6[0];
          ir6[0] = (v271_data + (v268_data * v269_data));
          float v282_data = s0[(v21_lead ^ ((v21_lead >> 5) & 31))];
          float v283_data = s1[9];
          float v285_data = ir6[1];
          ir6[1] = (v285_data + (v282_data * v283_data));
          float v296_data = s0[(v21_lead ^ ((v21_lead >> 5) & 31))];
          float v297_data = s1[18];
          float v299_data = ir6[2];
          ir6[2] = (v299_data + (v296_data * v297_data));
          float v310_data = s0[(v21_lead ^ ((v21_lead >> 5) & 31))];
          float v311_data = s1[27];
          float v313_data = ir6[3];
          ir6[3] = (v313_data + (v310_data * v311_data));
          float v324_data = s0[(v21_lead ^ ((v21_lead >> 5) & 31))];
          float v325_data = s1[36];
          float v327_data = ir6[4];
          ir6[4] = (v327_data + (v324_data * v325_data));
          float v338_data = s0[(v21_lead ^ ((v21_lead >> 5) & 31))];
          float v339_data = s1[45];
          float v341_data = ir6[5];
          ir6[5] = (v341_data + (v338_data * v339_data));
          float v352_data = s0[(v21_lead ^ ((v21_lead >> 5) & 31))];
          float v353_data = s1[54];
          float v355_data = ir6[6];
          ir6[6] = (v355_data + (v352_data * v353_data));
          float v366_data = s0[(v21_lead ^ ((v21_lead >> 5) & 31))];
          float v367_data = s1[63];
          float v369_data = ir6[7];
          ir6[7] = (v369_data + (v366_data * v367_data));
          float v380_data = s0[(v21_lead ^ ((v21_lead >> 5) & 31))];
          float v381_data = s1[72];
          float v383_data = ir6[8];
          ir6[8] = (v383_data + (v380_data * v381_data));
          int32_t v393_a = v21_lead + 32;
          float v397_data = s0[(v393_a ^ ((v393_a >> 5) & 31))];
          float v398_data = s1[1];
          float v400_data = ir6[0];
          ir6[0] = (v400_data + (v397_data * v398_data));
          int32_t v407_a = v21_lead + 32;
          float v411_data = s0[(v407_a ^ ((v407_a >> 5) & 31))];
          float v412_data = s1[10];
          float v414_data = ir6[1];
          ir6[1] = (v414_data + (v411_data * v412_data));
          int32_t v421_a = v21_lead + 32;
          float v425_data = s0[(v421_a ^ ((v421_a >> 5) & 31))];
          float v426_data = s1[19];
          float v428_data = ir6[2];
          ir6[2] = (v428_data + (v425_data * v426_data));
          int32_t v435_a = v21_lead + 32;
          float v439_data = s0[(v435_a ^ ((v435_a >> 5) & 31))];
          float v440_data = s1[28];
          float v442_data = ir6[3];
          ir6[3] = (v442_data + (v439_data * v440_data));
          int32_t v449_a = v21_lead + 32;
          float v453_data = s0[(v449_a ^ ((v449_a >> 5) & 31))];
          float v454_data = s1[37];
          float v456_data = ir6[4];
          ir6[4] = (v456_data + (v453_data * v454_data));
          int32_t v463_a = v21_lead + 32;
          float v467_data = s0[(v463_a ^ ((v463_a >> 5) & 31))];
          float v468_data = s1[46];
          float v470_data = ir6[5];
          ir6[5] = (v470_data + (v467_data * v468_data));
          int32_t v477_a = v21_lead + 32;
          float v481_data = s0[(v477_a ^ ((v477_a >> 5) & 31))];
          float v482_data = s1[55];
          float v484_data = ir6[6];
          ir6[6] = (v484_data + (v481_data * v482_data));
          int32_t v491_a = v21_lead + 32;
          float v495_data = s0[(v491_a ^ ((v491_a >> 5) & 31))];
          float v496_data = s1[64];
          float v498_data = ir6[7];
          ir6[7] = (v498_data + (v495_data * v496_data));
          int32_t v505_a = v21_lead + 32;
          float v509_data = s0[(v505_a ^ ((v505_a >> 5) & 31))];
          float v510_data = s1[73];
          float v512_data = ir6[8];
          ir6[8] = (v512_data + (v509_data * v510_data));
          int32_t v522_a = v21_lead + 64;
          float v526_data = s0[(v522_a ^ ((v522_a >> 5) & 31))];
          float v527_data = s1[2];
          float v529_data = ir6[0];
          ir6[0] = (v529_data + (v526_data * v527_data));
          int32_t v536_a = v21_lead + 64;
          float v540_data = s0[(v536_a ^ ((v536_a >> 5) & 31))];
          float v541_data = s1[11];
          float v543_data = ir6[1];
          ir6[1] = (v543_data + (v540_data * v541_data));
          int32_t v550_a = v21_lead + 64;
          float v554_data = s0[(v550_a ^ ((v550_a >> 5) & 31))];
          float v555_data = s1[20];
          float v557_data = ir6[2];
          ir6[2] = (v557_data + (v554_data * v555_data));
          int32_t v564_a = v21_lead + 64;
          float v568_data = s0[(v564_a ^ ((v564_a >> 5) & 31))];
          float v569_data = s1[29];
          float v571_data = ir6[3];
          ir6[3] = (v571_data + (v568_data * v569_data));
          int32_t v578_a = v21_lead + 64;
          float v582_data = s0[(v578_a ^ ((v578_a >> 5) & 31))];
          float v583_data = s1[38];
          float v585_data = ir6[4];
          ir6[4] = (v585_data + (v582_data * v583_data));
          int32_t v592_a = v21_lead + 64;
          float v596_data = s0[(v592_a ^ ((v592_a >> 5) & 31))];
          float v597_data = s1[47];
          float v599_data = ir6[5];
          ir6[5] = (v599_data + (v596_data * v597_data));
          int32_t v606_a = v21_lead + 64;
          float v610_data = s0[(v606_a ^ ((v606_a >> 5) & 31))];
          float v611_data = s1[56];
          float v613_data = ir6[6];
          ir6[6] = (v613_data + (v610_data * v611_data));
          int32_t v620_a = v21_lead + 64;
          float v624_data = s0[(v620_a ^ ((v620_a >> 5) & 31))];
          float v625_data = s1[65];
          float v627_data = ir6[7];
          ir6[7] = (v627_data + (v624_data * v625_data));
          int32_t v634_a = v21_lead + 64;
          float v638_data = s0[(v634_a ^ ((v634_a >> 5) & 31))];
          float v639_data = s1[74];
          float v641_data = ir6[8];
          ir6[8] = (v641_data + (v638_data * v639_data));
          int32_t v651_a = v21_lead + 96;
          float v655_data = s0[(v651_a ^ ((v651_a >> 5) & 31))];
          float v656_data = s1[3];
          float v658_data = ir6[0];
          ir6[0] = (v658_data + (v655_data * v656_data));
          int32_t v665_a = v21_lead + 96;
          float v669_data = s0[(v665_a ^ ((v665_a >> 5) & 31))];
          float v670_data = s1[12];
          float v672_data = ir6[1];
          ir6[1] = (v672_data + (v669_data * v670_data));
          int32_t v679_a = v21_lead + 96;
          float v683_data = s0[(v679_a ^ ((v679_a >> 5) & 31))];
          float v684_data = s1[21];
          float v686_data = ir6[2];
          ir6[2] = (v686_data + (v683_data * v684_data));
          int32_t v693_a = v21_lead + 96;
          float v697_data = s0[(v693_a ^ ((v693_a >> 5) & 31))];
          float v698_data = s1[30];
          float v700_data = ir6[3];
          ir6[3] = (v700_data + (v697_data * v698_data));
          int32_t v707_a = v21_lead + 96;
          float v711_data = s0[(v707_a ^ ((v707_a >> 5) & 31))];
          float v712_data = s1[39];
          float v714_data = ir6[4];
          ir6[4] = (v714_data + (v711_data * v712_data));
          int32_t v721_a = v21_lead + 96;
          float v725_data = s0[(v721_a ^ ((v721_a >> 5) & 31))];
          float v726_data = s1[48];
          float v728_data = ir6[5];
          ir6[5] = (v728_data + (v725_data * v726_data));
          int32_t v735_a = v21_lead + 96;
          float v739_data = s0[(v735_a ^ ((v735_a >> 5) & 31))];
          float v740_data = s1[57];
          float v742_data = ir6[6];
          ir6[6] = (v742_data + (v739_data * v740_data));
          int32_t v749_a = v21_lead + 96;
          float v753_data = s0[(v749_a ^ ((v749_a >> 5) & 31))];
          float v754_data = s1[66];
          float v756_data = ir6[7];
          ir6[7] = (v756_data + (v753_data * v754_data));
          int32_t v763_a = v21_lead + 96;
          float v767_data = s0[(v763_a ^ ((v763_a >> 5) & 31))];
          float v768_data = s1[75];
          float v770_data = ir6[8];
          ir6[8] = (v770_data + (v767_data * v768_data));
          int32_t v780_a = v21_lead + 128;
          float v784_data = s0[(v780_a ^ ((v780_a >> 5) & 31))];
          float v785_data = s1[4];
          float v787_data = ir6[0];
          ir6[0] = (v787_data + (v784_data * v785_data));
          int32_t v794_a = v21_lead + 128;
          float v798_data = s0[(v794_a ^ ((v794_a >> 5) & 31))];
          float v799_data = s1[13];
          float v801_data = ir6[1];
          ir6[1] = (v801_data + (v798_data * v799_data));
          int32_t v808_a = v21_lead + 128;
          float v812_data = s0[(v808_a ^ ((v808_a >> 5) & 31))];
          float v813_data = s1[22];
          float v815_data = ir6[2];
          ir6[2] = (v815_data + (v812_data * v813_data));
          int32_t v822_a = v21_lead + 128;
          float v826_data = s0[(v822_a ^ ((v822_a >> 5) & 31))];
          float v827_data = s1[31];
          float v829_data = ir6[3];
          ir6[3] = (v829_data + (v826_data * v827_data));
          int32_t v836_a = v21_lead + 128;
          float v840_data = s0[(v836_a ^ ((v836_a >> 5) & 31))];
          float v841_data = s1[40];
          float v843_data = ir6[4];
          ir6[4] = (v843_data + (v840_data * v841_data));
          int32_t v850_a = v21_lead + 128;
          float v854_data = s0[(v850_a ^ ((v850_a >> 5) & 31))];
          float v855_data = s1[49];
          float v857_data = ir6[5];
          ir6[5] = (v857_data + (v854_data * v855_data));
          int32_t v864_a = v21_lead + 128;
          float v868_data = s0[(v864_a ^ ((v864_a >> 5) & 31))];
          float v869_data = s1[58];
          float v871_data = ir6[6];
          ir6[6] = (v871_data + (v868_data * v869_data));
          int32_t v878_a = v21_lead + 128;
          float v882_data = s0[(v878_a ^ ((v878_a >> 5) & 31))];
          float v883_data = s1[67];
          float v885_data = ir6[7];
          ir6[7] = (v885_data + (v882_data * v883_data));
          int32_t v892_a = v21_lead + 128;
          float v896_data = s0[(v892_a ^ ((v892_a >> 5) & 31))];
          float v897_data = s1[76];
          float v899_data = ir6[8];
          ir6[8] = (v899_data + (v896_data * v897_data));
          int32_t v909_a = v21_lead + 160;
          float v913_data = s0[(v909_a ^ ((v909_a >> 5) & 31))];
          float v914_data = s1[5];
          float v916_data = ir6[0];
          ir6[0] = (v916_data + (v913_data * v914_data));
          int32_t v923_a = v21_lead + 160;
          float v927_data = s0[(v923_a ^ ((v923_a >> 5) & 31))];
          float v928_data = s1[14];
          float v930_data = ir6[1];
          ir6[1] = (v930_data + (v927_data * v928_data));
          int32_t v937_a = v21_lead + 160;
          float v941_data = s0[(v937_a ^ ((v937_a >> 5) & 31))];
          float v942_data = s1[23];
          float v944_data = ir6[2];
          ir6[2] = (v944_data + (v941_data * v942_data));
          int32_t v951_a = v21_lead + 160;
          float v955_data = s0[(v951_a ^ ((v951_a >> 5) & 31))];
          float v956_data = s1[32];
          float v958_data = ir6[3];
          ir6[3] = (v958_data + (v955_data * v956_data));
          int32_t v965_a = v21_lead + 160;
          float v969_data = s0[(v965_a ^ ((v965_a >> 5) & 31))];
          float v970_data = s1[41];
          float v972_data = ir6[4];
          ir6[4] = (v972_data + (v969_data * v970_data));
          int32_t v979_a = v21_lead + 160;
          float v983_data = s0[(v979_a ^ ((v979_a >> 5) & 31))];
          float v984_data = s1[50];
          float v986_data = ir6[5];
          ir6[5] = (v986_data + (v983_data * v984_data));
          int32_t v993_a = v21_lead + 160;
          float v997_data = s0[(v993_a ^ ((v993_a >> 5) & 31))];
          float v998_data = s1[59];
          float v1000_data = ir6[6];
          ir6[6] = (v1000_data + (v997_data * v998_data));
          int32_t v1007_a = v21_lead + 160;
          float v1011_data = s0[(v1007_a ^ ((v1007_a >> 5) & 31))];
          float v1012_data = s1[68];
          float v1014_data = ir6[7];
          ir6[7] = (v1014_data + (v1011_data * v1012_data));
          int32_t v1021_a = v21_lead + 160;
          float v1025_data = s0[(v1021_a ^ ((v1021_a >> 5) & 31))];
          float v1026_data = s1[77];
          float v1028_data = ir6[8];
          ir6[8] = (v1028_data + (v1025_data * v1026_data));
          int32_t v1038_a = v21_lead + 192;
          float v1042_data = s0[(v1038_a ^ ((v1038_a >> 5) & 31))];
          float v1043_data = s1[6];
          float v1045_data = ir6[0];
          ir6[0] = (v1045_data + (v1042_data * v1043_data));
          int32_t v1052_a = v21_lead + 192;
          float v1056_data = s0[(v1052_a ^ ((v1052_a >> 5) & 31))];
          float v1057_data = s1[15];
          float v1059_data = ir6[1];
          ir6[1] = (v1059_data + (v1056_data * v1057_data));
          int32_t v1066_a = v21_lead + 192;
          float v1070_data = s0[(v1066_a ^ ((v1066_a >> 5) & 31))];
          float v1071_data = s1[24];
          float v1073_data = ir6[2];
          ir6[2] = (v1073_data + (v1070_data * v1071_data));
          int32_t v1080_a = v21_lead + 192;
          float v1084_data = s0[(v1080_a ^ ((v1080_a >> 5) & 31))];
          float v1085_data = s1[33];
          float v1087_data = ir6[3];
          ir6[3] = (v1087_data + (v1084_data * v1085_data));
          int32_t v1094_a = v21_lead + 192;
          float v1098_data = s0[(v1094_a ^ ((v1094_a >> 5) & 31))];
          float v1099_data = s1[42];
          float v1101_data = ir6[4];
          ir6[4] = (v1101_data + (v1098_data * v1099_data));
          int32_t v1108_a = v21_lead + 192;
          float v1112_data = s0[(v1108_a ^ ((v1108_a >> 5) & 31))];
          float v1113_data = s1[51];
          float v1115_data = ir6[5];
          ir6[5] = (v1115_data + (v1112_data * v1113_data));
          int32_t v1122_a = v21_lead + 192;
          float v1126_data = s0[(v1122_a ^ ((v1122_a >> 5) & 31))];
          float v1127_data = s1[60];
          float v1129_data = ir6[6];
          ir6[6] = (v1129_data + (v1126_data * v1127_data));
          int32_t v1136_a = v21_lead + 192;
          float v1140_data = s0[(v1136_a ^ ((v1136_a >> 5) & 31))];
          float v1141_data = s1[69];
          float v1143_data = ir6[7];
          ir6[7] = (v1143_data + (v1140_data * v1141_data));
          int32_t v1150_a = v21_lead + 192;
          float v1154_data = s0[(v1150_a ^ ((v1150_a >> 5) & 31))];
          float v1155_data = s1[78];
          float v1157_data = ir6[8];
          ir6[8] = (v1157_data + (v1154_data * v1155_data));
          int32_t v1167_a = v21_lead + 224;
          float v1171_data = s0[(v1167_a ^ ((v1167_a >> 5) & 31))];
          float v1172_data = s1[7];
          float v1174_data = ir6[0];
          ir6[0] = (v1174_data + (v1171_data * v1172_data));
          int32_t v1181_a = v21_lead + 224;
          float v1185_data = s0[(v1181_a ^ ((v1181_a >> 5) & 31))];
          float v1186_data = s1[16];
          float v1188_data = ir6[1];
          ir6[1] = (v1188_data + (v1185_data * v1186_data));
          int32_t v1195_a = v21_lead + 224;
          float v1199_data = s0[(v1195_a ^ ((v1195_a >> 5) & 31))];
          float v1200_data = s1[25];
          float v1202_data = ir6[2];
          ir6[2] = (v1202_data + (v1199_data * v1200_data));
          int32_t v1209_a = v21_lead + 224;
          float v1213_data = s0[(v1209_a ^ ((v1209_a >> 5) & 31))];
          float v1214_data = s1[34];
          float v1216_data = ir6[3];
          ir6[3] = (v1216_data + (v1213_data * v1214_data));
          int32_t v1223_a = v21_lead + 224;
          float v1227_data = s0[(v1223_a ^ ((v1223_a >> 5) & 31))];
          float v1228_data = s1[43];
          float v1230_data = ir6[4];
          ir6[4] = (v1230_data + (v1227_data * v1228_data));
          int32_t v1237_a = v21_lead + 224;
          float v1241_data = s0[(v1237_a ^ ((v1237_a >> 5) & 31))];
          float v1242_data = s1[52];
          float v1244_data = ir6[5];
          ir6[5] = (v1244_data + (v1241_data * v1242_data));
          int32_t v1251_a = v21_lead + 224;
          float v1255_data = s0[(v1251_a ^ ((v1251_a >> 5) & 31))];
          float v1256_data = s1[61];
          float v1258_data = ir6[6];
          ir6[6] = (v1258_data + (v1255_data * v1256_data));
          int32_t v1265_a = v21_lead + 224;
          float v1269_data = s0[(v1265_a ^ ((v1265_a >> 5) & 31))];
          float v1270_data = s1[70];
          float v1272_data = ir6[7];
          ir6[7] = (v1272_data + (v1269_data * v1270_data));
          int32_t v1279_a = v21_lead + 224;
          float v1283_data = s0[(v1279_a ^ ((v1279_a >> 5) & 31))];
          float v1284_data = s1[79];
          float v1286_data = ir6[8];
          ir6[8] = (v1286_data + (v1283_data * v1284_data));
          int32_t v1296_a = v21_lead + 256;
          float v1300_data = s0[(v1296_a ^ ((v1296_a >> 5) & 31))];
          float v1301_data = s1[8];
          float v1303_data = ir6[0];
          ir6[0] = (v1303_data + (v1300_data * v1301_data));
          int32_t v1310_a = v21_lead + 256;
          float v1314_data = s0[(v1310_a ^ ((v1310_a >> 5) & 31))];
          float v1315_data = s1[17];
          float v1317_data = ir6[1];
          ir6[1] = (v1317_data + (v1314_data * v1315_data));
          int32_t v1324_a = v21_lead + 256;
          float v1328_data = s0[(v1324_a ^ ((v1324_a >> 5) & 31))];
          float v1329_data = s1[26];
          float v1331_data = ir6[2];
          ir6[2] = (v1331_data + (v1328_data * v1329_data));
          int32_t v1338_a = v21_lead + 256;
          float v1342_data = s0[(v1338_a ^ ((v1338_a >> 5) & 31))];
          float v1343_data = s1[35];
          float v1345_data = ir6[3];
          ir6[3] = (v1345_data + (v1342_data * v1343_data));
          int32_t v1352_a = v21_lead + 256;
          float v1356_data = s0[(v1352_a ^ ((v1352_a >> 5) & 31))];
          float v1357_data = s1[44];
          float v1359_data = ir6[4];
          ir6[4] = (v1359_data + (v1356_data * v1357_data));
          int32_t v1366_a = v21_lead + 256;
          float v1370_data = s0[(v1366_a ^ ((v1366_a >> 5) & 31))];
          float v1371_data = s1[53];
          float v1373_data = ir6[5];
          ir6[5] = (v1373_data + (v1370_data * v1371_data));
          int32_t v1380_a = v21_lead + 256;
          float v1384_data = s0[(v1380_a ^ ((v1380_a >> 5) & 31))];
          float v1385_data = s1[62];
          float v1387_data = ir6[6];
          ir6[6] = (v1387_data + (v1384_data * v1385_data));
          int32_t v1394_a = v21_lead + 256;
          float v1398_data = s0[(v1394_a ^ ((v1394_a >> 5) & 31))];
          float v1399_data = s1[71];
          float v1401_data = ir6[7];
          ir6[7] = (v1401_data + (v1398_data * v1399_data));
          int32_t v1408_a = v21_lead + 256;
          float v1412_data = s0[(v1408_a ^ ((v1408_a >> 5) & 31))];
          float v1413_data = s1[80];
          float v1415_data = ir6[8];
          ir6[8] = (v1415_data + (v1412_data * v1413_data));
          #pragma unroll
          for (int32_t v1420_n0 = 0; v1420_n0 < 1; ++v1420_n0) {
            #pragma unroll
            for (int32_t v1421_n1 = 0; v1421_n1 < 9; ++v1421_n1) {
              int32_t v1422_a = v1420_n0 + v1421_n1;
              float v1423_data = ir6[v1422_a];
              r6[v1422_a] = v1423_data;
            }
          }
          // glb_m3 = store{r>g}(r6);
          #pragma unroll
          for (int32_t v1428_i0 = 0; v1428_i0 < 1; ++v1428_i0) {
            int32_t v1436_lead = v21_lead + (v1428_i0 * 32);
            #pragma unroll
            for (int32_t v1429_i1 = 0; v1429_i1 < 9; ++v1429_i1) {
              float v1431_data = r6[(v1428_i0 + v1429_i1)];
              glb_m3[(v1436_lead + (v1429_i1 * 32))] = v1431_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

