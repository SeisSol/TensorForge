// === base name ===
kernel_69f2bb9311

// === header ===
void launcher_kernel_69f2bb9311(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_69f2bb9311(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (32, 8, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_69f2bb9311, block.x * block.y * block.z, 256 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_69f2bb9311, cudaFuncAttributeMaxDynamicSharedMemorySize, 256 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_69f2bb9311<<<grid,block,256 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_69f2bb9311(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 35×4(35×4) {0..35}×{0..4} strided
    // m1 35×8(35×8) {0..35}×{0..8} strided
    // m2 8×4(8×4) {0..8}×{0..4} strided
    // m0 35×4(35×4) {0..35}×{0..4} strided({0..35}×{0..4})[0, 1] = m1 35×8(35×8) {0..35}×{0..8} strided({0..35}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[32 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[32];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 140 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 280 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 32 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 32;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v16_lead = v11_i0 * 32;
            int32_t v17_lead = v10_lead + v16_lead;
            int32_t v24_lead = v10_lead + v16_lead;
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 8; ++v12_i1) {
              int32_t v18_a = v12_i1 * 35;
              int32_t v19_a = v17_lead + v18_a;
              float v27_data = __ldcg(&glb_m1[(v24_lead + v18_a)]);
              r0[(v11_i0 + (v12_i1 * 2))] = v27_data;
            }
          }
          if (v10_lead < 3) {
            int32_t v36_lead = v10_lead + 32_i32;
            int32_t v43_lead = v10_lead + 32_i32;
            #pragma unroll
            for (int32_t v31_i1 = 0; v31_i1 < 8; ++v31_i1) {
              int32_t v37_a = v31_i1 * 35;
              int32_t v38_a = v36_lead + v37_a;
              float v46_data = __ldcg(&glb_m1[(v43_lead + v37_a)]);
              r0[(1 + (v31_i1 * 2))] = v46_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[8]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 35), (0, 4)] [(0, 8)]
          float ir1[8]{};
          float v56_data = r0[0];
          float v57_data = s0[0];
          float v59_data = ir1[0];
          ir1[0] = (v59_data + (v56_data * v57_data));
          float v62_data = s0[8];
          float v64_data = ir1[2];
          ir1[2] = (v64_data + (v56_data * v62_data));
          float v67_data = s0[16];
          float v69_data = ir1[4];
          ir1[4] = (v69_data + (v56_data * v67_data));
          float v72_data = s0[24];
          float v74_data = ir1[6];
          ir1[6] = (v74_data + (v56_data * v72_data));
          if (v10_lead < 3) {
            float v77_data = r0[1];
            float v80_data = ir1[1];
            ir1[1] = (v80_data + (v77_data * v57_data));
            float v85_data = ir1[3];
            ir1[3] = (v85_data + (v77_data * v62_data));
            float v90_data = ir1[5];
            ir1[5] = (v90_data + (v77_data * v67_data));
            float v95_data = ir1[7];
            ir1[7] = (v95_data + (v77_data * v72_data));
          }
          float v100_data = r0[2];
          float v101_data = s0[1];
          float v103_data = ir1[0];
          ir1[0] = (v103_data + (v100_data * v101_data));
          float v106_data = s0[9];
          float v108_data = ir1[2];
          ir1[2] = (v108_data + (v100_data * v106_data));
          float v111_data = s0[17];
          float v113_data = ir1[4];
          ir1[4] = (v113_data + (v100_data * v111_data));
          float v116_data = s0[25];
          float v118_data = ir1[6];
          ir1[6] = (v118_data + (v100_data * v116_data));
          if (v10_lead < 3) {
            float v121_data = r0[3];
            float v124_data = ir1[1];
            ir1[1] = (v124_data + (v121_data * v101_data));
            float v129_data = ir1[3];
            ir1[3] = (v129_data + (v121_data * v106_data));
            float v134_data = ir1[5];
            ir1[5] = (v134_data + (v121_data * v111_data));
            float v139_data = ir1[7];
            ir1[7] = (v139_data + (v121_data * v116_data));
          }
          float v144_data = r0[4];
          float v145_data = s0[2];
          float v147_data = ir1[0];
          ir1[0] = (v147_data + (v144_data * v145_data));
          float v150_data = s0[10];
          float v152_data = ir1[2];
          ir1[2] = (v152_data + (v144_data * v150_data));
          float v155_data = s0[18];
          float v157_data = ir1[4];
          ir1[4] = (v157_data + (v144_data * v155_data));
          float v160_data = s0[26];
          float v162_data = ir1[6];
          ir1[6] = (v162_data + (v144_data * v160_data));
          if (v10_lead < 3) {
            float v165_data = r0[5];
            float v168_data = ir1[1];
            ir1[1] = (v168_data + (v165_data * v145_data));
            float v173_data = ir1[3];
            ir1[3] = (v173_data + (v165_data * v150_data));
            float v178_data = ir1[5];
            ir1[5] = (v178_data + (v165_data * v155_data));
            float v183_data = ir1[7];
            ir1[7] = (v183_data + (v165_data * v160_data));
          }
          float v188_data = r0[6];
          float v189_data = s0[3];
          float v191_data = ir1[0];
          ir1[0] = (v191_data + (v188_data * v189_data));
          float v194_data = s0[11];
          float v196_data = ir1[2];
          ir1[2] = (v196_data + (v188_data * v194_data));
          float v199_data = s0[19];
          float v201_data = ir1[4];
          ir1[4] = (v201_data + (v188_data * v199_data));
          float v204_data = s0[27];
          float v206_data = ir1[6];
          ir1[6] = (v206_data + (v188_data * v204_data));
          if (v10_lead < 3) {
            float v209_data = r0[7];
            float v212_data = ir1[1];
            ir1[1] = (v212_data + (v209_data * v189_data));
            float v217_data = ir1[3];
            ir1[3] = (v217_data + (v209_data * v194_data));
            float v222_data = ir1[5];
            ir1[5] = (v222_data + (v209_data * v199_data));
            float v227_data = ir1[7];
            ir1[7] = (v227_data + (v209_data * v204_data));
          }
          float v232_data = r0[8];
          float v233_data = s0[4];
          float v235_data = ir1[0];
          ir1[0] = (v235_data + (v232_data * v233_data));
          float v238_data = s0[12];
          float v240_data = ir1[2];
          ir1[2] = (v240_data + (v232_data * v238_data));
          float v243_data = s0[20];
          float v245_data = ir1[4];
          ir1[4] = (v245_data + (v232_data * v243_data));
          float v248_data = s0[28];
          float v250_data = ir1[6];
          ir1[6] = (v250_data + (v232_data * v248_data));
          if (v10_lead < 3) {
            float v253_data = r0[9];
            float v256_data = ir1[1];
            ir1[1] = (v256_data + (v253_data * v233_data));
            float v261_data = ir1[3];
            ir1[3] = (v261_data + (v253_data * v238_data));
            float v266_data = ir1[5];
            ir1[5] = (v266_data + (v253_data * v243_data));
            float v271_data = ir1[7];
            ir1[7] = (v271_data + (v253_data * v248_data));
          }
          float v276_data = r0[10];
          float v277_data = s0[5];
          float v279_data = ir1[0];
          ir1[0] = (v279_data + (v276_data * v277_data));
          float v282_data = s0[13];
          float v284_data = ir1[2];
          ir1[2] = (v284_data + (v276_data * v282_data));
          float v287_data = s0[21];
          float v289_data = ir1[4];
          ir1[4] = (v289_data + (v276_data * v287_data));
          float v292_data = s0[29];
          float v294_data = ir1[6];
          ir1[6] = (v294_data + (v276_data * v292_data));
          if (v10_lead < 3) {
            float v297_data = r0[11];
            float v300_data = ir1[1];
            ir1[1] = (v300_data + (v297_data * v277_data));
            float v305_data = ir1[3];
            ir1[3] = (v305_data + (v297_data * v282_data));
            float v310_data = ir1[5];
            ir1[5] = (v310_data + (v297_data * v287_data));
            float v315_data = ir1[7];
            ir1[7] = (v315_data + (v297_data * v292_data));
          }
          float v320_data = r0[12];
          float v321_data = s0[6];
          float v323_data = ir1[0];
          ir1[0] = (v323_data + (v320_data * v321_data));
          float v326_data = s0[14];
          float v328_data = ir1[2];
          ir1[2] = (v328_data + (v320_data * v326_data));
          float v331_data = s0[22];
          float v333_data = ir1[4];
          ir1[4] = (v333_data + (v320_data * v331_data));
          float v336_data = s0[30];
          float v338_data = ir1[6];
          ir1[6] = (v338_data + (v320_data * v336_data));
          if (v10_lead < 3) {
            float v341_data = r0[13];
            float v344_data = ir1[1];
            ir1[1] = (v344_data + (v341_data * v321_data));
            float v349_data = ir1[3];
            ir1[3] = (v349_data + (v341_data * v326_data));
            float v354_data = ir1[5];
            ir1[5] = (v354_data + (v341_data * v331_data));
            float v359_data = ir1[7];
            ir1[7] = (v359_data + (v341_data * v336_data));
          }
          float v364_data = r0[14];
          float v365_data = s0[7];
          float v367_data = ir1[0];
          ir1[0] = (v367_data + (v364_data * v365_data));
          float v370_data = s0[15];
          float v372_data = ir1[2];
          ir1[2] = (v372_data + (v364_data * v370_data));
          float v375_data = s0[23];
          float v377_data = ir1[4];
          ir1[4] = (v377_data + (v364_data * v375_data));
          float v380_data = s0[31];
          float v382_data = ir1[6];
          ir1[6] = (v382_data + (v364_data * v380_data));
          if (v10_lead < 3) {
            float v385_data = r0[15];
            float v388_data = ir1[1];
            ir1[1] = (v388_data + (v385_data * v365_data));
            float v393_data = ir1[3];
            ir1[3] = (v393_data + (v385_data * v370_data));
            float v398_data = ir1[5];
            ir1[5] = (v398_data + (v385_data * v375_data));
            float v403_data = ir1[7];
            ir1[7] = (v403_data + (v385_data * v380_data));
          }
          #pragma unroll
          for (int32_t v408_n0 = 0; v408_n0 < 1; ++v408_n0) {
            #pragma unroll
            for (int32_t v409_n1 = 0; v409_n1 < 4; ++v409_n1) {
              int32_t v410_a = v409_n1 * 2;
              int32_t v411_a = v408_n0 + v410_a;
              int32_t v413_a = v408_n0 + v410_a;
              float v414_data = ir1[v413_a];
              r1[v413_a] = v414_data;
            }
          }
          if (v10_lead < 3) {
            #pragma unroll
            for (int32_t v418_n1 = 0; v418_n1 < 4; ++v418_n1) {
              int32_t v419_a = v418_n1 * 2;
              int32_t v420_a = 1 + v419_a;
              int32_t v422_a = 1 + v419_a;
              float v423_data = ir1[v422_a];
              r1[v422_a] = v423_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v429_i0 = 0; v429_i0 < 1; ++v429_i0) {
            int32_t v440_lead = v10_lead + (v429_i0 * 32);
            #pragma unroll
            for (int32_t v430_i1 = 0; v430_i1 < 4; ++v430_i1) {
              int32_t v431_a = v430_i1 * 2;
              int32_t v432_a = v429_i0 + v431_a;
              float v435_data = r1[(v429_i0 + v431_a)];
              glb_m0[(v440_lead + (v430_i1 * 35))] = v435_data;
            }
          }
          if (v10_lead < 3) {
            int32_t v454_lead = v10_lead + 32_i32;
            #pragma unroll
            for (int32_t v444_i1 = 0; v444_i1 < 4; ++v444_i1) {
              int32_t v445_a = v444_i1 * 2;
              int32_t v446_a = 1 + v445_a;
              float v449_data = r1[(1 + v445_a)];
              glb_m0[(v454_lead + (v444_i1 * 35))] = v449_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

