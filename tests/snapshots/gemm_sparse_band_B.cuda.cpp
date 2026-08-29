// === base name ===
kernel_30948bd44e

// === header ===
void launcher_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_30948bd44e, block.x * block.y * block.z, 1280 * sizeof(float));
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
        cudaFuncSetAttribute(kernel_kernel_30948bd44e, cudaFuncAttributeMaxDynamicSharedMemorySize, 1280 * sizeof(float));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_30948bd44e<<<grid,block,1280 * sizeof(float),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_30948bd44e(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  extern __shared__ char totalShrMemPtr[];
   {
    // generated with TensorForge. Version: 0.0.1
    // meta data:
    // m0 16×16(16×16) {0..16}×{0..16} strided
    // m1 16×16(16×16) {0..16}×{0..16} strided
    // m2 16×16(16×16) {0..16}×{0..16} strided
    // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
    {
      cuda::pipeline<cuda::thread_scope_thread> pipeline = cuda::make_pipeline();
      const auto batchId_start = threadIdx.y + blockDim.y * (blockIdx.x);
      const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
      const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
      auto* totalShrMem = reinterpret_cast<float*>(totalShrMemPtr);
      float* localShrMem0 = &totalShrMem[80 * threadIdx.y + 0];
      float* tempShrMem = &localShrMem0[64];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          float *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const float *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          float r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v17_lead = v10_lead + (v11_i0 * 16);
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 16; ++v12_i1) {
              float v20_data = __ldcg(&glb_m1[(v17_lead + (v12_i1 * 16))]);
              r0[(v11_i0 + v12_i1)] = v20_data;
            }
          }
          float* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 4);
          __pipeline_commit();
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m2[0 + 0 + 1 * threadIdx.x + 16], 4);
          __pipeline_commit();
          if (threadIdx.x < 14) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], 4);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          float r1[16]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          float ir1[16]{};
          float v31_data = r0[0];
          float v35_data = s0[(0 ^ ((0 >> 4) & 15))];
          float v37_data = ir1[0];
          ir1[0] = (v37_data + (v31_data * v35_data));
          float v43_data = s0[(2 ^ ((2 >> 4) & 15))];
          float v45_data = ir1[1];
          ir1[1] = (v45_data + (v31_data * v43_data));
          float v64_data = r0[1];
          float v68_data = s0[(1 ^ ((1 >> 4) & 15))];
          float v70_data = ir1[0];
          ir1[0] = (v70_data + (v64_data * v68_data));
          float v76_data = s0[(3 ^ ((3 >> 4) & 15))];
          float v78_data = ir1[1];
          ir1[1] = (v78_data + (v64_data * v76_data));
          float v84_data = s0[(5 ^ ((5 >> 4) & 15))];
          float v86_data = ir1[2];
          ir1[2] = (v86_data + (v64_data * v84_data));
          float v104_data = r0[2];
          float v109_data = s0[(4 ^ ((4 >> 4) & 15))];
          float v111_data = ir1[1];
          ir1[1] = (v111_data + (v104_data * v109_data));
          float v117_data = s0[(6 ^ ((6 >> 4) & 15))];
          float v119_data = ir1[2];
          ir1[2] = (v119_data + (v104_data * v117_data));
          float v125_data = s0[(8 ^ ((8 >> 4) & 15))];
          float v127_data = ir1[3];
          ir1[3] = (v127_data + (v104_data * v125_data));
          float v144_data = r0[3];
          float v150_data = s0[(7 ^ ((7 >> 4) & 15))];
          float v152_data = ir1[2];
          ir1[2] = (v152_data + (v144_data * v150_data));
          float v158_data = s0[(9 ^ ((9 >> 4) & 15))];
          float v160_data = ir1[3];
          ir1[3] = (v160_data + (v144_data * v158_data));
          float v166_data = s0[(11 ^ ((11 >> 4) & 15))];
          float v168_data = ir1[4];
          ir1[4] = (v168_data + (v144_data * v166_data));
          float v184_data = r0[4];
          float v191_data = s0[(10 ^ ((10 >> 4) & 15))];
          float v193_data = ir1[3];
          ir1[3] = (v193_data + (v184_data * v191_data));
          float v199_data = s0[(12 ^ ((12 >> 4) & 15))];
          float v201_data = ir1[4];
          ir1[4] = (v201_data + (v184_data * v199_data));
          float v207_data = s0[(14 ^ ((14 >> 4) & 15))];
          float v209_data = ir1[5];
          ir1[5] = (v209_data + (v184_data * v207_data));
          float v224_data = r0[5];
          float v232_data = s0[(13 ^ ((13 >> 4) & 15))];
          float v234_data = ir1[4];
          ir1[4] = (v234_data + (v224_data * v232_data));
          float v240_data = s0[(15 ^ ((15 >> 4) & 15))];
          float v242_data = ir1[5];
          ir1[5] = (v242_data + (v224_data * v240_data));
          float v248_data = s0[(17 ^ ((17 >> 4) & 15))];
          float v250_data = ir1[6];
          ir1[6] = (v250_data + (v224_data * v248_data));
          float v264_data = r0[6];
          float v273_data = s0[(16 ^ ((16 >> 4) & 15))];
          float v275_data = ir1[5];
          ir1[5] = (v275_data + (v264_data * v273_data));
          float v281_data = s0[(18 ^ ((18 >> 4) & 15))];
          float v283_data = ir1[6];
          ir1[6] = (v283_data + (v264_data * v281_data));
          float v289_data = s0[(20 ^ ((20 >> 4) & 15))];
          float v291_data = ir1[7];
          ir1[7] = (v291_data + (v264_data * v289_data));
          float v304_data = r0[7];
          float v314_data = s0[(19 ^ ((19 >> 4) & 15))];
          float v316_data = ir1[6];
          ir1[6] = (v316_data + (v304_data * v314_data));
          float v322_data = s0[(21 ^ ((21 >> 4) & 15))];
          float v324_data = ir1[7];
          ir1[7] = (v324_data + (v304_data * v322_data));
          float v330_data = s0[(23 ^ ((23 >> 4) & 15))];
          float v332_data = ir1[8];
          ir1[8] = (v332_data + (v304_data * v330_data));
          float v344_data = r0[8];
          float v355_data = s0[(22 ^ ((22 >> 4) & 15))];
          float v357_data = ir1[7];
          ir1[7] = (v357_data + (v344_data * v355_data));
          float v363_data = s0[(24 ^ ((24 >> 4) & 15))];
          float v365_data = ir1[8];
          ir1[8] = (v365_data + (v344_data * v363_data));
          float v371_data = s0[(26 ^ ((26 >> 4) & 15))];
          float v373_data = ir1[9];
          ir1[9] = (v373_data + (v344_data * v371_data));
          float v384_data = r0[9];
          float v396_data = s0[(25 ^ ((25 >> 4) & 15))];
          float v398_data = ir1[8];
          ir1[8] = (v398_data + (v384_data * v396_data));
          float v404_data = s0[(27 ^ ((27 >> 4) & 15))];
          float v406_data = ir1[9];
          ir1[9] = (v406_data + (v384_data * v404_data));
          float v412_data = s0[(29 ^ ((29 >> 4) & 15))];
          float v414_data = ir1[10];
          ir1[10] = (v414_data + (v384_data * v412_data));
          float v424_data = r0[10];
          float v437_data = s0[(28 ^ ((28 >> 4) & 15))];
          float v439_data = ir1[9];
          ir1[9] = (v439_data + (v424_data * v437_data));
          float v445_data = s0[(30 ^ ((30 >> 4) & 15))];
          float v447_data = ir1[10];
          ir1[10] = (v447_data + (v424_data * v445_data));
          float v453_data = s0[(32 ^ ((32 >> 4) & 15))];
          float v455_data = ir1[11];
          ir1[11] = (v455_data + (v424_data * v453_data));
          float v464_data = r0[11];
          float v478_data = s0[(31 ^ ((31 >> 4) & 15))];
          float v480_data = ir1[10];
          ir1[10] = (v480_data + (v464_data * v478_data));
          float v486_data = s0[(33 ^ ((33 >> 4) & 15))];
          float v488_data = ir1[11];
          ir1[11] = (v488_data + (v464_data * v486_data));
          float v494_data = s0[(35 ^ ((35 >> 4) & 15))];
          float v496_data = ir1[12];
          ir1[12] = (v496_data + (v464_data * v494_data));
          float v504_data = r0[12];
          float v519_data = s0[(34 ^ ((34 >> 4) & 15))];
          float v521_data = ir1[11];
          ir1[11] = (v521_data + (v504_data * v519_data));
          float v527_data = s0[(36 ^ ((36 >> 4) & 15))];
          float v529_data = ir1[12];
          ir1[12] = (v529_data + (v504_data * v527_data));
          float v535_data = s0[(38 ^ ((38 >> 4) & 15))];
          float v537_data = ir1[13];
          ir1[13] = (v537_data + (v504_data * v535_data));
          float v544_data = r0[13];
          float v560_data = s0[(37 ^ ((37 >> 4) & 15))];
          float v562_data = ir1[12];
          ir1[12] = (v562_data + (v544_data * v560_data));
          float v568_data = s0[(39 ^ ((39 >> 4) & 15))];
          float v570_data = ir1[13];
          ir1[13] = (v570_data + (v544_data * v568_data));
          float v576_data = s0[(41 ^ ((41 >> 4) & 15))];
          float v578_data = ir1[14];
          ir1[14] = (v578_data + (v544_data * v576_data));
          float v584_data = r0[14];
          float v601_data = s0[(40 ^ ((40 >> 4) & 15))];
          float v603_data = ir1[13];
          ir1[13] = (v603_data + (v584_data * v601_data));
          float v609_data = s0[(42 ^ ((42 >> 4) & 15))];
          float v611_data = ir1[14];
          ir1[14] = (v611_data + (v584_data * v609_data));
          float v617_data = s0[(44 ^ ((44 >> 4) & 15))];
          float v619_data = ir1[15];
          ir1[15] = (v619_data + (v584_data * v617_data));
          float v624_data = r0[15];
          float v642_data = s0[(43 ^ ((43 >> 4) & 15))];
          float v644_data = ir1[14];
          ir1[14] = (v644_data + (v624_data * v642_data));
          float v650_data = s0[(45 ^ ((45 >> 4) & 15))];
          float v652_data = ir1[15];
          ir1[15] = (v652_data + (v624_data * v650_data));
          #pragma unroll
          for (int32_t v657_n0 = 0; v657_n0 < 1; ++v657_n0) {
            #pragma unroll
            for (int32_t v658_n1 = 0; v658_n1 < 16; ++v658_n1) {
              int32_t v659_a = v657_n0 + v658_n1;
              float v660_data = ir1[v659_a];
              r1[v659_a] = v660_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v665_i0 = 0; v665_i0 < 1; ++v665_i0) {
            int32_t v673_lead = v10_lead + (v665_i0 * 16);
            #pragma unroll
            for (int32_t v666_i1 = 0; v666_i1 < 16; ++v666_i1) {
              float v668_data = r1[(v665_i0 + v666_i1)];
              glb_m0[(v673_lead + (v666_i1 * 16))] = v668_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

