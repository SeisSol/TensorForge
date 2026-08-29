// === base name ===
kernel_417e1ddcc4

// === header ===
void launcher_kernel_417e1ddcc4(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_417e1ddcc4(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  dim3 block (16, 16, 1);
  static std::size_t gridsize = 0;
      if (gridsize == 0) {
        int device, smCount, blocksPerSM;
        cudaGetDevice(&device);
        CHECK_ERR;
        cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, device);
        CHECK_ERR;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocksPerSM, kernel_kernel_417e1ddcc4, block.x * block.y * block.z, 1024 * sizeof(double));
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
        cudaFuncSetAttribute(kernel_kernel_417e1ddcc4, cudaFuncAttributeMaxDynamicSharedMemorySize, 1024 * sizeof(double));
        CHECK_ERR;
        shmemsizeset = true;
      }
      
  cudaStream_t stream = (streamPtr != nullptr) ? static_cast<cudaStream_t>(streamPtr) : 0;
  kernel_kernel_417e1ddcc4<<<grid,block,1024 * sizeof(double),stream>>>( m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
__global__ void 
__launch_bounds__(256)
 kernel_kernel_417e1ddcc4(double* m0, unsigned m0_extraOffset, const double* m1, unsigned m1_extraOffset, const double* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
      auto* totalShrMem = reinterpret_cast<double*>(totalShrMemPtr);
      double* localShrMem0 = &totalShrMem[64 * threadIdx.y + 0];
      double* tempShrMem = &localShrMem0[48];
      for (size_t batchId0 = threadIdx.y + blockDim.y * (blockIdx.x); batchId0 < numElements0; batchId0 += (gridDim.x * blockDim.y)) {
        const auto batchId1 = batchId0 + (gridDim.x * blockDim.y) < numElements0 ? batchId0 + (gridDim.x * blockDim.y) : batchId0;
        const auto batchId2 = batchId1 + (gridDim.x * blockDim.y) < numElements0 ? batchId1 + (gridDim.x * blockDim.y) : batchId1;
        const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
        if (allowed) {
          double *const __restrict__ glb_m0 = &m0[batchId0 * 256 + 0 + m0_extraOffset];
          const double *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
          const double *const __restrict__ glb_m2 = &m2[batchId0 * 256 + 0 + m2_extraOffset];
          double r0[16]{};
          // r0 = load{g>r}(glb_m1);
          int32_t v10_lead = threadIdx.x % 16;
          #pragma unroll
          for (int32_t v11_i0 = 0; v11_i0 < 1; ++v11_i0) {
            int32_t v17_lead = v10_lead + (v11_i0 * 16);
            #pragma unroll
            for (int32_t v12_i1 = 0; v12_i1 < 16; ++v12_i1) {
              double v20_data = __ldcg(&glb_m1[(v17_lead + (v12_i1 * 16))]);
              r0[(v11_i0 + v12_i1)] = v20_data;
            }
          }
          double* __restrict__ s0 = &localShrMem0[0];
          // s0 = load{g>s}(glb_m2[0, 1])
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 0], &glb_m2[0 + 0 + 1 * threadIdx.x + 0], 8);
          __pipeline_commit();
          __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 16], &glb_m2[0 + 0 + 1 * threadIdx.x + 16], 8);
          __pipeline_commit();
          if (threadIdx.x < 14) {
            __pipeline_memcpy_async(&s0[0 + 0 + 1 * threadIdx.x + 32], &glb_m2[0 + 0 + 1 * threadIdx.x + 32], 8);
            __pipeline_commit();
          }
          // wait(r0 = load{g>r}(glb_m1););
          // wait(s0 = load{g>s}(glb_m2[0, 1]));
          __pipeline_wait_prior(0);
          double r1[16]{};
          __syncwarp();
          // r1 = +(r0 * s0) + None
          // [(0, 16), (0, 16)] [(0, 16)]
          double ir1[16]{};
          double v31_data = r0[0];
          double v35_data = s0[(0 ^ ((0 >> 4) & 15))];
          double v37_data = ir1[0];
          ir1[0] = (v37_data + (v31_data * v35_data));
          double v43_data = s0[(2 ^ ((2 >> 4) & 15))];
          double v45_data = ir1[1];
          ir1[1] = (v45_data + (v31_data * v43_data));
          double v64_data = r0[1];
          double v68_data = s0[(1 ^ ((1 >> 4) & 15))];
          double v70_data = ir1[0];
          ir1[0] = (v70_data + (v64_data * v68_data));
          double v76_data = s0[(3 ^ ((3 >> 4) & 15))];
          double v78_data = ir1[1];
          ir1[1] = (v78_data + (v64_data * v76_data));
          double v84_data = s0[(5 ^ ((5 >> 4) & 15))];
          double v86_data = ir1[2];
          ir1[2] = (v86_data + (v64_data * v84_data));
          double v104_data = r0[2];
          double v109_data = s0[(4 ^ ((4 >> 4) & 15))];
          double v111_data = ir1[1];
          ir1[1] = (v111_data + (v104_data * v109_data));
          double v117_data = s0[(6 ^ ((6 >> 4) & 15))];
          double v119_data = ir1[2];
          ir1[2] = (v119_data + (v104_data * v117_data));
          double v125_data = s0[(8 ^ ((8 >> 4) & 15))];
          double v127_data = ir1[3];
          ir1[3] = (v127_data + (v104_data * v125_data));
          double v144_data = r0[3];
          double v150_data = s0[(7 ^ ((7 >> 4) & 15))];
          double v152_data = ir1[2];
          ir1[2] = (v152_data + (v144_data * v150_data));
          double v158_data = s0[(9 ^ ((9 >> 4) & 15))];
          double v160_data = ir1[3];
          ir1[3] = (v160_data + (v144_data * v158_data));
          double v166_data = s0[(11 ^ ((11 >> 4) & 15))];
          double v168_data = ir1[4];
          ir1[4] = (v168_data + (v144_data * v166_data));
          double v184_data = r0[4];
          double v191_data = s0[(10 ^ ((10 >> 4) & 15))];
          double v193_data = ir1[3];
          ir1[3] = (v193_data + (v184_data * v191_data));
          double v199_data = s0[(12 ^ ((12 >> 4) & 15))];
          double v201_data = ir1[4];
          ir1[4] = (v201_data + (v184_data * v199_data));
          double v207_data = s0[(14 ^ ((14 >> 4) & 15))];
          double v209_data = ir1[5];
          ir1[5] = (v209_data + (v184_data * v207_data));
          double v224_data = r0[5];
          double v232_data = s0[(13 ^ ((13 >> 4) & 15))];
          double v234_data = ir1[4];
          ir1[4] = (v234_data + (v224_data * v232_data));
          double v240_data = s0[(15 ^ ((15 >> 4) & 15))];
          double v242_data = ir1[5];
          ir1[5] = (v242_data + (v224_data * v240_data));
          double v248_data = s0[(17 ^ ((17 >> 4) & 15))];
          double v250_data = ir1[6];
          ir1[6] = (v250_data + (v224_data * v248_data));
          double v264_data = r0[6];
          double v273_data = s0[(16 ^ ((16 >> 4) & 15))];
          double v275_data = ir1[5];
          ir1[5] = (v275_data + (v264_data * v273_data));
          double v281_data = s0[(18 ^ ((18 >> 4) & 15))];
          double v283_data = ir1[6];
          ir1[6] = (v283_data + (v264_data * v281_data));
          double v289_data = s0[(20 ^ ((20 >> 4) & 15))];
          double v291_data = ir1[7];
          ir1[7] = (v291_data + (v264_data * v289_data));
          double v304_data = r0[7];
          double v314_data = s0[(19 ^ ((19 >> 4) & 15))];
          double v316_data = ir1[6];
          ir1[6] = (v316_data + (v304_data * v314_data));
          double v322_data = s0[(21 ^ ((21 >> 4) & 15))];
          double v324_data = ir1[7];
          ir1[7] = (v324_data + (v304_data * v322_data));
          double v330_data = s0[(23 ^ ((23 >> 4) & 15))];
          double v332_data = ir1[8];
          ir1[8] = (v332_data + (v304_data * v330_data));
          double v344_data = r0[8];
          double v355_data = s0[(22 ^ ((22 >> 4) & 15))];
          double v357_data = ir1[7];
          ir1[7] = (v357_data + (v344_data * v355_data));
          double v363_data = s0[(24 ^ ((24 >> 4) & 15))];
          double v365_data = ir1[8];
          ir1[8] = (v365_data + (v344_data * v363_data));
          double v371_data = s0[(26 ^ ((26 >> 4) & 15))];
          double v373_data = ir1[9];
          ir1[9] = (v373_data + (v344_data * v371_data));
          double v384_data = r0[9];
          double v396_data = s0[(25 ^ ((25 >> 4) & 15))];
          double v398_data = ir1[8];
          ir1[8] = (v398_data + (v384_data * v396_data));
          double v404_data = s0[(27 ^ ((27 >> 4) & 15))];
          double v406_data = ir1[9];
          ir1[9] = (v406_data + (v384_data * v404_data));
          double v412_data = s0[(29 ^ ((29 >> 4) & 15))];
          double v414_data = ir1[10];
          ir1[10] = (v414_data + (v384_data * v412_data));
          double v424_data = r0[10];
          double v437_data = s0[(28 ^ ((28 >> 4) & 15))];
          double v439_data = ir1[9];
          ir1[9] = (v439_data + (v424_data * v437_data));
          double v445_data = s0[(30 ^ ((30 >> 4) & 15))];
          double v447_data = ir1[10];
          ir1[10] = (v447_data + (v424_data * v445_data));
          double v453_data = s0[(32 ^ ((32 >> 4) & 15))];
          double v455_data = ir1[11];
          ir1[11] = (v455_data + (v424_data * v453_data));
          double v464_data = r0[11];
          double v478_data = s0[(31 ^ ((31 >> 4) & 15))];
          double v480_data = ir1[10];
          ir1[10] = (v480_data + (v464_data * v478_data));
          double v486_data = s0[(33 ^ ((33 >> 4) & 15))];
          double v488_data = ir1[11];
          ir1[11] = (v488_data + (v464_data * v486_data));
          double v494_data = s0[(35 ^ ((35 >> 4) & 15))];
          double v496_data = ir1[12];
          ir1[12] = (v496_data + (v464_data * v494_data));
          double v504_data = r0[12];
          double v519_data = s0[(34 ^ ((34 >> 4) & 15))];
          double v521_data = ir1[11];
          ir1[11] = (v521_data + (v504_data * v519_data));
          double v527_data = s0[(36 ^ ((36 >> 4) & 15))];
          double v529_data = ir1[12];
          ir1[12] = (v529_data + (v504_data * v527_data));
          double v535_data = s0[(38 ^ ((38 >> 4) & 15))];
          double v537_data = ir1[13];
          ir1[13] = (v537_data + (v504_data * v535_data));
          double v544_data = r0[13];
          double v560_data = s0[(37 ^ ((37 >> 4) & 15))];
          double v562_data = ir1[12];
          ir1[12] = (v562_data + (v544_data * v560_data));
          double v568_data = s0[(39 ^ ((39 >> 4) & 15))];
          double v570_data = ir1[13];
          ir1[13] = (v570_data + (v544_data * v568_data));
          double v576_data = s0[(41 ^ ((41 >> 4) & 15))];
          double v578_data = ir1[14];
          ir1[14] = (v578_data + (v544_data * v576_data));
          double v584_data = r0[14];
          double v601_data = s0[(40 ^ ((40 >> 4) & 15))];
          double v603_data = ir1[13];
          ir1[13] = (v603_data + (v584_data * v601_data));
          double v609_data = s0[(42 ^ ((42 >> 4) & 15))];
          double v611_data = ir1[14];
          ir1[14] = (v611_data + (v584_data * v609_data));
          double v617_data = s0[(44 ^ ((44 >> 4) & 15))];
          double v619_data = ir1[15];
          ir1[15] = (v619_data + (v584_data * v617_data));
          double v624_data = r0[15];
          double v642_data = s0[(43 ^ ((43 >> 4) & 15))];
          double v644_data = ir1[14];
          ir1[14] = (v644_data + (v624_data * v642_data));
          double v650_data = s0[(45 ^ ((45 >> 4) & 15))];
          double v652_data = ir1[15];
          ir1[15] = (v652_data + (v624_data * v650_data));
          #pragma unroll
          for (int32_t v657_n0 = 0; v657_n0 < 1; ++v657_n0) {
            #pragma unroll
            for (int32_t v658_n1 = 0; v658_n1 < 16; ++v658_n1) {
              int32_t v659_a = v657_n0 + v658_n1;
              double v660_data = ir1[v659_a];
              r1[v659_a] = v660_data;
            }
          }
          // glb_m0 = store{r>g}(r1);
          #pragma unroll
          for (int32_t v665_i0 = 0; v665_i0 < 1; ++v665_i0) {
            int32_t v673_lead = v10_lead + (v665_i0 * 16);
            #pragma unroll
            for (int32_t v666_i1 = 0; v666_i1 < 16; ++v666_i1) {
              double v668_data = r1[(v665_i0 + v666_i1)];
              glb_m0[(v673_lead + (v666_i1 * 16))] = v668_data;
            }
          }
          __syncwarp();
        }
      }
    }
  }
}

