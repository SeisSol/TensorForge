// === base name ===
kernel_87f2838a59

// === header ===
void launcher_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_87f2838a59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_87f2838a59(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_87f2838a59(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 16×8(16×8) {0..16}×{0..8} strided
        // m1 32×32(32×32) {0..32}×{0..32} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 32×32(32×32) {0..32}×{0..32} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[144 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[128];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            bool allowed = true;
            if (flags0 != nullptr) {
              allowed = static_cast<bool>(flags0[batchId0]);
            }
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 1024 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 64];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[8]{};
              ;
              // r0 = +(glb_m1 * s0) + None
              // [(0, 16), (0, 8)] [(0, 16)]
              float ir0[8]{};
              int32_t v11_a = 8_i32 + 256;
              int32_t v14_a = 8_i32 + 256;
              int32_t v17_a = 8_i32 + 256;
              int32_t v20_a = 8_i32 + 256;
              int32_t v23_a = 8_i32 + 256;
              int32_t v26_a = 8_i32 + 256;
              int32_t v29_a = 8_i32 + 256;
              int32_t v32_a = 8_i32 + 256;
              int32_t v38_a = 8_i32 + 288;
              int32_t v41_a = 8_i32 + 288;
              int32_t v44_a = 8_i32 + 288;
              int32_t v47_a = 8_i32 + 288;
              int32_t v50_a = 8_i32 + 288;
              int32_t v53_a = 8_i32 + 288;
              int32_t v56_a = 8_i32 + 288;
              int32_t v59_a = 8_i32 + 288;
              int32_t v65_a = 8_i32 + 320;
              int32_t v68_a = 8_i32 + 320;
              int32_t v71_a = 8_i32 + 320;
              int32_t v74_a = 8_i32 + 320;
              int32_t v77_a = 8_i32 + 320;
              int32_t v80_a = 8_i32 + 320;
              int32_t v83_a = 8_i32 + 320;
              int32_t v86_a = 8_i32 + 320;
              int32_t v92_a = 8_i32 + 352;
              int32_t v95_a = 8_i32 + 352;
              int32_t v98_a = 8_i32 + 352;
              int32_t v101_a = 8_i32 + 352;
              int32_t v104_a = 8_i32 + 352;
              int32_t v107_a = 8_i32 + 352;
              int32_t v110_a = 8_i32 + 352;
              int32_t v113_a = 8_i32 + 352;
              int32_t v119_a = 8_i32 + 384;
              int32_t v122_a = 8_i32 + 384;
              int32_t v125_a = 8_i32 + 384;
              int32_t v128_a = 8_i32 + 384;
              int32_t v131_a = 8_i32 + 384;
              int32_t v134_a = 8_i32 + 384;
              int32_t v137_a = 8_i32 + 384;
              int32_t v140_a = 8_i32 + 384;
              int32_t v146_a = 8_i32 + 416;
              int32_t v149_a = 8_i32 + 416;
              int32_t v152_a = 8_i32 + 416;
              int32_t v155_a = 8_i32 + 416;
              int32_t v158_a = 8_i32 + 416;
              int32_t v161_a = 8_i32 + 416;
              int32_t v164_a = 8_i32 + 416;
              int32_t v167_a = 8_i32 + 416;
              int32_t v173_a = 8_i32 + 448;
              int32_t v176_a = 8_i32 + 448;
              int32_t v179_a = 8_i32 + 448;
              int32_t v182_a = 8_i32 + 448;
              int32_t v185_a = 8_i32 + 448;
              int32_t v188_a = 8_i32 + 448;
              int32_t v191_a = 8_i32 + 448;
              int32_t v194_a = 8_i32 + 448;
              int32_t v200_a = 8_i32 + 480;
              int32_t v203_a = 8_i32 + 480;
              int32_t v206_a = 8_i32 + 480;
              int32_t v209_a = 8_i32 + 480;
              int32_t v212_a = 8_i32 + 480;
              int32_t v215_a = 8_i32 + 480;
              int32_t v218_a = 8_i32 + 480;
              int32_t v221_a = 8_i32 + 480;
              int32_t v227_a = 8_i32 + 512;
              int32_t v230_a = 8_i32 + 512;
              int32_t v233_a = 8_i32 + 512;
              int32_t v236_a = 8_i32 + 512;
              int32_t v239_a = 8_i32 + 512;
              int32_t v242_a = 8_i32 + 512;
              int32_t v245_a = 8_i32 + 512;
              int32_t v248_a = 8_i32 + 512;
              int32_t v254_a = 8_i32 + 544;
              int32_t v257_a = 8_i32 + 544;
              int32_t v260_a = 8_i32 + 544;
              int32_t v263_a = 8_i32 + 544;
              int32_t v266_a = 8_i32 + 544;
              int32_t v269_a = 8_i32 + 544;
              int32_t v272_a = 8_i32 + 544;
              int32_t v275_a = 8_i32 + 544;
              int32_t v281_a = 8_i32 + 576;
              int32_t v284_a = 8_i32 + 576;
              int32_t v287_a = 8_i32 + 576;
              int32_t v290_a = 8_i32 + 576;
              int32_t v293_a = 8_i32 + 576;
              int32_t v296_a = 8_i32 + 576;
              int32_t v299_a = 8_i32 + 576;
              int32_t v302_a = 8_i32 + 576;
              int32_t v308_a = 8_i32 + 608;
              int32_t v311_a = 8_i32 + 608;
              int32_t v314_a = 8_i32 + 608;
              int32_t v317_a = 8_i32 + 608;
              int32_t v320_a = 8_i32 + 608;
              int32_t v323_a = 8_i32 + 608;
              int32_t v326_a = 8_i32 + 608;
              int32_t v329_a = 8_i32 + 608;
              int32_t v335_a = 8_i32 + 640;
              int32_t v338_a = 8_i32 + 640;
              int32_t v341_a = 8_i32 + 640;
              int32_t v344_a = 8_i32 + 640;
              int32_t v347_a = 8_i32 + 640;
              int32_t v350_a = 8_i32 + 640;
              int32_t v353_a = 8_i32 + 640;
              int32_t v356_a = 8_i32 + 640;
              int32_t v362_a = 8_i32 + 672;
              int32_t v365_a = 8_i32 + 672;
              int32_t v368_a = 8_i32 + 672;
              int32_t v371_a = 8_i32 + 672;
              int32_t v374_a = 8_i32 + 672;
              int32_t v377_a = 8_i32 + 672;
              int32_t v380_a = 8_i32 + 672;
              int32_t v383_a = 8_i32 + 672;
              int32_t v389_a = 8_i32 + 704;
              int32_t v392_a = 8_i32 + 704;
              int32_t v395_a = 8_i32 + 704;
              int32_t v398_a = 8_i32 + 704;
              int32_t v401_a = 8_i32 + 704;
              int32_t v404_a = 8_i32 + 704;
              int32_t v407_a = 8_i32 + 704;
              int32_t v410_a = 8_i32 + 704;
              int32_t v416_a = 8_i32 + 736;
              int32_t v419_a = 8_i32 + 736;
              int32_t v422_a = 8_i32 + 736;
              int32_t v425_a = 8_i32 + 736;
              int32_t v428_a = 8_i32 + 736;
              int32_t v431_a = 8_i32 + 736;
              int32_t v434_a = 8_i32 + 736;
              int32_t v437_a = 8_i32 + 736;
              #pragma unroll
              for (int32_t v441_n0 = 0; v441_n0 < 1; ++v441_n0) {
                #pragma unroll
                for (int32_t v442_n1 = 0; v442_n1 < 8; ++v442_n1) {
                  int32_t v443_a = v441_n0 + v442_n1;
                  int32_t v444_a = v441_n0 + v442_n1;
                  None = r0[v444_a];
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v448_i0 = 0; v448_i0 < 1; ++v448_i0) {
                int32_t v451_lead = v448_i0 * 16;
                #pragma unroll
                for (int32_t v449_i1 = 0; v449_i1 < 8; ++v449_i1) {
                  int32_t v450_a = v448_i0 + v449_i1;
                  int32_t v453_a = v451_lead + (v449_i1 * 16);
                  None.copy_to(glb_m0[v453_a]);
                }
              }
            }
          }
        }
      });
    }
  });
}

