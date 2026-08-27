// === base name ===
kernel_4b59b6f027

// === header ===
void launcher_kernel_4b59b6f027(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_4b59b6f027(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4b59b6f027(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4b59b6f027(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 16×8(12×8) {4..16}×{0..8} strided
        // m1 16×16(12×16) {4..16}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 16×8(12×8) {4..16}×{0..8} strided({4..16}×{0..8})[0, 1] = m1 16×16(12×16) {4..16}×{0..16} strided({4..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 64];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[8]{};
              ;
              // r0 = +(glb_m1 * s0) + None
              // [(4, 16), (0, 8)] [(0, 16)]
              float ir0[8]{};
              int32_t v8_lead = item.get_local_id(0) % 16;
              if (v8_lead >= 4) {
                int32_t v12_a = -4_i32 + 0;
                int32_t v15_a = -4_i32 + 0;
                int32_t v18_a = -4_i32 + 0;
                int32_t v21_a = -4_i32 + 0;
                int32_t v24_a = -4_i32 + 0;
                int32_t v27_a = -4_i32 + 0;
                int32_t v30_a = -4_i32 + 0;
                int32_t v33_a = -4_i32 + 0;
              }
              if (v8_lead >= 4) {
                int32_t v40_a = -4_i32 + 12;
                int32_t v43_a = -4_i32 + 12;
                int32_t v46_a = -4_i32 + 12;
                int32_t v49_a = -4_i32 + 12;
                int32_t v52_a = -4_i32 + 12;
                int32_t v55_a = -4_i32 + 12;
                int32_t v58_a = -4_i32 + 12;
                int32_t v61_a = -4_i32 + 12;
              }
              if (v8_lead >= 4) {
                int32_t v68_a = -4_i32 + 24;
                int32_t v71_a = -4_i32 + 24;
                int32_t v74_a = -4_i32 + 24;
                int32_t v77_a = -4_i32 + 24;
                int32_t v80_a = -4_i32 + 24;
                int32_t v83_a = -4_i32 + 24;
                int32_t v86_a = -4_i32 + 24;
                int32_t v89_a = -4_i32 + 24;
              }
              if (v8_lead >= 4) {
                int32_t v96_a = -4_i32 + 36;
                int32_t v99_a = -4_i32 + 36;
                int32_t v102_a = -4_i32 + 36;
                int32_t v105_a = -4_i32 + 36;
                int32_t v108_a = -4_i32 + 36;
                int32_t v111_a = -4_i32 + 36;
                int32_t v114_a = -4_i32 + 36;
                int32_t v117_a = -4_i32 + 36;
              }
              if (v8_lead >= 4) {
                int32_t v124_a = -4_i32 + 48;
                int32_t v127_a = -4_i32 + 48;
                int32_t v130_a = -4_i32 + 48;
                int32_t v133_a = -4_i32 + 48;
                int32_t v136_a = -4_i32 + 48;
                int32_t v139_a = -4_i32 + 48;
                int32_t v142_a = -4_i32 + 48;
                int32_t v145_a = -4_i32 + 48;
              }
              if (v8_lead >= 4) {
                int32_t v152_a = -4_i32 + 60;
                int32_t v155_a = -4_i32 + 60;
                int32_t v158_a = -4_i32 + 60;
                int32_t v161_a = -4_i32 + 60;
                int32_t v164_a = -4_i32 + 60;
                int32_t v167_a = -4_i32 + 60;
                int32_t v170_a = -4_i32 + 60;
                int32_t v173_a = -4_i32 + 60;
              }
              if (v8_lead >= 4) {
                int32_t v180_a = -4_i32 + 72;
                int32_t v183_a = -4_i32 + 72;
                int32_t v186_a = -4_i32 + 72;
                int32_t v189_a = -4_i32 + 72;
                int32_t v192_a = -4_i32 + 72;
                int32_t v195_a = -4_i32 + 72;
                int32_t v198_a = -4_i32 + 72;
                int32_t v201_a = -4_i32 + 72;
              }
              if (v8_lead >= 4) {
                int32_t v208_a = -4_i32 + 84;
                int32_t v211_a = -4_i32 + 84;
                int32_t v214_a = -4_i32 + 84;
                int32_t v217_a = -4_i32 + 84;
                int32_t v220_a = -4_i32 + 84;
                int32_t v223_a = -4_i32 + 84;
                int32_t v226_a = -4_i32 + 84;
                int32_t v229_a = -4_i32 + 84;
              }
              if (v8_lead >= 4) {
                int32_t v236_a = -4_i32 + 96;
                int32_t v239_a = -4_i32 + 96;
                int32_t v242_a = -4_i32 + 96;
                int32_t v245_a = -4_i32 + 96;
                int32_t v248_a = -4_i32 + 96;
                int32_t v251_a = -4_i32 + 96;
                int32_t v254_a = -4_i32 + 96;
                int32_t v257_a = -4_i32 + 96;
              }
              if (v8_lead >= 4) {
                int32_t v264_a = -4_i32 + 108;
                int32_t v267_a = -4_i32 + 108;
                int32_t v270_a = -4_i32 + 108;
                int32_t v273_a = -4_i32 + 108;
                int32_t v276_a = -4_i32 + 108;
                int32_t v279_a = -4_i32 + 108;
                int32_t v282_a = -4_i32 + 108;
                int32_t v285_a = -4_i32 + 108;
              }
              if (v8_lead >= 4) {
                int32_t v292_a = -4_i32 + 120;
                int32_t v295_a = -4_i32 + 120;
                int32_t v298_a = -4_i32 + 120;
                int32_t v301_a = -4_i32 + 120;
                int32_t v304_a = -4_i32 + 120;
                int32_t v307_a = -4_i32 + 120;
                int32_t v310_a = -4_i32 + 120;
                int32_t v313_a = -4_i32 + 120;
              }
              if (v8_lead >= 4) {
                int32_t v320_a = -4_i32 + 132;
                int32_t v323_a = -4_i32 + 132;
                int32_t v326_a = -4_i32 + 132;
                int32_t v329_a = -4_i32 + 132;
                int32_t v332_a = -4_i32 + 132;
                int32_t v335_a = -4_i32 + 132;
                int32_t v338_a = -4_i32 + 132;
                int32_t v341_a = -4_i32 + 132;
              }
              if (v8_lead >= 4) {
                int32_t v348_a = -4_i32 + 144;
                int32_t v351_a = -4_i32 + 144;
                int32_t v354_a = -4_i32 + 144;
                int32_t v357_a = -4_i32 + 144;
                int32_t v360_a = -4_i32 + 144;
                int32_t v363_a = -4_i32 + 144;
                int32_t v366_a = -4_i32 + 144;
                int32_t v369_a = -4_i32 + 144;
              }
              if (v8_lead >= 4) {
                int32_t v376_a = -4_i32 + 156;
                int32_t v379_a = -4_i32 + 156;
                int32_t v382_a = -4_i32 + 156;
                int32_t v385_a = -4_i32 + 156;
                int32_t v388_a = -4_i32 + 156;
                int32_t v391_a = -4_i32 + 156;
                int32_t v394_a = -4_i32 + 156;
                int32_t v397_a = -4_i32 + 156;
              }
              if (v8_lead >= 4) {
                int32_t v404_a = -4_i32 + 168;
                int32_t v407_a = -4_i32 + 168;
                int32_t v410_a = -4_i32 + 168;
                int32_t v413_a = -4_i32 + 168;
                int32_t v416_a = -4_i32 + 168;
                int32_t v419_a = -4_i32 + 168;
                int32_t v422_a = -4_i32 + 168;
                int32_t v425_a = -4_i32 + 168;
              }
              if (v8_lead >= 4) {
                int32_t v432_a = -4_i32 + 180;
                int32_t v435_a = -4_i32 + 180;
                int32_t v438_a = -4_i32 + 180;
                int32_t v441_a = -4_i32 + 180;
                int32_t v444_a = -4_i32 + 180;
                int32_t v447_a = -4_i32 + 180;
                int32_t v450_a = -4_i32 + 180;
                int32_t v453_a = -4_i32 + 180;
              }
              if (v8_lead >= 4) {
                #pragma unroll
                for (int32_t v458_n1 = 0; v458_n1 < 8; ++v458_n1) {
                  int32_t v459_a = 0 + v458_n1;
                  int32_t v460_a = 0 + v458_n1;
                  None = r0[v460_a];
                }
              }
              // glb_m0 = store{r>g}(r0);
              if (v8_lead >= 4) {
                #pragma unroll
                for (int32_t v465_i1 = 0; v465_i1 < 8; ++v465_i1) {
                  int32_t v466_a = 0 + v465_i1;
                  int32_t v470_a = -4_i32 + (v465_i1 * 12);
                  None.copy_to(glb_m0[v470_a]);
                }
              }
            }
          }
        }
      });
    }
  });
}

