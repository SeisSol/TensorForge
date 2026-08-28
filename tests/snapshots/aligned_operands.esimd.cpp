// === base name ===
kernel_21138a3fa2

// === header ===
void launcher_kernel_21138a3fa2(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_21138a3fa2(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_21138a3fa2(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_21138a3fa2(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 16×8(16×8) {0..16}×{0..8} strided
        // m1 16×16(16×16) {0..16}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[144 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[128];
          const size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          if (batchId0 < numElements0) {
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 128 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 64];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[128]{};
              // r0 = +(glb_m1 * s0) + None
              // [(0, 16), (0, 8)] [(0, 16)]
              float ir0[128]{};
              int32_t v8_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v12_data;
              v12_data.copy_from(glb_m1 + (0_i32));
              int32_t v15_a = 0_i32 + 16;
              tensorforge::intel_esimd::simd<float, 16> v19_data;
              v19_data.copy_from(glb_m1 + (16_i32));
              int32_t v22_a = 0_i32 + 32;
              tensorforge::intel_esimd::simd<float, 16> v26_data;
              v26_data.copy_from(glb_m1 + (32_i32));
              int32_t v29_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(glb_m1 + (48_i32));
              int32_t v36_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(glb_m1 + (64_i32));
              int32_t v43_a = 0_i32 + 80;
              tensorforge::intel_esimd::simd<float, 16> v47_data;
              v47_data.copy_from(glb_m1 + (80_i32));
              int32_t v50_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v54_data;
              v54_data.copy_from(glb_m1 + (96_i32));
              int32_t v57_a = 0_i32 + 112;
              tensorforge::intel_esimd::simd<float, 16> v61_data;
              v61_data.copy_from(glb_m1 + (112_i32));
              int32_t v64_a = 0_i32 + 128;
              tensorforge::intel_esimd::simd<float, 16> v68_data;
              v68_data.copy_from(glb_m1 + (128_i32));
              int32_t v71_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v75_data;
              v75_data.copy_from(glb_m1 + (144_i32));
              int32_t v78_a = 0_i32 + 160;
              tensorforge::intel_esimd::simd<float, 16> v82_data;
              v82_data.copy_from(glb_m1 + (160_i32));
              int32_t v85_a = 0_i32 + 176;
              tensorforge::intel_esimd::simd<float, 16> v89_data;
              v89_data.copy_from(glb_m1 + (176_i32));
              int32_t v92_a = 0_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v96_data;
              v96_data.copy_from(glb_m1 + (192_i32));
              int32_t v99_a = 0_i32 + 208;
              tensorforge::intel_esimd::simd<float, 16> v103_data;
              v103_data.copy_from(glb_m1 + (208_i32));
              int32_t v106_a = 0_i32 + 224;
              tensorforge::intel_esimd::simd<float, 16> v110_data;
              v110_data.copy_from(glb_m1 + (224_i32));
              int32_t v113_a = 0_i32 + 240;
              tensorforge::intel_esimd::simd<float, 16> v117_data;
              v117_data.copy_from(glb_m1 + (240_i32));
              tensorforge::intel_esimd::simd<float, 16> v118_acc{};
              int32_t v121_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v125_data;
              v125_data.copy_from(s0 + (0_i32));
              v118_acc += ((v125_data[0]) * v12_data);
              v118_acc += ((v125_data[1]) * v19_data);
              v118_acc += ((v125_data[2]) * v26_data);
              v118_acc += ((v125_data[3]) * v33_data);
              v118_acc += ((v125_data[4]) * v40_data);
              v118_acc += ((v125_data[5]) * v47_data);
              v118_acc += ((v125_data[6]) * v54_data);
              v118_acc += ((v125_data[7]) * v61_data);
              v118_acc += ((v125_data[8]) * v68_data);
              v118_acc += ((v125_data[9]) * v75_data);
              v118_acc += ((v125_data[10]) * v82_data);
              v118_acc += ((v125_data[11]) * v89_data);
              v118_acc += ((v125_data[12]) * v96_data);
              v118_acc += ((v125_data[13]) * v103_data);
              v118_acc += ((v125_data[14]) * v110_data);
              v118_acc += ((v125_data[15]) * v117_data);
              v118_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v158_acc{};
              int32_t v161_a = 0_i32 + 16;
              tensorforge::intel_esimd::simd<float, 16> v165_data;
              v165_data.copy_from(s0 + (16_i32));
              v158_acc += ((v165_data[0]) * v12_data);
              v158_acc += ((v165_data[1]) * v19_data);
              v158_acc += ((v165_data[2]) * v26_data);
              v158_acc += ((v165_data[3]) * v33_data);
              v158_acc += ((v165_data[4]) * v40_data);
              v158_acc += ((v165_data[5]) * v47_data);
              v158_acc += ((v165_data[6]) * v54_data);
              v158_acc += ((v165_data[7]) * v61_data);
              v158_acc += ((v165_data[8]) * v68_data);
              v158_acc += ((v165_data[9]) * v75_data);
              v158_acc += ((v165_data[10]) * v82_data);
              v158_acc += ((v165_data[11]) * v89_data);
              v158_acc += ((v165_data[12]) * v96_data);
              v158_acc += ((v165_data[13]) * v103_data);
              v158_acc += ((v165_data[14]) * v110_data);
              v158_acc += ((v165_data[15]) * v117_data);
              v158_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v198_acc{};
              int32_t v201_a = 0_i32 + 32;
              tensorforge::intel_esimd::simd<float, 16> v205_data;
              v205_data.copy_from(s0 + (32_i32));
              v198_acc += ((v205_data[0]) * v12_data);
              v198_acc += ((v205_data[1]) * v19_data);
              v198_acc += ((v205_data[2]) * v26_data);
              v198_acc += ((v205_data[3]) * v33_data);
              v198_acc += ((v205_data[4]) * v40_data);
              v198_acc += ((v205_data[5]) * v47_data);
              v198_acc += ((v205_data[6]) * v54_data);
              v198_acc += ((v205_data[7]) * v61_data);
              v198_acc += ((v205_data[8]) * v68_data);
              v198_acc += ((v205_data[9]) * v75_data);
              v198_acc += ((v205_data[10]) * v82_data);
              v198_acc += ((v205_data[11]) * v89_data);
              v198_acc += ((v205_data[12]) * v96_data);
              v198_acc += ((v205_data[13]) * v103_data);
              v198_acc += ((v205_data[14]) * v110_data);
              v198_acc += ((v205_data[15]) * v117_data);
              v198_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v238_acc{};
              int32_t v241_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v245_data;
              v245_data.copy_from(s0 + (48_i32));
              v238_acc += ((v245_data[0]) * v12_data);
              v238_acc += ((v245_data[1]) * v19_data);
              v238_acc += ((v245_data[2]) * v26_data);
              v238_acc += ((v245_data[3]) * v33_data);
              v238_acc += ((v245_data[4]) * v40_data);
              v238_acc += ((v245_data[5]) * v47_data);
              v238_acc += ((v245_data[6]) * v54_data);
              v238_acc += ((v245_data[7]) * v61_data);
              v238_acc += ((v245_data[8]) * v68_data);
              v238_acc += ((v245_data[9]) * v75_data);
              v238_acc += ((v245_data[10]) * v82_data);
              v238_acc += ((v245_data[11]) * v89_data);
              v238_acc += ((v245_data[12]) * v96_data);
              v238_acc += ((v245_data[13]) * v103_data);
              v238_acc += ((v245_data[14]) * v110_data);
              v238_acc += ((v245_data[15]) * v117_data);
              v238_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v278_acc{};
              int32_t v281_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 16> v285_data;
              v285_data.copy_from(s0 + (64_i32));
              v278_acc += ((v285_data[0]) * v12_data);
              v278_acc += ((v285_data[1]) * v19_data);
              v278_acc += ((v285_data[2]) * v26_data);
              v278_acc += ((v285_data[3]) * v33_data);
              v278_acc += ((v285_data[4]) * v40_data);
              v278_acc += ((v285_data[5]) * v47_data);
              v278_acc += ((v285_data[6]) * v54_data);
              v278_acc += ((v285_data[7]) * v61_data);
              v278_acc += ((v285_data[8]) * v68_data);
              v278_acc += ((v285_data[9]) * v75_data);
              v278_acc += ((v285_data[10]) * v82_data);
              v278_acc += ((v285_data[11]) * v89_data);
              v278_acc += ((v285_data[12]) * v96_data);
              v278_acc += ((v285_data[13]) * v103_data);
              v278_acc += ((v285_data[14]) * v110_data);
              v278_acc += ((v285_data[15]) * v117_data);
              v278_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v318_acc{};
              int32_t v321_a = 0_i32 + 80;
              tensorforge::intel_esimd::simd<float, 16> v325_data;
              v325_data.copy_from(s0 + (80_i32));
              v318_acc += ((v325_data[0]) * v12_data);
              v318_acc += ((v325_data[1]) * v19_data);
              v318_acc += ((v325_data[2]) * v26_data);
              v318_acc += ((v325_data[3]) * v33_data);
              v318_acc += ((v325_data[4]) * v40_data);
              v318_acc += ((v325_data[5]) * v47_data);
              v318_acc += ((v325_data[6]) * v54_data);
              v318_acc += ((v325_data[7]) * v61_data);
              v318_acc += ((v325_data[8]) * v68_data);
              v318_acc += ((v325_data[9]) * v75_data);
              v318_acc += ((v325_data[10]) * v82_data);
              v318_acc += ((v325_data[11]) * v89_data);
              v318_acc += ((v325_data[12]) * v96_data);
              v318_acc += ((v325_data[13]) * v103_data);
              v318_acc += ((v325_data[14]) * v110_data);
              v318_acc += ((v325_data[15]) * v117_data);
              v318_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v358_acc{};
              int32_t v361_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v365_data;
              v365_data.copy_from(s0 + (96_i32));
              v358_acc += ((v365_data[0]) * v12_data);
              v358_acc += ((v365_data[1]) * v19_data);
              v358_acc += ((v365_data[2]) * v26_data);
              v358_acc += ((v365_data[3]) * v33_data);
              v358_acc += ((v365_data[4]) * v40_data);
              v358_acc += ((v365_data[5]) * v47_data);
              v358_acc += ((v365_data[6]) * v54_data);
              v358_acc += ((v365_data[7]) * v61_data);
              v358_acc += ((v365_data[8]) * v68_data);
              v358_acc += ((v365_data[9]) * v75_data);
              v358_acc += ((v365_data[10]) * v82_data);
              v358_acc += ((v365_data[11]) * v89_data);
              v358_acc += ((v365_data[12]) * v96_data);
              v358_acc += ((v365_data[13]) * v103_data);
              v358_acc += ((v365_data[14]) * v110_data);
              v358_acc += ((v365_data[15]) * v117_data);
              v358_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v398_acc{};
              int32_t v401_a = 0_i32 + 112;
              tensorforge::intel_esimd::simd<float, 16> v405_data;
              v405_data.copy_from(s0 + (112_i32));
              v398_acc += ((v405_data[0]) * v12_data);
              v398_acc += ((v405_data[1]) * v19_data);
              v398_acc += ((v405_data[2]) * v26_data);
              v398_acc += ((v405_data[3]) * v33_data);
              v398_acc += ((v405_data[4]) * v40_data);
              v398_acc += ((v405_data[5]) * v47_data);
              v398_acc += ((v405_data[6]) * v54_data);
              v398_acc += ((v405_data[7]) * v61_data);
              v398_acc += ((v405_data[8]) * v68_data);
              v398_acc += ((v405_data[9]) * v75_data);
              v398_acc += ((v405_data[10]) * v82_data);
              v398_acc += ((v405_data[11]) * v89_data);
              v398_acc += ((v405_data[12]) * v96_data);
              v398_acc += ((v405_data[13]) * v103_data);
              v398_acc += ((v405_data[14]) * v110_data);
              v398_acc += ((v405_data[15]) * v117_data);
              v398_acc.copy_to(ir0 + (112));
              #pragma unroll
              for (int32_t v438_n0 = 0; v438_n0 < 1; ++v438_n0) {
                int32_t v440_a = v438_n0 * 16;
                #pragma unroll
                for (int32_t v439_n1 = 0; v439_n1 < 8; ++v439_n1) {
                  int32_t v441_a = v439_n1 * 16;
                  int32_t v442_a = v440_a + v441_a;
                  int32_t v445_a = v440_a + v441_a;
                  tensorforge::intel_esimd::simd<float, 16> v446_data;
                  v446_data.copy_from(ir0 + (v445_a));
                  v446_data.copy_to(r0 + (v445_a));
                }
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v450_i0 = 0; v450_i0 < 1; ++v450_i0) {
                int32_t v452_a = v450_i0 * 16;
                #pragma unroll
                for (int32_t v451_i1 = 0; v451_i1 < 8; ++v451_i1) {
                  int32_t v453_a = v451_i1 * 16;
                  int32_t v454_a = v452_a + v453_a;
                  int32_t v457_a = v452_a + v453_a;
                  tensorforge::intel_esimd::simd<float, 16> v458_data;
                  v458_data.copy_from(r0 + (v457_a));
                  v458_data.copy_to(glb_m0 + (v457_a));
                }
              }
            }
          }
        }
      });
    }
  });
}

