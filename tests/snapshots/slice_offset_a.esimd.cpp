// === base name ===
kernel_f61651fe59

// === header ===
void launcher_kernel_f61651fe59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_f61651fe59(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_f61651fe59(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_f61651fe59(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 32×16(12×16) {4..16}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 32×16(12×16) {4..16}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float* __restrict__ s0 = &localShrMem0[0];
              // s0 = load{g>s}(glb_m2[0, 1])
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 0] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 0];
              *(sycl::vec<float, 4>*)&s0[0 + 0 + 4 * item.get_local_id(0) + 64] = *(sycl::vec<float, 4>*)&glb_m2[0 + 0 + 4 * item.get_local_id(0) + 64];
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r0[128]{};
              // r0 = +(glb_m1 * s0) + None
              // [(0, 12), (0, 8)] [(0, 16)]
              float ir0[128]{};
              int32_t v10_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v16_data;
              v16_data.copy_from(glb_m1 + (0_i32));
              int32_t v21_a = 0_i32 + 12;
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(glb_m1 + (12_i32));
              int32_t v32_a = 0_i32 + 24;
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(glb_m1 + (24_i32));
              int32_t v43_a = 0_i32 + 36;
              tensorforge::intel_esimd::simd<float, 16> v49_data;
              v49_data.copy_from(glb_m1 + (36_i32));
              int32_t v54_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v60_data;
              v60_data.copy_from(glb_m1 + (48_i32));
              int32_t v65_a = 0_i32 + 60;
              tensorforge::intel_esimd::simd<float, 16> v71_data;
              v71_data.copy_from(glb_m1 + (60_i32));
              int32_t v76_a = 0_i32 + 72;
              tensorforge::intel_esimd::simd<float, 16> v82_data;
              v82_data.copy_from(glb_m1 + (72_i32));
              int32_t v87_a = 0_i32 + 84;
              tensorforge::intel_esimd::simd<float, 16> v93_data;
              v93_data.copy_from(glb_m1 + (84_i32));
              int32_t v98_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v104_data;
              v104_data.copy_from(glb_m1 + (96_i32));
              int32_t v109_a = 0_i32 + 108;
              tensorforge::intel_esimd::simd<float, 16> v115_data;
              v115_data.copy_from(glb_m1 + (108_i32));
              int32_t v120_a = 0_i32 + 120;
              tensorforge::intel_esimd::simd<float, 16> v126_data;
              v126_data.copy_from(glb_m1 + (120_i32));
              int32_t v131_a = 0_i32 + 132;
              tensorforge::intel_esimd::simd<float, 16> v137_data;
              v137_data.copy_from(glb_m1 + (132_i32));
              int32_t v142_a = 0_i32 + 144;
              tensorforge::intel_esimd::simd<float, 16> v148_data;
              v148_data.copy_from(glb_m1 + (144_i32));
              int32_t v153_a = 0_i32 + 156;
              tensorforge::intel_esimd::simd<float, 16> v159_data;
              v159_data.copy_from(glb_m1 + (156_i32));
              int32_t v164_a = 0_i32 + 168;
              tensorforge::intel_esimd::simd<float, 16> v170_data;
              v170_data.copy_from(glb_m1 + (168_i32));
              int32_t v175_a = 0_i32 + 180;
              tensorforge::intel_esimd::simd<float, 16> v181_data;
              v181_data.copy_from(glb_m1 + (180_i32));
              tensorforge::intel_esimd::simd<float, 16> v182_acc{};
              int32_t v185_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v189_data;
              v189_data.copy_from(s0 + (0_i32));
              v182_acc += ((v189_data[0]) * v16_data);
              v182_acc += ((v189_data[1]) * v27_data);
              v182_acc += ((v189_data[2]) * v38_data);
              v182_acc += ((v189_data[3]) * v49_data);
              v182_acc += ((v189_data[4]) * v60_data);
              v182_acc += ((v189_data[5]) * v71_data);
              v182_acc += ((v189_data[6]) * v82_data);
              v182_acc += ((v189_data[7]) * v93_data);
              v182_acc += ((v189_data[8]) * v104_data);
              v182_acc += ((v189_data[9]) * v115_data);
              v182_acc += ((v189_data[10]) * v126_data);
              v182_acc += ((v189_data[11]) * v137_data);
              v182_acc += ((v189_data[12]) * v148_data);
              v182_acc += ((v189_data[13]) * v159_data);
              v182_acc += ((v189_data[14]) * v170_data);
              v182_acc += ((v189_data[15]) * v181_data);
              v182_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v222_acc{};
              int32_t v225_a = 0_i32 + 16;
              tensorforge::intel_esimd::simd<float, 16> v229_data;
              v229_data.copy_from(s0 + (16_i32));
              v222_acc += ((v229_data[0]) * v16_data);
              v222_acc += ((v229_data[1]) * v27_data);
              v222_acc += ((v229_data[2]) * v38_data);
              v222_acc += ((v229_data[3]) * v49_data);
              v222_acc += ((v229_data[4]) * v60_data);
              v222_acc += ((v229_data[5]) * v71_data);
              v222_acc += ((v229_data[6]) * v82_data);
              v222_acc += ((v229_data[7]) * v93_data);
              v222_acc += ((v229_data[8]) * v104_data);
              v222_acc += ((v229_data[9]) * v115_data);
              v222_acc += ((v229_data[10]) * v126_data);
              v222_acc += ((v229_data[11]) * v137_data);
              v222_acc += ((v229_data[12]) * v148_data);
              v222_acc += ((v229_data[13]) * v159_data);
              v222_acc += ((v229_data[14]) * v170_data);
              v222_acc += ((v229_data[15]) * v181_data);
              v222_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v262_acc{};
              int32_t v265_a = 0_i32 + 32;
              tensorforge::intel_esimd::simd<float, 16> v269_data;
              v269_data.copy_from(s0 + (32_i32));
              v262_acc += ((v269_data[0]) * v16_data);
              v262_acc += ((v269_data[1]) * v27_data);
              v262_acc += ((v269_data[2]) * v38_data);
              v262_acc += ((v269_data[3]) * v49_data);
              v262_acc += ((v269_data[4]) * v60_data);
              v262_acc += ((v269_data[5]) * v71_data);
              v262_acc += ((v269_data[6]) * v82_data);
              v262_acc += ((v269_data[7]) * v93_data);
              v262_acc += ((v269_data[8]) * v104_data);
              v262_acc += ((v269_data[9]) * v115_data);
              v262_acc += ((v269_data[10]) * v126_data);
              v262_acc += ((v269_data[11]) * v137_data);
              v262_acc += ((v269_data[12]) * v148_data);
              v262_acc += ((v269_data[13]) * v159_data);
              v262_acc += ((v269_data[14]) * v170_data);
              v262_acc += ((v269_data[15]) * v181_data);
              v262_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v302_acc{};
              int32_t v305_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v309_data;
              v309_data.copy_from(s0 + (48_i32));
              v302_acc += ((v309_data[0]) * v16_data);
              v302_acc += ((v309_data[1]) * v27_data);
              v302_acc += ((v309_data[2]) * v38_data);
              v302_acc += ((v309_data[3]) * v49_data);
              v302_acc += ((v309_data[4]) * v60_data);
              v302_acc += ((v309_data[5]) * v71_data);
              v302_acc += ((v309_data[6]) * v82_data);
              v302_acc += ((v309_data[7]) * v93_data);
              v302_acc += ((v309_data[8]) * v104_data);
              v302_acc += ((v309_data[9]) * v115_data);
              v302_acc += ((v309_data[10]) * v126_data);
              v302_acc += ((v309_data[11]) * v137_data);
              v302_acc += ((v309_data[12]) * v148_data);
              v302_acc += ((v309_data[13]) * v159_data);
              v302_acc += ((v309_data[14]) * v170_data);
              v302_acc += ((v309_data[15]) * v181_data);
              v302_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v342_acc{};
              int32_t v345_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 16> v349_data;
              v349_data.copy_from(s0 + (64_i32));
              v342_acc += ((v349_data[0]) * v16_data);
              v342_acc += ((v349_data[1]) * v27_data);
              v342_acc += ((v349_data[2]) * v38_data);
              v342_acc += ((v349_data[3]) * v49_data);
              v342_acc += ((v349_data[4]) * v60_data);
              v342_acc += ((v349_data[5]) * v71_data);
              v342_acc += ((v349_data[6]) * v82_data);
              v342_acc += ((v349_data[7]) * v93_data);
              v342_acc += ((v349_data[8]) * v104_data);
              v342_acc += ((v349_data[9]) * v115_data);
              v342_acc += ((v349_data[10]) * v126_data);
              v342_acc += ((v349_data[11]) * v137_data);
              v342_acc += ((v349_data[12]) * v148_data);
              v342_acc += ((v349_data[13]) * v159_data);
              v342_acc += ((v349_data[14]) * v170_data);
              v342_acc += ((v349_data[15]) * v181_data);
              v342_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v382_acc{};
              int32_t v385_a = 0_i32 + 80;
              tensorforge::intel_esimd::simd<float, 16> v389_data;
              v389_data.copy_from(s0 + (80_i32));
              v382_acc += ((v389_data[0]) * v16_data);
              v382_acc += ((v389_data[1]) * v27_data);
              v382_acc += ((v389_data[2]) * v38_data);
              v382_acc += ((v389_data[3]) * v49_data);
              v382_acc += ((v389_data[4]) * v60_data);
              v382_acc += ((v389_data[5]) * v71_data);
              v382_acc += ((v389_data[6]) * v82_data);
              v382_acc += ((v389_data[7]) * v93_data);
              v382_acc += ((v389_data[8]) * v104_data);
              v382_acc += ((v389_data[9]) * v115_data);
              v382_acc += ((v389_data[10]) * v126_data);
              v382_acc += ((v389_data[11]) * v137_data);
              v382_acc += ((v389_data[12]) * v148_data);
              v382_acc += ((v389_data[13]) * v159_data);
              v382_acc += ((v389_data[14]) * v170_data);
              v382_acc += ((v389_data[15]) * v181_data);
              v382_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v422_acc{};
              int32_t v425_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v429_data;
              v429_data.copy_from(s0 + (96_i32));
              v422_acc += ((v429_data[0]) * v16_data);
              v422_acc += ((v429_data[1]) * v27_data);
              v422_acc += ((v429_data[2]) * v38_data);
              v422_acc += ((v429_data[3]) * v49_data);
              v422_acc += ((v429_data[4]) * v60_data);
              v422_acc += ((v429_data[5]) * v71_data);
              v422_acc += ((v429_data[6]) * v82_data);
              v422_acc += ((v429_data[7]) * v93_data);
              v422_acc += ((v429_data[8]) * v104_data);
              v422_acc += ((v429_data[9]) * v115_data);
              v422_acc += ((v429_data[10]) * v126_data);
              v422_acc += ((v429_data[11]) * v137_data);
              v422_acc += ((v429_data[12]) * v148_data);
              v422_acc += ((v429_data[13]) * v159_data);
              v422_acc += ((v429_data[14]) * v170_data);
              v422_acc += ((v429_data[15]) * v181_data);
              v422_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v462_acc{};
              int32_t v465_a = 0_i32 + 112;
              tensorforge::intel_esimd::simd<float, 16> v469_data;
              v469_data.copy_from(s0 + (112_i32));
              v462_acc += ((v469_data[0]) * v16_data);
              v462_acc += ((v469_data[1]) * v27_data);
              v462_acc += ((v469_data[2]) * v38_data);
              v462_acc += ((v469_data[3]) * v49_data);
              v462_acc += ((v469_data[4]) * v60_data);
              v462_acc += ((v469_data[5]) * v71_data);
              v462_acc += ((v469_data[6]) * v82_data);
              v462_acc += ((v469_data[7]) * v93_data);
              v462_acc += ((v469_data[8]) * v104_data);
              v462_acc += ((v469_data[9]) * v115_data);
              v462_acc += ((v469_data[10]) * v126_data);
              v462_acc += ((v469_data[11]) * v137_data);
              v462_acc += ((v469_data[12]) * v148_data);
              v462_acc += ((v469_data[13]) * v159_data);
              v462_acc += ((v469_data[14]) * v170_data);
              v462_acc += ((v469_data[15]) * v181_data);
              v462_acc.copy_to(ir0 + (112));
              #pragma unroll
              for (int32_t v502_n1 = 0; v502_n1 < 8; ++v502_n1) {
                int32_t v503_a = v502_n1 * 16;
                int32_t v504_a = 0 + v503_a;
                tensorforge::intel_esimd::simd<float, 12> v507_data;
                v507_data.copy_from(ir0 + (v503_a));
                v507_data.copy_to(r0 + (v503_a));
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v510_i1 = 0; v510_i1 < 8; ++v510_i1) {
                int32_t v511_a = v510_i1 * 16;
                int32_t v512_a = 0 + v511_a;
                tensorforge::intel_esimd::simd<float, 12> v515_data;
                v515_data.copy_from(r0 + (v511_a));
                v515_data.copy_to(glb_m0 + ((v510_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

