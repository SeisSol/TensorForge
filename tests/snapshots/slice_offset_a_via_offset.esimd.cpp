// === base name ===
kernel_ead773dd51

// === header ===
void launcher_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_ead773dd51(float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid ((numElements0 + 16 - 1) / 16, 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_ead773dd51(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_ead773dd51(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, unsigned m0_extraOffset, const float* m1, unsigned m1_extraOffset, const float* m2, unsigned m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // meta data:
        // m0 12×8(12×8) {0..12}×{0..8} strided
        // m1 32×16(32×16) {0..32}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[0, 1] = m1 32×16(32×16) {0..32}×{0..16} strided({0..12}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
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
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 512 + 0 + m1_extraOffset];
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
              int32_t v9_a = 4_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v14_data;
              v14_data.copy_from(glb_m1 + (4_i32));
              int32_t v18_a = 4_i32 + 32;
              tensorforge::intel_esimd::simd<float, 16> v23_data;
              v23_data.copy_from(glb_m1 + (36_i32));
              int32_t v27_a = 4_i32 + 64;
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(glb_m1 + (68_i32));
              int32_t v36_a = 4_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(glb_m1 + (100_i32));
              int32_t v45_a = 4_i32 + 128;
              tensorforge::intel_esimd::simd<float, 16> v50_data;
              v50_data.copy_from(glb_m1 + (132_i32));
              int32_t v54_a = 4_i32 + 160;
              tensorforge::intel_esimd::simd<float, 16> v59_data;
              v59_data.copy_from(glb_m1 + (164_i32));
              int32_t v63_a = 4_i32 + 192;
              tensorforge::intel_esimd::simd<float, 16> v68_data;
              v68_data.copy_from(glb_m1 + (196_i32));
              int32_t v72_a = 4_i32 + 224;
              tensorforge::intel_esimd::simd<float, 16> v77_data;
              v77_data.copy_from(glb_m1 + (228_i32));
              int32_t v81_a = 4_i32 + 256;
              tensorforge::intel_esimd::simd<float, 16> v86_data;
              v86_data.copy_from(glb_m1 + (260_i32));
              int32_t v90_a = 4_i32 + 288;
              tensorforge::intel_esimd::simd<float, 16> v95_data;
              v95_data.copy_from(glb_m1 + (292_i32));
              int32_t v99_a = 4_i32 + 320;
              tensorforge::intel_esimd::simd<float, 16> v104_data;
              v104_data.copy_from(glb_m1 + (324_i32));
              int32_t v108_a = 4_i32 + 352;
              tensorforge::intel_esimd::simd<float, 16> v113_data;
              v113_data.copy_from(glb_m1 + (356_i32));
              int32_t v117_a = 4_i32 + 384;
              tensorforge::intel_esimd::simd<float, 16> v122_data;
              v122_data.copy_from(glb_m1 + (388_i32));
              int32_t v126_a = 4_i32 + 416;
              tensorforge::intel_esimd::simd<float, 16> v131_data;
              v131_data.copy_from(glb_m1 + (420_i32));
              int32_t v135_a = 4_i32 + 448;
              tensorforge::intel_esimd::simd<float, 16> v140_data;
              v140_data.copy_from(glb_m1 + (452_i32));
              int32_t v144_a = 4_i32 + 480;
              tensorforge::intel_esimd::simd<float, 16> v149_data;
              v149_data.copy_from(glb_m1 + (484_i32));
              tensorforge::intel_esimd::simd<float, 16> v150_acc{};
              int32_t v153_a = 0_i32 + 0;
              tensorforge::intel_esimd::simd<float, 16> v157_data;
              v157_data.copy_from(s0 + (0_i32));
              v150_acc += ((v157_data[0]) * v14_data);
              v150_acc += ((v157_data[1]) * v23_data);
              v150_acc += ((v157_data[2]) * v32_data);
              v150_acc += ((v157_data[3]) * v41_data);
              v150_acc += ((v157_data[4]) * v50_data);
              v150_acc += ((v157_data[5]) * v59_data);
              v150_acc += ((v157_data[6]) * v68_data);
              v150_acc += ((v157_data[7]) * v77_data);
              v150_acc += ((v157_data[8]) * v86_data);
              v150_acc += ((v157_data[9]) * v95_data);
              v150_acc += ((v157_data[10]) * v104_data);
              v150_acc += ((v157_data[11]) * v113_data);
              v150_acc += ((v157_data[12]) * v122_data);
              v150_acc += ((v157_data[13]) * v131_data);
              v150_acc += ((v157_data[14]) * v140_data);
              v150_acc += ((v157_data[15]) * v149_data);
              v150_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v190_acc{};
              int32_t v193_a = 0_i32 + 16;
              tensorforge::intel_esimd::simd<float, 16> v197_data;
              v197_data.copy_from(s0 + (16_i32));
              v190_acc += ((v197_data[0]) * v14_data);
              v190_acc += ((v197_data[1]) * v23_data);
              v190_acc += ((v197_data[2]) * v32_data);
              v190_acc += ((v197_data[3]) * v41_data);
              v190_acc += ((v197_data[4]) * v50_data);
              v190_acc += ((v197_data[5]) * v59_data);
              v190_acc += ((v197_data[6]) * v68_data);
              v190_acc += ((v197_data[7]) * v77_data);
              v190_acc += ((v197_data[8]) * v86_data);
              v190_acc += ((v197_data[9]) * v95_data);
              v190_acc += ((v197_data[10]) * v104_data);
              v190_acc += ((v197_data[11]) * v113_data);
              v190_acc += ((v197_data[12]) * v122_data);
              v190_acc += ((v197_data[13]) * v131_data);
              v190_acc += ((v197_data[14]) * v140_data);
              v190_acc += ((v197_data[15]) * v149_data);
              v190_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v230_acc{};
              int32_t v233_a = 0_i32 + 32;
              tensorforge::intel_esimd::simd<float, 16> v237_data;
              v237_data.copy_from(s0 + (32_i32));
              v230_acc += ((v237_data[0]) * v14_data);
              v230_acc += ((v237_data[1]) * v23_data);
              v230_acc += ((v237_data[2]) * v32_data);
              v230_acc += ((v237_data[3]) * v41_data);
              v230_acc += ((v237_data[4]) * v50_data);
              v230_acc += ((v237_data[5]) * v59_data);
              v230_acc += ((v237_data[6]) * v68_data);
              v230_acc += ((v237_data[7]) * v77_data);
              v230_acc += ((v237_data[8]) * v86_data);
              v230_acc += ((v237_data[9]) * v95_data);
              v230_acc += ((v237_data[10]) * v104_data);
              v230_acc += ((v237_data[11]) * v113_data);
              v230_acc += ((v237_data[12]) * v122_data);
              v230_acc += ((v237_data[13]) * v131_data);
              v230_acc += ((v237_data[14]) * v140_data);
              v230_acc += ((v237_data[15]) * v149_data);
              v230_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v270_acc{};
              int32_t v273_a = 0_i32 + 48;
              tensorforge::intel_esimd::simd<float, 16> v277_data;
              v277_data.copy_from(s0 + (48_i32));
              v270_acc += ((v277_data[0]) * v14_data);
              v270_acc += ((v277_data[1]) * v23_data);
              v270_acc += ((v277_data[2]) * v32_data);
              v270_acc += ((v277_data[3]) * v41_data);
              v270_acc += ((v277_data[4]) * v50_data);
              v270_acc += ((v277_data[5]) * v59_data);
              v270_acc += ((v277_data[6]) * v68_data);
              v270_acc += ((v277_data[7]) * v77_data);
              v270_acc += ((v277_data[8]) * v86_data);
              v270_acc += ((v277_data[9]) * v95_data);
              v270_acc += ((v277_data[10]) * v104_data);
              v270_acc += ((v277_data[11]) * v113_data);
              v270_acc += ((v277_data[12]) * v122_data);
              v270_acc += ((v277_data[13]) * v131_data);
              v270_acc += ((v277_data[14]) * v140_data);
              v270_acc += ((v277_data[15]) * v149_data);
              v270_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v310_acc{};
              int32_t v313_a = 0_i32 + 64;
              tensorforge::intel_esimd::simd<float, 16> v317_data;
              v317_data.copy_from(s0 + (64_i32));
              v310_acc += ((v317_data[0]) * v14_data);
              v310_acc += ((v317_data[1]) * v23_data);
              v310_acc += ((v317_data[2]) * v32_data);
              v310_acc += ((v317_data[3]) * v41_data);
              v310_acc += ((v317_data[4]) * v50_data);
              v310_acc += ((v317_data[5]) * v59_data);
              v310_acc += ((v317_data[6]) * v68_data);
              v310_acc += ((v317_data[7]) * v77_data);
              v310_acc += ((v317_data[8]) * v86_data);
              v310_acc += ((v317_data[9]) * v95_data);
              v310_acc += ((v317_data[10]) * v104_data);
              v310_acc += ((v317_data[11]) * v113_data);
              v310_acc += ((v317_data[12]) * v122_data);
              v310_acc += ((v317_data[13]) * v131_data);
              v310_acc += ((v317_data[14]) * v140_data);
              v310_acc += ((v317_data[15]) * v149_data);
              v310_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v350_acc{};
              int32_t v353_a = 0_i32 + 80;
              tensorforge::intel_esimd::simd<float, 16> v357_data;
              v357_data.copy_from(s0 + (80_i32));
              v350_acc += ((v357_data[0]) * v14_data);
              v350_acc += ((v357_data[1]) * v23_data);
              v350_acc += ((v357_data[2]) * v32_data);
              v350_acc += ((v357_data[3]) * v41_data);
              v350_acc += ((v357_data[4]) * v50_data);
              v350_acc += ((v357_data[5]) * v59_data);
              v350_acc += ((v357_data[6]) * v68_data);
              v350_acc += ((v357_data[7]) * v77_data);
              v350_acc += ((v357_data[8]) * v86_data);
              v350_acc += ((v357_data[9]) * v95_data);
              v350_acc += ((v357_data[10]) * v104_data);
              v350_acc += ((v357_data[11]) * v113_data);
              v350_acc += ((v357_data[12]) * v122_data);
              v350_acc += ((v357_data[13]) * v131_data);
              v350_acc += ((v357_data[14]) * v140_data);
              v350_acc += ((v357_data[15]) * v149_data);
              v350_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v390_acc{};
              int32_t v393_a = 0_i32 + 96;
              tensorforge::intel_esimd::simd<float, 16> v397_data;
              v397_data.copy_from(s0 + (96_i32));
              v390_acc += ((v397_data[0]) * v14_data);
              v390_acc += ((v397_data[1]) * v23_data);
              v390_acc += ((v397_data[2]) * v32_data);
              v390_acc += ((v397_data[3]) * v41_data);
              v390_acc += ((v397_data[4]) * v50_data);
              v390_acc += ((v397_data[5]) * v59_data);
              v390_acc += ((v397_data[6]) * v68_data);
              v390_acc += ((v397_data[7]) * v77_data);
              v390_acc += ((v397_data[8]) * v86_data);
              v390_acc += ((v397_data[9]) * v95_data);
              v390_acc += ((v397_data[10]) * v104_data);
              v390_acc += ((v397_data[11]) * v113_data);
              v390_acc += ((v397_data[12]) * v122_data);
              v390_acc += ((v397_data[13]) * v131_data);
              v390_acc += ((v397_data[14]) * v140_data);
              v390_acc += ((v397_data[15]) * v149_data);
              v390_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v430_acc{};
              int32_t v433_a = 0_i32 + 112;
              tensorforge::intel_esimd::simd<float, 16> v437_data;
              v437_data.copy_from(s0 + (112_i32));
              v430_acc += ((v437_data[0]) * v14_data);
              v430_acc += ((v437_data[1]) * v23_data);
              v430_acc += ((v437_data[2]) * v32_data);
              v430_acc += ((v437_data[3]) * v41_data);
              v430_acc += ((v437_data[4]) * v50_data);
              v430_acc += ((v437_data[5]) * v59_data);
              v430_acc += ((v437_data[6]) * v68_data);
              v430_acc += ((v437_data[7]) * v77_data);
              v430_acc += ((v437_data[8]) * v86_data);
              v430_acc += ((v437_data[9]) * v95_data);
              v430_acc += ((v437_data[10]) * v104_data);
              v430_acc += ((v437_data[11]) * v113_data);
              v430_acc += ((v437_data[12]) * v122_data);
              v430_acc += ((v437_data[13]) * v131_data);
              v430_acc += ((v437_data[14]) * v140_data);
              v430_acc += ((v437_data[15]) * v149_data);
              v430_acc.copy_to(ir0 + (112));
              #pragma unroll
              for (int32_t v470_n1 = 0; v470_n1 < 8; ++v470_n1) {
                int32_t v471_a = v470_n1 * 16;
                int32_t v472_a = 0 + v471_a;
                tensorforge::intel_esimd::simd<float, 12> v475_data;
                v475_data.copy_from(ir0 + (v471_a));
                v475_data.copy_to(r0 + (v471_a));
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v478_i1 = 0; v478_i1 < 8; ++v478_i1) {
                int32_t v479_a = v478_i1 * 16;
                int32_t v480_a = 0 + v479_a;
                tensorforge::intel_esimd::simd<float, 12> v483_data;
                v483_data.copy_from(r0 + (v479_a));
                v483_data.copy_to(glb_m0 + ((v478_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

