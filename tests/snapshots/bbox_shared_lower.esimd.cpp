// === base name ===
kernel_9d61a6b1dafe4409

// === header ===
void launcher_kernel_9d61a6b1dafe4409(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_9d61a6b1dafe4409(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_9d61a6b1dafe4409(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_9d61a6b1dafe4409(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
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
          float* __restrict__ s0 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 128 + 0 + m2_extraOffset];
              float r0[256]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v11_i1 = 0; v11_i1 < 16; ++v11_i1) {
                tensorforge::intel_esimd::simd<float, 12> v18_data;
                v18_data.copy_from(glb_m1 + ((v11_i1 * 12)));
                v18_data.copy_to(r0 + ((v11_i1 * 16)));
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v21_ld;
              v21_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v21_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 64> v22_ld;
              v22_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              v22_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(16, 28), (0, 8)] [(0, 16)]
              float ir1[128]{};
              tensorforge::intel_esimd::simd<float, 16> v25_data;
              v25_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v26_data;
              v26_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v41_acc{};
              tensorforge::intel_esimd::simd<float, 16> v45_data;
              v45_data.copy_from(s0 + (0_i32));
              v41_acc += ((v45_data[0]) * v25_data);
              v41_acc += ((v45_data[1]) * v26_data);
              v41_acc += ((v45_data[2]) * v27_data);
              v41_acc += ((v45_data[3]) * v28_data);
              v41_acc += ((v45_data[4]) * v29_data);
              v41_acc += ((v45_data[5]) * v30_data);
              v41_acc += ((v45_data[6]) * v31_data);
              v41_acc += ((v45_data[7]) * v32_data);
              v41_acc += ((v45_data[8]) * v33_data);
              v41_acc += ((v45_data[9]) * v34_data);
              v41_acc += ((v45_data[10]) * v35_data);
              v41_acc += ((v45_data[11]) * v36_data);
              v41_acc += ((v45_data[12]) * v37_data);
              v41_acc += ((v45_data[13]) * v38_data);
              v41_acc += ((v45_data[14]) * v39_data);
              v41_acc += ((v45_data[15]) * v40_data);
              v41_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v78_acc{};
              tensorforge::intel_esimd::simd<float, 16> v82_data;
              v82_data.copy_from(s0 + (16_i32));
              v78_acc += ((v82_data[0]) * v25_data);
              v78_acc += ((v82_data[1]) * v26_data);
              v78_acc += ((v82_data[2]) * v27_data);
              v78_acc += ((v82_data[3]) * v28_data);
              v78_acc += ((v82_data[4]) * v29_data);
              v78_acc += ((v82_data[5]) * v30_data);
              v78_acc += ((v82_data[6]) * v31_data);
              v78_acc += ((v82_data[7]) * v32_data);
              v78_acc += ((v82_data[8]) * v33_data);
              v78_acc += ((v82_data[9]) * v34_data);
              v78_acc += ((v82_data[10]) * v35_data);
              v78_acc += ((v82_data[11]) * v36_data);
              v78_acc += ((v82_data[12]) * v37_data);
              v78_acc += ((v82_data[13]) * v38_data);
              v78_acc += ((v82_data[14]) * v39_data);
              v78_acc += ((v82_data[15]) * v40_data);
              v78_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v115_acc{};
              tensorforge::intel_esimd::simd<float, 16> v119_data;
              v119_data.copy_from(s0 + (32_i32));
              v115_acc += ((v119_data[0]) * v25_data);
              v115_acc += ((v119_data[1]) * v26_data);
              v115_acc += ((v119_data[2]) * v27_data);
              v115_acc += ((v119_data[3]) * v28_data);
              v115_acc += ((v119_data[4]) * v29_data);
              v115_acc += ((v119_data[5]) * v30_data);
              v115_acc += ((v119_data[6]) * v31_data);
              v115_acc += ((v119_data[7]) * v32_data);
              v115_acc += ((v119_data[8]) * v33_data);
              v115_acc += ((v119_data[9]) * v34_data);
              v115_acc += ((v119_data[10]) * v35_data);
              v115_acc += ((v119_data[11]) * v36_data);
              v115_acc += ((v119_data[12]) * v37_data);
              v115_acc += ((v119_data[13]) * v38_data);
              v115_acc += ((v119_data[14]) * v39_data);
              v115_acc += ((v119_data[15]) * v40_data);
              v115_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v152_acc{};
              tensorforge::intel_esimd::simd<float, 16> v156_data;
              v156_data.copy_from(s0 + (48_i32));
              v152_acc += ((v156_data[0]) * v25_data);
              v152_acc += ((v156_data[1]) * v26_data);
              v152_acc += ((v156_data[2]) * v27_data);
              v152_acc += ((v156_data[3]) * v28_data);
              v152_acc += ((v156_data[4]) * v29_data);
              v152_acc += ((v156_data[5]) * v30_data);
              v152_acc += ((v156_data[6]) * v31_data);
              v152_acc += ((v156_data[7]) * v32_data);
              v152_acc += ((v156_data[8]) * v33_data);
              v152_acc += ((v156_data[9]) * v34_data);
              v152_acc += ((v156_data[10]) * v35_data);
              v152_acc += ((v156_data[11]) * v36_data);
              v152_acc += ((v156_data[12]) * v37_data);
              v152_acc += ((v156_data[13]) * v38_data);
              v152_acc += ((v156_data[14]) * v39_data);
              v152_acc += ((v156_data[15]) * v40_data);
              v152_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v189_acc{};
              tensorforge::intel_esimd::simd<float, 16> v193_data;
              v193_data.copy_from(s0 + (64_i32));
              v189_acc += ((v193_data[0]) * v25_data);
              v189_acc += ((v193_data[1]) * v26_data);
              v189_acc += ((v193_data[2]) * v27_data);
              v189_acc += ((v193_data[3]) * v28_data);
              v189_acc += ((v193_data[4]) * v29_data);
              v189_acc += ((v193_data[5]) * v30_data);
              v189_acc += ((v193_data[6]) * v31_data);
              v189_acc += ((v193_data[7]) * v32_data);
              v189_acc += ((v193_data[8]) * v33_data);
              v189_acc += ((v193_data[9]) * v34_data);
              v189_acc += ((v193_data[10]) * v35_data);
              v189_acc += ((v193_data[11]) * v36_data);
              v189_acc += ((v193_data[12]) * v37_data);
              v189_acc += ((v193_data[13]) * v38_data);
              v189_acc += ((v193_data[14]) * v39_data);
              v189_acc += ((v193_data[15]) * v40_data);
              v189_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v226_acc{};
              tensorforge::intel_esimd::simd<float, 16> v230_data;
              v230_data.copy_from(s0 + (80_i32));
              v226_acc += ((v230_data[0]) * v25_data);
              v226_acc += ((v230_data[1]) * v26_data);
              v226_acc += ((v230_data[2]) * v27_data);
              v226_acc += ((v230_data[3]) * v28_data);
              v226_acc += ((v230_data[4]) * v29_data);
              v226_acc += ((v230_data[5]) * v30_data);
              v226_acc += ((v230_data[6]) * v31_data);
              v226_acc += ((v230_data[7]) * v32_data);
              v226_acc += ((v230_data[8]) * v33_data);
              v226_acc += ((v230_data[9]) * v34_data);
              v226_acc += ((v230_data[10]) * v35_data);
              v226_acc += ((v230_data[11]) * v36_data);
              v226_acc += ((v230_data[12]) * v37_data);
              v226_acc += ((v230_data[13]) * v38_data);
              v226_acc += ((v230_data[14]) * v39_data);
              v226_acc += ((v230_data[15]) * v40_data);
              v226_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v263_acc{};
              tensorforge::intel_esimd::simd<float, 16> v267_data;
              v267_data.copy_from(s0 + (96_i32));
              v263_acc += ((v267_data[0]) * v25_data);
              v263_acc += ((v267_data[1]) * v26_data);
              v263_acc += ((v267_data[2]) * v27_data);
              v263_acc += ((v267_data[3]) * v28_data);
              v263_acc += ((v267_data[4]) * v29_data);
              v263_acc += ((v267_data[5]) * v30_data);
              v263_acc += ((v267_data[6]) * v31_data);
              v263_acc += ((v267_data[7]) * v32_data);
              v263_acc += ((v267_data[8]) * v33_data);
              v263_acc += ((v267_data[9]) * v34_data);
              v263_acc += ((v267_data[10]) * v35_data);
              v263_acc += ((v267_data[11]) * v36_data);
              v263_acc += ((v267_data[12]) * v37_data);
              v263_acc += ((v267_data[13]) * v38_data);
              v263_acc += ((v267_data[14]) * v39_data);
              v263_acc += ((v267_data[15]) * v40_data);
              v263_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v300_acc{};
              tensorforge::intel_esimd::simd<float, 16> v304_data;
              v304_data.copy_from(s0 + (112_i32));
              v300_acc += ((v304_data[0]) * v25_data);
              v300_acc += ((v304_data[1]) * v26_data);
              v300_acc += ((v304_data[2]) * v27_data);
              v300_acc += ((v304_data[3]) * v28_data);
              v300_acc += ((v304_data[4]) * v29_data);
              v300_acc += ((v304_data[5]) * v30_data);
              v300_acc += ((v304_data[6]) * v31_data);
              v300_acc += ((v304_data[7]) * v32_data);
              v300_acc += ((v304_data[8]) * v33_data);
              v300_acc += ((v304_data[9]) * v34_data);
              v300_acc += ((v304_data[10]) * v35_data);
              v300_acc += ((v304_data[11]) * v36_data);
              v300_acc += ((v304_data[12]) * v37_data);
              v300_acc += ((v304_data[13]) * v38_data);
              v300_acc += ((v304_data[14]) * v39_data);
              v300_acc += ((v304_data[15]) * v40_data);
              v300_acc.copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v337_n1 = 0; v337_n1 < 8; ++v337_n1) {
                int32_t v338_a = v337_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v340_data;
                v340_data.copy_from(ir1 + (v338_a));
                v340_data.copy_to(r1 + (v338_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v343_i1 = 0; v343_i1 < 8; ++v343_i1) {
                tensorforge::intel_esimd::simd<float, 12> v346_data;
                v346_data.copy_from(r1 + ((v343_i1 * 16)));
                v346_data.copy_to(glb_m0 + ((v343_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

