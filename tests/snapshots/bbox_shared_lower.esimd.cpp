// === base name ===
kernel_b50aab1ea7e00df3

// === header ===
void launcher_kernel_b50aab1ea7e00df3(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_b50aab1ea7e00df3(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_b50aab1ea7e00df3(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_b50aab1ea7e00df3(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (2304, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 16×8(12×8) {4..16}×{0..8} strided
        // m1 16×16(12×16) {4..16}×{0..16} strided
        // m2 16×8(16×8) {0..16}×{0..8} strided
        // m0 16×8(12×8) {4..16}×{0..8} strided({4..16}×{0..8})[0, 1] = m1 16×16(12×16) {4..16}×{0..16} strided({4..16}×{0..16})[0, -1]×m2 16×8(16×8) {0..16}×{0..8} strided({0..16}×{0..8})[-1, 1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[144 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[128];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 96 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 192 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 128 + 0 + m2_extraOffset];
              float r0[256]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v15_i1 = 0; v15_i1 < 16; ++v15_i1) {
                tensorforge::intel_esimd::simd<float, 12> v22_data;
                v22_data.copy_from(glb_m1 + ((v15_i1 * 12)));
                v22_data.copy_to(r0 + ((v15_i1 * 16)));
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v25_ld;
              v25_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v25_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 64> v26_ld;
              v26_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              v26_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 64));
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(16, 28), (0, 8)] [(0, 16)]
              float ir1[128]{};
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v42_data;
              v42_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v44_data;
              v44_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v45_acc{};
              tensorforge::intel_esimd::simd<float, 16> v49_data;
              v49_data.copy_from(s0 + (0_i32));
              v45_acc += ((static_cast<float>(v49_data[0])) * v29_data);
              v45_acc += ((static_cast<float>(v49_data[1])) * v30_data);
              v45_acc += ((static_cast<float>(v49_data[2])) * v31_data);
              v45_acc += ((static_cast<float>(v49_data[3])) * v32_data);
              v45_acc += ((static_cast<float>(v49_data[4])) * v33_data);
              v45_acc += ((static_cast<float>(v49_data[5])) * v34_data);
              v45_acc += ((static_cast<float>(v49_data[6])) * v35_data);
              v45_acc += ((static_cast<float>(v49_data[7])) * v36_data);
              v45_acc += ((static_cast<float>(v49_data[8])) * v37_data);
              v45_acc += ((static_cast<float>(v49_data[9])) * v38_data);
              v45_acc += ((static_cast<float>(v49_data[10])) * v39_data);
              v45_acc += ((static_cast<float>(v49_data[11])) * v40_data);
              v45_acc += ((static_cast<float>(v49_data[12])) * v41_data);
              v45_acc += ((static_cast<float>(v49_data[13])) * v42_data);
              v45_acc += ((static_cast<float>(v49_data[14])) * v43_data);
              v45_acc += ((static_cast<float>(v49_data[15])) * v44_data);
              v45_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v82_acc{};
              tensorforge::intel_esimd::simd<float, 16> v86_data;
              v86_data.copy_from(s0 + (16_i32));
              v82_acc += ((static_cast<float>(v86_data[0])) * v29_data);
              v82_acc += ((static_cast<float>(v86_data[1])) * v30_data);
              v82_acc += ((static_cast<float>(v86_data[2])) * v31_data);
              v82_acc += ((static_cast<float>(v86_data[3])) * v32_data);
              v82_acc += ((static_cast<float>(v86_data[4])) * v33_data);
              v82_acc += ((static_cast<float>(v86_data[5])) * v34_data);
              v82_acc += ((static_cast<float>(v86_data[6])) * v35_data);
              v82_acc += ((static_cast<float>(v86_data[7])) * v36_data);
              v82_acc += ((static_cast<float>(v86_data[8])) * v37_data);
              v82_acc += ((static_cast<float>(v86_data[9])) * v38_data);
              v82_acc += ((static_cast<float>(v86_data[10])) * v39_data);
              v82_acc += ((static_cast<float>(v86_data[11])) * v40_data);
              v82_acc += ((static_cast<float>(v86_data[12])) * v41_data);
              v82_acc += ((static_cast<float>(v86_data[13])) * v42_data);
              v82_acc += ((static_cast<float>(v86_data[14])) * v43_data);
              v82_acc += ((static_cast<float>(v86_data[15])) * v44_data);
              v82_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v119_acc{};
              tensorforge::intel_esimd::simd<float, 16> v123_data;
              v123_data.copy_from(s0 + (32_i32));
              v119_acc += ((static_cast<float>(v123_data[0])) * v29_data);
              v119_acc += ((static_cast<float>(v123_data[1])) * v30_data);
              v119_acc += ((static_cast<float>(v123_data[2])) * v31_data);
              v119_acc += ((static_cast<float>(v123_data[3])) * v32_data);
              v119_acc += ((static_cast<float>(v123_data[4])) * v33_data);
              v119_acc += ((static_cast<float>(v123_data[5])) * v34_data);
              v119_acc += ((static_cast<float>(v123_data[6])) * v35_data);
              v119_acc += ((static_cast<float>(v123_data[7])) * v36_data);
              v119_acc += ((static_cast<float>(v123_data[8])) * v37_data);
              v119_acc += ((static_cast<float>(v123_data[9])) * v38_data);
              v119_acc += ((static_cast<float>(v123_data[10])) * v39_data);
              v119_acc += ((static_cast<float>(v123_data[11])) * v40_data);
              v119_acc += ((static_cast<float>(v123_data[12])) * v41_data);
              v119_acc += ((static_cast<float>(v123_data[13])) * v42_data);
              v119_acc += ((static_cast<float>(v123_data[14])) * v43_data);
              v119_acc += ((static_cast<float>(v123_data[15])) * v44_data);
              v119_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v156_acc{};
              tensorforge::intel_esimd::simd<float, 16> v160_data;
              v160_data.copy_from(s0 + (48_i32));
              v156_acc += ((static_cast<float>(v160_data[0])) * v29_data);
              v156_acc += ((static_cast<float>(v160_data[1])) * v30_data);
              v156_acc += ((static_cast<float>(v160_data[2])) * v31_data);
              v156_acc += ((static_cast<float>(v160_data[3])) * v32_data);
              v156_acc += ((static_cast<float>(v160_data[4])) * v33_data);
              v156_acc += ((static_cast<float>(v160_data[5])) * v34_data);
              v156_acc += ((static_cast<float>(v160_data[6])) * v35_data);
              v156_acc += ((static_cast<float>(v160_data[7])) * v36_data);
              v156_acc += ((static_cast<float>(v160_data[8])) * v37_data);
              v156_acc += ((static_cast<float>(v160_data[9])) * v38_data);
              v156_acc += ((static_cast<float>(v160_data[10])) * v39_data);
              v156_acc += ((static_cast<float>(v160_data[11])) * v40_data);
              v156_acc += ((static_cast<float>(v160_data[12])) * v41_data);
              v156_acc += ((static_cast<float>(v160_data[13])) * v42_data);
              v156_acc += ((static_cast<float>(v160_data[14])) * v43_data);
              v156_acc += ((static_cast<float>(v160_data[15])) * v44_data);
              v156_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v193_acc{};
              tensorforge::intel_esimd::simd<float, 16> v197_data;
              v197_data.copy_from(s0 + (64_i32));
              v193_acc += ((static_cast<float>(v197_data[0])) * v29_data);
              v193_acc += ((static_cast<float>(v197_data[1])) * v30_data);
              v193_acc += ((static_cast<float>(v197_data[2])) * v31_data);
              v193_acc += ((static_cast<float>(v197_data[3])) * v32_data);
              v193_acc += ((static_cast<float>(v197_data[4])) * v33_data);
              v193_acc += ((static_cast<float>(v197_data[5])) * v34_data);
              v193_acc += ((static_cast<float>(v197_data[6])) * v35_data);
              v193_acc += ((static_cast<float>(v197_data[7])) * v36_data);
              v193_acc += ((static_cast<float>(v197_data[8])) * v37_data);
              v193_acc += ((static_cast<float>(v197_data[9])) * v38_data);
              v193_acc += ((static_cast<float>(v197_data[10])) * v39_data);
              v193_acc += ((static_cast<float>(v197_data[11])) * v40_data);
              v193_acc += ((static_cast<float>(v197_data[12])) * v41_data);
              v193_acc += ((static_cast<float>(v197_data[13])) * v42_data);
              v193_acc += ((static_cast<float>(v197_data[14])) * v43_data);
              v193_acc += ((static_cast<float>(v197_data[15])) * v44_data);
              v193_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v230_acc{};
              tensorforge::intel_esimd::simd<float, 16> v234_data;
              v234_data.copy_from(s0 + (80_i32));
              v230_acc += ((static_cast<float>(v234_data[0])) * v29_data);
              v230_acc += ((static_cast<float>(v234_data[1])) * v30_data);
              v230_acc += ((static_cast<float>(v234_data[2])) * v31_data);
              v230_acc += ((static_cast<float>(v234_data[3])) * v32_data);
              v230_acc += ((static_cast<float>(v234_data[4])) * v33_data);
              v230_acc += ((static_cast<float>(v234_data[5])) * v34_data);
              v230_acc += ((static_cast<float>(v234_data[6])) * v35_data);
              v230_acc += ((static_cast<float>(v234_data[7])) * v36_data);
              v230_acc += ((static_cast<float>(v234_data[8])) * v37_data);
              v230_acc += ((static_cast<float>(v234_data[9])) * v38_data);
              v230_acc += ((static_cast<float>(v234_data[10])) * v39_data);
              v230_acc += ((static_cast<float>(v234_data[11])) * v40_data);
              v230_acc += ((static_cast<float>(v234_data[12])) * v41_data);
              v230_acc += ((static_cast<float>(v234_data[13])) * v42_data);
              v230_acc += ((static_cast<float>(v234_data[14])) * v43_data);
              v230_acc += ((static_cast<float>(v234_data[15])) * v44_data);
              v230_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v267_acc{};
              tensorforge::intel_esimd::simd<float, 16> v271_data;
              v271_data.copy_from(s0 + (96_i32));
              v267_acc += ((static_cast<float>(v271_data[0])) * v29_data);
              v267_acc += ((static_cast<float>(v271_data[1])) * v30_data);
              v267_acc += ((static_cast<float>(v271_data[2])) * v31_data);
              v267_acc += ((static_cast<float>(v271_data[3])) * v32_data);
              v267_acc += ((static_cast<float>(v271_data[4])) * v33_data);
              v267_acc += ((static_cast<float>(v271_data[5])) * v34_data);
              v267_acc += ((static_cast<float>(v271_data[6])) * v35_data);
              v267_acc += ((static_cast<float>(v271_data[7])) * v36_data);
              v267_acc += ((static_cast<float>(v271_data[8])) * v37_data);
              v267_acc += ((static_cast<float>(v271_data[9])) * v38_data);
              v267_acc += ((static_cast<float>(v271_data[10])) * v39_data);
              v267_acc += ((static_cast<float>(v271_data[11])) * v40_data);
              v267_acc += ((static_cast<float>(v271_data[12])) * v41_data);
              v267_acc += ((static_cast<float>(v271_data[13])) * v42_data);
              v267_acc += ((static_cast<float>(v271_data[14])) * v43_data);
              v267_acc += ((static_cast<float>(v271_data[15])) * v44_data);
              v267_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v304_acc{};
              tensorforge::intel_esimd::simd<float, 16> v308_data;
              v308_data.copy_from(s0 + (112_i32));
              v304_acc += ((static_cast<float>(v308_data[0])) * v29_data);
              v304_acc += ((static_cast<float>(v308_data[1])) * v30_data);
              v304_acc += ((static_cast<float>(v308_data[2])) * v31_data);
              v304_acc += ((static_cast<float>(v308_data[3])) * v32_data);
              v304_acc += ((static_cast<float>(v308_data[4])) * v33_data);
              v304_acc += ((static_cast<float>(v308_data[5])) * v34_data);
              v304_acc += ((static_cast<float>(v308_data[6])) * v35_data);
              v304_acc += ((static_cast<float>(v308_data[7])) * v36_data);
              v304_acc += ((static_cast<float>(v308_data[8])) * v37_data);
              v304_acc += ((static_cast<float>(v308_data[9])) * v38_data);
              v304_acc += ((static_cast<float>(v308_data[10])) * v39_data);
              v304_acc += ((static_cast<float>(v308_data[11])) * v40_data);
              v304_acc += ((static_cast<float>(v308_data[12])) * v41_data);
              v304_acc += ((static_cast<float>(v308_data[13])) * v42_data);
              v304_acc += ((static_cast<float>(v308_data[14])) * v43_data);
              v304_acc += ((static_cast<float>(v308_data[15])) * v44_data);
              v304_acc.copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v341_n1 = 0; v341_n1 < 8; ++v341_n1) {
                int32_t v342_a = v341_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v344_data;
                v344_data.copy_from(ir1 + (v342_a));
                v344_data.copy_to(r1 + (v342_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v347_i1 = 0; v347_i1 < 8; ++v347_i1) {
                tensorforge::intel_esimd::simd<float, 12> v350_data;
                v350_data.copy_from(r1 + ((v347_i1 * 16)));
                v350_data.copy_to(glb_m0 + ((v347_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

