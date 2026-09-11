// === base name ===
kernel_4175b157f0b7cb6a

// === header ===
void launcher_kernel_4175b157f0b7cb6a(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_4175b157f0b7cb6a(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_4175b157f0b7cb6a(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4175b157f0b7cb6a(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8×8(8×8) {0..8}×{0..8} strided
        // m3 8×8(8×8) {0..8}×{0..8} strided
        // m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, 1] = m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // C = abs(M)
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[80 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 64 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v3_batchId0 * 64 + 0 + m3_extraOffset];
              float r0[128]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v16_i1 = 0; v16_i1 < 8; ++v16_i1) {
                tensorforge::intel_esimd::simd<float, 8> v21_data;
                v21_data.copy_from(glb_m1 + ((v16_i1 * 8)));
                v21_data.copy_to(r0 + ((v16_i1 * 16)));
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v24_ld;
              v24_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v24_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              float ir1[128]{};
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v35_acc{};
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(s0 + (0_i32));
              v35_acc += ((static_cast<float>(v39_data[0])) * v27_data);
              v35_acc += ((static_cast<float>(v39_data[1])) * v28_data);
              v35_acc += ((static_cast<float>(v39_data[2])) * v29_data);
              v35_acc += ((static_cast<float>(v39_data[3])) * v30_data);
              v35_acc += ((static_cast<float>(v39_data[4])) * v31_data);
              v35_acc += ((static_cast<float>(v39_data[5])) * v32_data);
              v35_acc += ((static_cast<float>(v39_data[6])) * v33_data);
              v35_acc += ((static_cast<float>(v39_data[7])) * v34_data);
              v35_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v56_acc{};
              tensorforge::intel_esimd::simd<float, 16> v60_data;
              v60_data.copy_from(s0 + (8_i32));
              v56_acc += ((static_cast<float>(v60_data[0])) * v27_data);
              v56_acc += ((static_cast<float>(v60_data[1])) * v28_data);
              v56_acc += ((static_cast<float>(v60_data[2])) * v29_data);
              v56_acc += ((static_cast<float>(v60_data[3])) * v30_data);
              v56_acc += ((static_cast<float>(v60_data[4])) * v31_data);
              v56_acc += ((static_cast<float>(v60_data[5])) * v32_data);
              v56_acc += ((static_cast<float>(v60_data[6])) * v33_data);
              v56_acc += ((static_cast<float>(v60_data[7])) * v34_data);
              v56_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v77_acc{};
              tensorforge::intel_esimd::simd<float, 16> v81_data;
              v81_data.copy_from(s0 + (16_i32));
              v77_acc += ((static_cast<float>(v81_data[0])) * v27_data);
              v77_acc += ((static_cast<float>(v81_data[1])) * v28_data);
              v77_acc += ((static_cast<float>(v81_data[2])) * v29_data);
              v77_acc += ((static_cast<float>(v81_data[3])) * v30_data);
              v77_acc += ((static_cast<float>(v81_data[4])) * v31_data);
              v77_acc += ((static_cast<float>(v81_data[5])) * v32_data);
              v77_acc += ((static_cast<float>(v81_data[6])) * v33_data);
              v77_acc += ((static_cast<float>(v81_data[7])) * v34_data);
              v77_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v98_acc{};
              tensorforge::intel_esimd::simd<float, 16> v102_data;
              v102_data.copy_from(s0 + (24_i32));
              v98_acc += ((static_cast<float>(v102_data[0])) * v27_data);
              v98_acc += ((static_cast<float>(v102_data[1])) * v28_data);
              v98_acc += ((static_cast<float>(v102_data[2])) * v29_data);
              v98_acc += ((static_cast<float>(v102_data[3])) * v30_data);
              v98_acc += ((static_cast<float>(v102_data[4])) * v31_data);
              v98_acc += ((static_cast<float>(v102_data[5])) * v32_data);
              v98_acc += ((static_cast<float>(v102_data[6])) * v33_data);
              v98_acc += ((static_cast<float>(v102_data[7])) * v34_data);
              v98_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v119_acc{};
              tensorforge::intel_esimd::simd<float, 16> v123_data;
              v123_data.copy_from(s0 + (32_i32));
              v119_acc += ((static_cast<float>(v123_data[0])) * v27_data);
              v119_acc += ((static_cast<float>(v123_data[1])) * v28_data);
              v119_acc += ((static_cast<float>(v123_data[2])) * v29_data);
              v119_acc += ((static_cast<float>(v123_data[3])) * v30_data);
              v119_acc += ((static_cast<float>(v123_data[4])) * v31_data);
              v119_acc += ((static_cast<float>(v123_data[5])) * v32_data);
              v119_acc += ((static_cast<float>(v123_data[6])) * v33_data);
              v119_acc += ((static_cast<float>(v123_data[7])) * v34_data);
              v119_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v140_acc{};
              tensorforge::intel_esimd::simd<float, 16> v144_data;
              v144_data.copy_from(s0 + (40_i32));
              v140_acc += ((static_cast<float>(v144_data[0])) * v27_data);
              v140_acc += ((static_cast<float>(v144_data[1])) * v28_data);
              v140_acc += ((static_cast<float>(v144_data[2])) * v29_data);
              v140_acc += ((static_cast<float>(v144_data[3])) * v30_data);
              v140_acc += ((static_cast<float>(v144_data[4])) * v31_data);
              v140_acc += ((static_cast<float>(v144_data[5])) * v32_data);
              v140_acc += ((static_cast<float>(v144_data[6])) * v33_data);
              v140_acc += ((static_cast<float>(v144_data[7])) * v34_data);
              v140_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v161_acc{};
              tensorforge::intel_esimd::simd<float, 16> v165_data;
              v165_data.copy_from(s0 + (48_i32));
              v161_acc += ((static_cast<float>(v165_data[0])) * v27_data);
              v161_acc += ((static_cast<float>(v165_data[1])) * v28_data);
              v161_acc += ((static_cast<float>(v165_data[2])) * v29_data);
              v161_acc += ((static_cast<float>(v165_data[3])) * v30_data);
              v161_acc += ((static_cast<float>(v165_data[4])) * v31_data);
              v161_acc += ((static_cast<float>(v165_data[5])) * v32_data);
              v161_acc += ((static_cast<float>(v165_data[6])) * v33_data);
              v161_acc += ((static_cast<float>(v165_data[7])) * v34_data);
              v161_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v182_acc{};
              tensorforge::intel_esimd::simd<float, 16> v186_data;
              v186_data.copy_from(s0 + (56_i32));
              v182_acc += ((static_cast<float>(v186_data[0])) * v27_data);
              v182_acc += ((static_cast<float>(v186_data[1])) * v28_data);
              v182_acc += ((static_cast<float>(v186_data[2])) * v29_data);
              v182_acc += ((static_cast<float>(v186_data[3])) * v30_data);
              v182_acc += ((static_cast<float>(v186_data[4])) * v31_data);
              v182_acc += ((static_cast<float>(v186_data[5])) * v32_data);
              v182_acc += ((static_cast<float>(v186_data[6])) * v33_data);
              v182_acc += ((static_cast<float>(v186_data[7])) * v34_data);
              v182_acc.copy_to(ir1 + (112));
              #pragma unroll
              for (int32_t v203_n1 = 0; v203_n1 < 8; ++v203_n1) {
                int32_t v204_a = v203_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v206_data;
                v206_data.copy_from(ir1 + (v204_a));
                v206_data.copy_to(r1 + (v204_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v209_i1 = 0; v209_i1 < 8; ++v209_i1) {
                tensorforge::intel_esimd::simd<float, 8> v212_data;
                v212_data.copy_from(r1 + ((v209_i1 * 16)));
                v212_data.copy_to(glb_m0 + ((v209_i1 * 8)));
              }
              // glb_m3 = abs(glb_m0)
              #pragma unroll
              for (int32_t v217_k1 = 0; v217_k1 < 8; ++v217_k1) {
                int32_t v220_a = v217_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v222_data;
                v222_data.copy_from(glb_m0 + (v220_a));
                (tensorforge::intel_esimd::abs(v222_data)).copy_to(glb_m3 + (v220_a));
              }
            }
          }
        }
      });
    }
  });
}

