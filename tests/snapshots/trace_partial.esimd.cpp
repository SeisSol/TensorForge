// === base name ===
kernel_cd5efd48738b4d84

// === header ===
void launcher_kernel_cd5efd48738b4d84(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_cd5efd48738b4d84(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_cd5efd48738b4d84(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_cd5efd48738b4d84(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::grf_size(256)]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 16(16) {0..16} strided
        // m1 16×16(16×16) {0..16}×{0..16} strided
        // m0 16(16) {0..16} strided({0..16})[0] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 16 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 256 + 0 + m1_extraOffset];
              float r0[256]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v9_i0 = 0; v9_i0 < 1; ++v9_i0) {
                int32_t v11_lead = v9_i0 * 16;
                #pragma unroll
                for (int32_t v10_i1 = 0; v10_i1 < 16; ++v10_i1) {
                  int32_t v14_a = v11_lead + (v10_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v15_data;
                  v15_data.copy_from(glb_m1 + (v14_a));
                  v15_data.copy_to(r0 + (v14_a));
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r1[16]{};
              // r1 = +(r0) + None
              // [(0, 16)] [(0, 16)]
              float ir1[16]{};
              tensorforge::intel_esimd::simd<float, 16> v21_data;
              v21_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v22_data;
              v22_data.copy_from(ir1 + (0));
              (v22_data + v21_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v24_data;
              v24_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v25_data;
              v25_data.copy_from(ir1 + (0));
              (v25_data + v24_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v27_data;
              v27_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(ir1 + (0));
              (v28_data + v27_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(ir1 + (0));
              (v31_data + v30_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(ir1 + (0));
              (v34_data + v33_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(ir1 + (0));
              (v37_data + v36_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(ir1 + (0));
              (v40_data + v39_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v42_data;
              v42_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(ir1 + (0));
              (v43_data + v42_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v45_data;
              v45_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v46_data;
              v46_data.copy_from(ir1 + (0));
              (v46_data + v45_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v48_data;
              v48_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v49_data;
              v49_data.copy_from(ir1 + (0));
              (v49_data + v48_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v51_data;
              v51_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v52_data;
              v52_data.copy_from(ir1 + (0));
              (v52_data + v51_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v54_data;
              v54_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v55_data;
              v55_data.copy_from(ir1 + (0));
              (v55_data + v54_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v57_data;
              v57_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v58_data;
              v58_data.copy_from(ir1 + (0));
              (v58_data + v57_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v60_data;
              v60_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v61_data;
              v61_data.copy_from(ir1 + (0));
              (v61_data + v60_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v63_data;
              v63_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v64_data;
              v64_data.copy_from(ir1 + (0));
              (v64_data + v63_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v66_data;
              v66_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v67_data;
              v67_data.copy_from(ir1 + (0));
              (v67_data + v66_data).copy_to(ir1 + (0));
              #pragma unroll
              for (int32_t v69_n0 = 0; v69_n0 < 1; ++v69_n0) {
                int32_t v70_a = v69_n0 * 16;
                tensorforge::intel_esimd::simd<float, 16> v71_data;
                v71_data.copy_from(ir1 + (v70_a));
                v71_data.copy_to(r1 + (v70_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v73_i0 = 0; v73_i0 < 1; ++v73_i0) {
                int32_t v74_a = v73_i0 * 16;
                tensorforge::intel_esimd::simd<float, 16> v75_data;
                v75_data.copy_from(r1 + (v74_a));
                v75_data.copy_to(glb_m0 + (v74_a));
              }
            }
          }
        }
      });
    }
  });
}

