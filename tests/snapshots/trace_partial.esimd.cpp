// === base name ===
kernel_98405d0c3443a287

// === header ===
void launcher_kernel_98405d0c3443a287(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_98405d0c3443a287(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_98405d0c3443a287(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_98405d0c3443a287(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 16(16) {0..16} strided
        // m1 16×16(16×16) {0..16}×{0..16} strided
        // m0 16(16) {0..16} strided({0..16})[0] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v2_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v2_batchId0 < numElements0; v2_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v3_ahead1 = v2_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v5_batchId1 = (v3_ahead1 < numElements0) ? v3_ahead1 : v2_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v2_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v2_batchId0 * 16 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v2_batchId0 * 256 + 0 + m1_extraOffset];
              float r0[256]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
                int32_t v15_lead = v13_i0 * 16;
                #pragma unroll
                for (int32_t v14_i1 = 0; v14_i1 < 16; ++v14_i1) {
                  int32_t v18_a = v15_lead + (v14_i1 * 16);
                  tensorforge::intel_esimd::simd<float, 16> v19_data;
                  v19_data.copy_from(glb_m1 + (v18_a));
                  v19_data.copy_to(r0 + (v18_a));
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r1[16]{};
              // r1 = +(r0) + None
              // [(0, 16)] [(0, 16)]
              float ir1[16]{};
              tensorforge::intel_esimd::simd<float, 16> v25_data;
              v25_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v26_data;
              v26_data.copy_from(ir1 + (0));
              (v26_data + v25_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(ir1 + (0));
              (v29_data + v28_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(ir1 + (0));
              (v32_data + v31_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(ir1 + (0));
              (v35_data + v34_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(ir1 + (0));
              (v38_data + v37_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(ir1 + (0));
              (v41_data + v40_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v44_data;
              v44_data.copy_from(ir1 + (0));
              (v44_data + v43_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v46_data;
              v46_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v47_data;
              v47_data.copy_from(ir1 + (0));
              (v47_data + v46_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v49_data;
              v49_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v50_data;
              v50_data.copy_from(ir1 + (0));
              (v50_data + v49_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v52_data;
              v52_data.copy_from(r0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v53_data;
              v53_data.copy_from(ir1 + (0));
              (v53_data + v52_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v55_data;
              v55_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v56_data;
              v56_data.copy_from(ir1 + (0));
              (v56_data + v55_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v58_data;
              v58_data.copy_from(r0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v59_data;
              v59_data.copy_from(ir1 + (0));
              (v59_data + v58_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v61_data;
              v61_data.copy_from(r0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v62_data;
              v62_data.copy_from(ir1 + (0));
              (v62_data + v61_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v64_data;
              v64_data.copy_from(r0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v65_data;
              v65_data.copy_from(ir1 + (0));
              (v65_data + v64_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v67_data;
              v67_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v68_data;
              v68_data.copy_from(ir1 + (0));
              (v68_data + v67_data).copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v70_data;
              v70_data.copy_from(r0 + (240));
              tensorforge::intel_esimd::simd<float, 16> v71_data;
              v71_data.copy_from(ir1 + (0));
              (v71_data + v70_data).copy_to(ir1 + (0));
              #pragma unroll
              for (int32_t v73_n0 = 0; v73_n0 < 1; ++v73_n0) {
                int32_t v74_a = v73_n0 * 16;
                tensorforge::intel_esimd::simd<float, 16> v75_data;
                v75_data.copy_from(ir1 + (v74_a));
                v75_data.copy_to(r1 + (v74_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v77_i0 = 0; v77_i0 < 1; ++v77_i0) {
                int32_t v78_a = v77_i0 * 16;
                tensorforge::intel_esimd::simd<float, 16> v79_data;
                v79_data.copy_from(r1 + (v78_a));
                v79_data.copy_to(glb_m0 + (v78_a));
              }
            }
          }
        }
      });
    }
  });
}

