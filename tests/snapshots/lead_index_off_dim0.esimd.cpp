// === base name ===
kernel_c632a22f48915d10

// === header ===
void launcher_kernel_c632a22f48915d10(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_c632a22f48915d10(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 8, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_c632a22f48915d10(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_c632a22f48915d10(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (128, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 20×9(20×9) {0..20}×{0..9} strided
        // m1 1×20(1×20) {0..1}×{0..20} strided
        // m2 1×9(1×9) {0..1}×{0..9} strided
        // m0 20×9(20×9) {0..20}×{0..9} strided({0..20}×{0..9})[0, 1] = m1 1×20(1×20) {0..1}×{0..20} strided({0..1}×{0..20})[-1, 0]×m2 1×9(1×9) {0..1}×{0..9} strided({0..1}×{0..9})[-1, 1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[16];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 180 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 20 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 9 + 0 + m2_extraOffset];
              float r0[32]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
                tensorforge::intel_esimd::simd<float, 20> v19_data;
                v19_data.copy_from(glb_m1 + (v15_i0));
                v19_data.copy_to(r0 + (v15_i0));
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              if (item.get_local_id(0) < 9) {
                tensorforge::intel_esimd::simd<float, 32> v21_ld;
                v21_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 0));
                v21_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 0));
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[288]{};
              // r1 = +(r0 * s0) + None
              // [(0, 20), (0, 9)] [(0, 1)]
              float ir1[288]{};
              tensorforge::intel_esimd::simd<float, 20> v24_data;
              v24_data.copy_from(r0 + (0));
              float v25_data = s0[0];
              tensorforge::intel_esimd::simd<float, 20> v27_data;
              v27_data.copy_from(ir1 + (0));
              (v27_data + (v24_data * v25_data)).copy_to(ir1 + (0));
              float v30_data = s0[1];
              tensorforge::intel_esimd::simd<float, 20> v32_data;
              v32_data.copy_from(ir1 + (32));
              (v32_data + (v24_data * v30_data)).copy_to(ir1 + (32));
              float v35_data = s0[2];
              tensorforge::intel_esimd::simd<float, 20> v37_data;
              v37_data.copy_from(ir1 + (64));
              (v37_data + (v24_data * v35_data)).copy_to(ir1 + (64));
              float v40_data = s0[3];
              tensorforge::intel_esimd::simd<float, 20> v42_data;
              v42_data.copy_from(ir1 + (96));
              (v42_data + (v24_data * v40_data)).copy_to(ir1 + (96));
              float v45_data = s0[4];
              tensorforge::intel_esimd::simd<float, 20> v47_data;
              v47_data.copy_from(ir1 + (128));
              (v47_data + (v24_data * v45_data)).copy_to(ir1 + (128));
              float v50_data = s0[5];
              tensorforge::intel_esimd::simd<float, 20> v52_data;
              v52_data.copy_from(ir1 + (160));
              (v52_data + (v24_data * v50_data)).copy_to(ir1 + (160));
              float v55_data = s0[6];
              tensorforge::intel_esimd::simd<float, 20> v57_data;
              v57_data.copy_from(ir1 + (192));
              (v57_data + (v24_data * v55_data)).copy_to(ir1 + (192));
              float v60_data = s0[7];
              tensorforge::intel_esimd::simd<float, 20> v62_data;
              v62_data.copy_from(ir1 + (224));
              (v62_data + (v24_data * v60_data)).copy_to(ir1 + (224));
              float v65_data = s0[8];
              tensorforge::intel_esimd::simd<float, 20> v67_data;
              v67_data.copy_from(ir1 + (256));
              (v67_data + (v24_data * v65_data)).copy_to(ir1 + (256));
              #pragma unroll
              for (int32_t v69_n1 = 0; v69_n1 < 9; ++v69_n1) {
                int32_t v70_a = v69_n1 * 32;
                tensorforge::intel_esimd::simd<float, 20> v72_data;
                v72_data.copy_from(ir1 + (v70_a));
                v72_data.copy_to(r1 + (v70_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v75_i1 = 0; v75_i1 < 9; ++v75_i1) {
                tensorforge::intel_esimd::simd<float, 20> v78_data;
                v78_data.copy_from(r1 + ((v75_i1 * 32)));
                v78_data.copy_to(glb_m0 + ((v75_i1 * 20)));
              }
            }
          }
        }
      });
    }
  });
}

