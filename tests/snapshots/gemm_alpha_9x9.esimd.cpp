// === base name ===
kernel_82b16e9b4452defb

// === header ===
void launcher_kernel_82b16e9b4452defb(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_82b16e9b4452defb(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_82b16e9b4452defb(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_82b16e9b4452defb(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1792, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 9×9(9×9) {0..9}×{0..9} strided
        // m1 9×9(9×9) {0..9}×{0..9} strided
        // m2 9×9(9×9) {0..9}×{0..9} strided
        // m3 ()  scalar
        // m0 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, 1] = m1 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[0, -1]×m2 9×9(9×9) {0..9}×{0..9} strided({0..9}×{0..9})[-1, 1]×m3 ()  scalar()[]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[112 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[96];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 81 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 81 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 81 + 0 + m2_extraOffset];
              float r0[144]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v15_i1 = 0; v15_i1 < 9; ++v15_i1) {
                tensorforge::intel_esimd::simd<float, 9> v20_data;
                v20_data.copy_from(glb_m1 + ((v15_i1 * 9)));
                v20_data.copy_to(r0 + ((v15_i1 * 16)));
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v23_ld;
              v23_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v23_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              tensorforge::intel_esimd::simd<float, 16> v24_ld;
              v24_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 64));
              v24_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 64));
              if (item.get_local_id(0) < 1) {
                tensorforge::intel_esimd::simd<float, 16> v25_ld;
                v25_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 80));
                v25_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 80));
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[144]{};
              // r1 = +(r0 * s0) + None
              // [(0, 9), (0, 9)] [(0, 9)]
              float ir1[144]{};
              tensorforge::intel_esimd::simd<float, 16> v28_data;
              v28_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v29_data;
              v29_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v30_data;
              v30_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v31_data;
              v31_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v32_data;
              v32_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v33_data;
              v33_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v34_data;
              v34_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v35_data;
              v35_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v36_data;
              v36_data.copy_from(r0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v37_acc{};
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(s0 + (0_i32));
              v37_acc += ((static_cast<float>(v41_data[0])) * v28_data);
              v37_acc += ((static_cast<float>(v41_data[1])) * v29_data);
              v37_acc += ((static_cast<float>(v41_data[2])) * v30_data);
              v37_acc += ((static_cast<float>(v41_data[3])) * v31_data);
              v37_acc += ((static_cast<float>(v41_data[4])) * v32_data);
              v37_acc += ((static_cast<float>(v41_data[5])) * v33_data);
              v37_acc += ((static_cast<float>(v41_data[6])) * v34_data);
              v37_acc += ((static_cast<float>(v41_data[7])) * v35_data);
              v37_acc += ((static_cast<float>(v41_data[8])) * v36_data);
              v37_acc.copy_to(ir1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v60_acc{};
              tensorforge::intel_esimd::simd<float, 16> v64_data;
              v64_data.copy_from(s0 + (9_i32));
              v60_acc += ((static_cast<float>(v64_data[0])) * v28_data);
              v60_acc += ((static_cast<float>(v64_data[1])) * v29_data);
              v60_acc += ((static_cast<float>(v64_data[2])) * v30_data);
              v60_acc += ((static_cast<float>(v64_data[3])) * v31_data);
              v60_acc += ((static_cast<float>(v64_data[4])) * v32_data);
              v60_acc += ((static_cast<float>(v64_data[5])) * v33_data);
              v60_acc += ((static_cast<float>(v64_data[6])) * v34_data);
              v60_acc += ((static_cast<float>(v64_data[7])) * v35_data);
              v60_acc += ((static_cast<float>(v64_data[8])) * v36_data);
              v60_acc.copy_to(ir1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v83_acc{};
              tensorforge::intel_esimd::simd<float, 16> v87_data;
              v87_data.copy_from(s0 + (18_i32));
              v83_acc += ((static_cast<float>(v87_data[0])) * v28_data);
              v83_acc += ((static_cast<float>(v87_data[1])) * v29_data);
              v83_acc += ((static_cast<float>(v87_data[2])) * v30_data);
              v83_acc += ((static_cast<float>(v87_data[3])) * v31_data);
              v83_acc += ((static_cast<float>(v87_data[4])) * v32_data);
              v83_acc += ((static_cast<float>(v87_data[5])) * v33_data);
              v83_acc += ((static_cast<float>(v87_data[6])) * v34_data);
              v83_acc += ((static_cast<float>(v87_data[7])) * v35_data);
              v83_acc += ((static_cast<float>(v87_data[8])) * v36_data);
              v83_acc.copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v106_acc{};
              tensorforge::intel_esimd::simd<float, 16> v110_data;
              v110_data.copy_from(s0 + (27_i32));
              v106_acc += ((static_cast<float>(v110_data[0])) * v28_data);
              v106_acc += ((static_cast<float>(v110_data[1])) * v29_data);
              v106_acc += ((static_cast<float>(v110_data[2])) * v30_data);
              v106_acc += ((static_cast<float>(v110_data[3])) * v31_data);
              v106_acc += ((static_cast<float>(v110_data[4])) * v32_data);
              v106_acc += ((static_cast<float>(v110_data[5])) * v33_data);
              v106_acc += ((static_cast<float>(v110_data[6])) * v34_data);
              v106_acc += ((static_cast<float>(v110_data[7])) * v35_data);
              v106_acc += ((static_cast<float>(v110_data[8])) * v36_data);
              v106_acc.copy_to(ir1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v129_acc{};
              tensorforge::intel_esimd::simd<float, 16> v133_data;
              v133_data.copy_from(s0 + (36_i32));
              v129_acc += ((static_cast<float>(v133_data[0])) * v28_data);
              v129_acc += ((static_cast<float>(v133_data[1])) * v29_data);
              v129_acc += ((static_cast<float>(v133_data[2])) * v30_data);
              v129_acc += ((static_cast<float>(v133_data[3])) * v31_data);
              v129_acc += ((static_cast<float>(v133_data[4])) * v32_data);
              v129_acc += ((static_cast<float>(v133_data[5])) * v33_data);
              v129_acc += ((static_cast<float>(v133_data[6])) * v34_data);
              v129_acc += ((static_cast<float>(v133_data[7])) * v35_data);
              v129_acc += ((static_cast<float>(v133_data[8])) * v36_data);
              v129_acc.copy_to(ir1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v152_acc{};
              tensorforge::intel_esimd::simd<float, 16> v156_data;
              v156_data.copy_from(s0 + (45_i32));
              v152_acc += ((static_cast<float>(v156_data[0])) * v28_data);
              v152_acc += ((static_cast<float>(v156_data[1])) * v29_data);
              v152_acc += ((static_cast<float>(v156_data[2])) * v30_data);
              v152_acc += ((static_cast<float>(v156_data[3])) * v31_data);
              v152_acc += ((static_cast<float>(v156_data[4])) * v32_data);
              v152_acc += ((static_cast<float>(v156_data[5])) * v33_data);
              v152_acc += ((static_cast<float>(v156_data[6])) * v34_data);
              v152_acc += ((static_cast<float>(v156_data[7])) * v35_data);
              v152_acc += ((static_cast<float>(v156_data[8])) * v36_data);
              v152_acc.copy_to(ir1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v175_acc{};
              tensorforge::intel_esimd::simd<float, 16> v179_data;
              v179_data.copy_from(s0 + (54_i32));
              v175_acc += ((static_cast<float>(v179_data[0])) * v28_data);
              v175_acc += ((static_cast<float>(v179_data[1])) * v29_data);
              v175_acc += ((static_cast<float>(v179_data[2])) * v30_data);
              v175_acc += ((static_cast<float>(v179_data[3])) * v31_data);
              v175_acc += ((static_cast<float>(v179_data[4])) * v32_data);
              v175_acc += ((static_cast<float>(v179_data[5])) * v33_data);
              v175_acc += ((static_cast<float>(v179_data[6])) * v34_data);
              v175_acc += ((static_cast<float>(v179_data[7])) * v35_data);
              v175_acc += ((static_cast<float>(v179_data[8])) * v36_data);
              v175_acc.copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v198_acc{};
              tensorforge::intel_esimd::simd<float, 16> v202_data;
              v202_data.copy_from(s0 + (63_i32));
              v198_acc += ((static_cast<float>(v202_data[0])) * v28_data);
              v198_acc += ((static_cast<float>(v202_data[1])) * v29_data);
              v198_acc += ((static_cast<float>(v202_data[2])) * v30_data);
              v198_acc += ((static_cast<float>(v202_data[3])) * v31_data);
              v198_acc += ((static_cast<float>(v202_data[4])) * v32_data);
              v198_acc += ((static_cast<float>(v202_data[5])) * v33_data);
              v198_acc += ((static_cast<float>(v202_data[6])) * v34_data);
              v198_acc += ((static_cast<float>(v202_data[7])) * v35_data);
              v198_acc += ((static_cast<float>(v202_data[8])) * v36_data);
              v198_acc.copy_to(ir1 + (112));
              tensorforge::intel_esimd::simd<float, 16> v221_acc{};
              tensorforge::intel_esimd::simd<float, 16> v225_data;
              v225_data.copy_from(s0 + (72_i32));
              v221_acc += ((static_cast<float>(v225_data[0])) * v28_data);
              v221_acc += ((static_cast<float>(v225_data[1])) * v29_data);
              v221_acc += ((static_cast<float>(v225_data[2])) * v30_data);
              v221_acc += ((static_cast<float>(v225_data[3])) * v31_data);
              v221_acc += ((static_cast<float>(v225_data[4])) * v32_data);
              v221_acc += ((static_cast<float>(v225_data[5])) * v33_data);
              v221_acc += ((static_cast<float>(v225_data[6])) * v34_data);
              v221_acc += ((static_cast<float>(v225_data[7])) * v35_data);
              v221_acc += ((static_cast<float>(v225_data[8])) * v36_data);
              v221_acc.copy_to(ir1 + (128));
              #pragma unroll
              for (int32_t v245_n1 = 0; v245_n1 < 9; ++v245_n1) {
                int32_t v246_a = v245_n1 * 16;
                tensorforge::intel_esimd::simd<float, 9> v248_data;
                v248_data.copy_from(ir1 + (v246_a));
                (v248_data * 13.0f).copy_to(r1 + (v246_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v252_i1 = 0; v252_i1 < 9; ++v252_i1) {
                tensorforge::intel_esimd::simd<float, 9> v255_data;
                v255_data.copy_from(r1 + ((v252_i1 * 16)));
                v255_data.copy_to(glb_m0 + ((v252_i1 * 9)));
              }
            }
          }
        }
      });
    }
  });
}

