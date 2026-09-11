// === base name ===
kernel_7abf2269e28f0c35

// === header ===
void launcher_kernel_7abf2269e28f0c35(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_7abf2269e28f0c35(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_7abf2269e28f0c35(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_7abf2269e28f0c35(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (1536, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 32×32(12×6) {0..12}×{0..6} strided
        // m1 32×32(6×6) {0..6}×{0..6} strided
        // m2 32×32(12×6) {0..12}×{0..6} strided
        // m3 32×32(12×12) {0..12}×{0..12} strided
        // t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[0, 1] = m0 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, -1]×m1 32×32(6×6) {0..6}×{0..6} strided({0..6}×{0..6})[-1, 1]
        // m2 32×32(12×6) {0..12}×{0..6} strided({0..12}×{0..6})[0, 1] = m3 32×32(12×12) {0..12}×{0..12} strided({0..12}×{0..12})[0, -1]×t0 12×6(12×6) {0..12}×{0..6} pointer_based({0..12}×{0..6})[-1, 1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[96 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[80];
          float * __restrict__ s0 = &localShrMem0[0];
          float * __restrict__ s1 = &localShrMem0[0];
          for (size_t v4_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v4_batchId0 < numElements0; v4_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v5_ahead1 = v4_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 36 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v4_batchId0 * 144 + 0 + m3_extraOffset];
              float r0[96]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v17_i1 = 0; v17_i1 < 6; ++v17_i1) {
                tensorforge::intel_esimd::simd<float, 12> v22_data;
                v22_data.copy_from(glb_m0 + ((v17_i1 * 12)));
                v22_data.copy_to(r0 + ((v17_i1 * 16)));
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v25_ld;
              v25_ld.copy_from(glb_m1 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              v25_ld.copy_to(s0 + (0 + 0 + 2 * item.get_local_id(0) + 0));
              if (item.get_local_id(0) < 4) {
                tensorforge::intel_esimd::simd<float, 16> v26_ld;
                v26_ld.copy_from(glb_m1 + (0 + 0 + 1 * item.get_local_id(0) + 32));
                v26_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 32));
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r2[192]{};
              // r2 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v28_i1 = 0; v28_i1 < 12; ++v28_i1) {
                tensorforge::intel_esimd::simd<float, 12> v33_data;
                v33_data.copy_from(glb_m3 + ((v28_i1 * 12)));
                v33_data.copy_to(r2 + ((v28_i1 * 16)));
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r1[96]{};
              // r1 = +(r0 * s0) + None
              // [(0, 12), (0, 6)] [(0, 6)]
              tensorforge::intel_esimd::simd<float, 16> v37_data;
              v37_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v42_data;
              v42_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v43_acc{};
              tensorforge::intel_esimd::simd<float, 16> v47_data;
              v47_data.copy_from(s0 + (0_i32));
              v43_acc += ((static_cast<float>(v47_data[0])) * v37_data);
              v43_acc += ((static_cast<float>(v47_data[1])) * v38_data);
              v43_acc += ((static_cast<float>(v47_data[2])) * v39_data);
              v43_acc += ((static_cast<float>(v47_data[3])) * v40_data);
              v43_acc += ((static_cast<float>(v47_data[4])) * v41_data);
              v43_acc += ((static_cast<float>(v47_data[5])) * v42_data);
              v43_acc.copy_to(r1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v60_acc{};
              tensorforge::intel_esimd::simd<float, 16> v64_data;
              v64_data.copy_from(s0 + (6_i32));
              v60_acc += ((static_cast<float>(v64_data[0])) * v37_data);
              v60_acc += ((static_cast<float>(v64_data[1])) * v38_data);
              v60_acc += ((static_cast<float>(v64_data[2])) * v39_data);
              v60_acc += ((static_cast<float>(v64_data[3])) * v40_data);
              v60_acc += ((static_cast<float>(v64_data[4])) * v41_data);
              v60_acc += ((static_cast<float>(v64_data[5])) * v42_data);
              v60_acc.copy_to(r1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v77_acc{};
              tensorforge::intel_esimd::simd<float, 16> v81_data;
              v81_data.copy_from(s0 + (12_i32));
              v77_acc += ((static_cast<float>(v81_data[0])) * v37_data);
              v77_acc += ((static_cast<float>(v81_data[1])) * v38_data);
              v77_acc += ((static_cast<float>(v81_data[2])) * v39_data);
              v77_acc += ((static_cast<float>(v81_data[3])) * v40_data);
              v77_acc += ((static_cast<float>(v81_data[4])) * v41_data);
              v77_acc += ((static_cast<float>(v81_data[5])) * v42_data);
              v77_acc.copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v94_acc{};
              tensorforge::intel_esimd::simd<float, 16> v98_data;
              v98_data.copy_from(s0 + (18_i32));
              v94_acc += ((static_cast<float>(v98_data[0])) * v37_data);
              v94_acc += ((static_cast<float>(v98_data[1])) * v38_data);
              v94_acc += ((static_cast<float>(v98_data[2])) * v39_data);
              v94_acc += ((static_cast<float>(v98_data[3])) * v40_data);
              v94_acc += ((static_cast<float>(v98_data[4])) * v41_data);
              v94_acc += ((static_cast<float>(v98_data[5])) * v42_data);
              v94_acc.copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v111_acc{};
              tensorforge::intel_esimd::simd<float, 16> v115_data;
              v115_data.copy_from(s0 + (24_i32));
              v111_acc += ((static_cast<float>(v115_data[0])) * v37_data);
              v111_acc += ((static_cast<float>(v115_data[1])) * v38_data);
              v111_acc += ((static_cast<float>(v115_data[2])) * v39_data);
              v111_acc += ((static_cast<float>(v115_data[3])) * v40_data);
              v111_acc += ((static_cast<float>(v115_data[4])) * v41_data);
              v111_acc += ((static_cast<float>(v115_data[5])) * v42_data);
              v111_acc.copy_to(r1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v128_acc{};
              tensorforge::intel_esimd::simd<float, 16> v132_data;
              v132_data.copy_from(s0 + (30_i32));
              v128_acc += ((static_cast<float>(v132_data[0])) * v37_data);
              v128_acc += ((static_cast<float>(v132_data[1])) * v38_data);
              v128_acc += ((static_cast<float>(v132_data[2])) * v39_data);
              v128_acc += ((static_cast<float>(v132_data[3])) * v40_data);
              v128_acc += ((static_cast<float>(v132_data[4])) * v41_data);
              v128_acc += ((static_cast<float>(v132_data[5])) * v42_data);
              v128_acc.copy_to(r1 + (80));
              // wait(r2 = load{g>r}(glb_m3););
              // s1 = store{r>s}(localShrMem0, r1);
              #pragma unroll
              for (int32_t v145_i1 = 0; v145_i1 < 6; ++v145_i1) {
                tensorforge::intel_esimd::simd<float, 12> v148_data;
                v148_data.copy_from(r1 + ((v145_i1 * 16)));
                v148_data.copy_to(s1 + ((v145_i1 * 12)));
              }
              float r3[96]{};
              // r3 = +(r2 * s1) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              float ir3[96]{};
              tensorforge::intel_esimd::simd<float, 16> v155_data;
              v155_data.copy_from(r2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v156_data;
              v156_data.copy_from(r2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v157_data;
              v157_data.copy_from(r2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v158_data;
              v158_data.copy_from(r2 + (48));
              tensorforge::intel_esimd::simd<float, 16> v159_data;
              v159_data.copy_from(r2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v160_data;
              v160_data.copy_from(r2 + (80));
              tensorforge::intel_esimd::simd<float, 16> v161_data;
              v161_data.copy_from(r2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v162_data;
              v162_data.copy_from(r2 + (112));
              tensorforge::intel_esimd::simd<float, 16> v163_data;
              v163_data.copy_from(r2 + (128));
              tensorforge::intel_esimd::simd<float, 16> v164_data;
              v164_data.copy_from(r2 + (144));
              tensorforge::intel_esimd::simd<float, 16> v165_data;
              v165_data.copy_from(r2 + (160));
              tensorforge::intel_esimd::simd<float, 16> v166_data;
              v166_data.copy_from(r2 + (176));
              tensorforge::intel_esimd::simd<float, 16> v167_acc{};
              tensorforge::intel_esimd::simd<float, 16> v171_data;
              v171_data.copy_from(s1 + (0_i32));
              v167_acc += ((static_cast<float>(v171_data[0])) * v155_data);
              v167_acc += ((static_cast<float>(v171_data[1])) * v156_data);
              v167_acc += ((static_cast<float>(v171_data[2])) * v157_data);
              v167_acc += ((static_cast<float>(v171_data[3])) * v158_data);
              v167_acc += ((static_cast<float>(v171_data[4])) * v159_data);
              v167_acc += ((static_cast<float>(v171_data[5])) * v160_data);
              v167_acc += ((static_cast<float>(v171_data[6])) * v161_data);
              v167_acc += ((static_cast<float>(v171_data[7])) * v162_data);
              v167_acc += ((static_cast<float>(v171_data[8])) * v163_data);
              v167_acc += ((static_cast<float>(v171_data[9])) * v164_data);
              v167_acc += ((static_cast<float>(v171_data[10])) * v165_data);
              v167_acc += ((static_cast<float>(v171_data[11])) * v166_data);
              v167_acc.copy_to(ir3 + (0));
              tensorforge::intel_esimd::simd<float, 16> v196_acc{};
              tensorforge::intel_esimd::simd<float, 16> v200_data;
              v200_data.copy_from(s1 + (12_i32));
              v196_acc += ((static_cast<float>(v200_data[0])) * v155_data);
              v196_acc += ((static_cast<float>(v200_data[1])) * v156_data);
              v196_acc += ((static_cast<float>(v200_data[2])) * v157_data);
              v196_acc += ((static_cast<float>(v200_data[3])) * v158_data);
              v196_acc += ((static_cast<float>(v200_data[4])) * v159_data);
              v196_acc += ((static_cast<float>(v200_data[5])) * v160_data);
              v196_acc += ((static_cast<float>(v200_data[6])) * v161_data);
              v196_acc += ((static_cast<float>(v200_data[7])) * v162_data);
              v196_acc += ((static_cast<float>(v200_data[8])) * v163_data);
              v196_acc += ((static_cast<float>(v200_data[9])) * v164_data);
              v196_acc += ((static_cast<float>(v200_data[10])) * v165_data);
              v196_acc += ((static_cast<float>(v200_data[11])) * v166_data);
              v196_acc.copy_to(ir3 + (16));
              tensorforge::intel_esimd::simd<float, 16> v225_acc{};
              tensorforge::intel_esimd::simd<float, 16> v229_data;
              v229_data.copy_from(s1 + (24_i32));
              v225_acc += ((static_cast<float>(v229_data[0])) * v155_data);
              v225_acc += ((static_cast<float>(v229_data[1])) * v156_data);
              v225_acc += ((static_cast<float>(v229_data[2])) * v157_data);
              v225_acc += ((static_cast<float>(v229_data[3])) * v158_data);
              v225_acc += ((static_cast<float>(v229_data[4])) * v159_data);
              v225_acc += ((static_cast<float>(v229_data[5])) * v160_data);
              v225_acc += ((static_cast<float>(v229_data[6])) * v161_data);
              v225_acc += ((static_cast<float>(v229_data[7])) * v162_data);
              v225_acc += ((static_cast<float>(v229_data[8])) * v163_data);
              v225_acc += ((static_cast<float>(v229_data[9])) * v164_data);
              v225_acc += ((static_cast<float>(v229_data[10])) * v165_data);
              v225_acc += ((static_cast<float>(v229_data[11])) * v166_data);
              v225_acc.copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 16> v254_acc{};
              tensorforge::intel_esimd::simd<float, 16> v258_data;
              v258_data.copy_from(s1 + (36_i32));
              v254_acc += ((static_cast<float>(v258_data[0])) * v155_data);
              v254_acc += ((static_cast<float>(v258_data[1])) * v156_data);
              v254_acc += ((static_cast<float>(v258_data[2])) * v157_data);
              v254_acc += ((static_cast<float>(v258_data[3])) * v158_data);
              v254_acc += ((static_cast<float>(v258_data[4])) * v159_data);
              v254_acc += ((static_cast<float>(v258_data[5])) * v160_data);
              v254_acc += ((static_cast<float>(v258_data[6])) * v161_data);
              v254_acc += ((static_cast<float>(v258_data[7])) * v162_data);
              v254_acc += ((static_cast<float>(v258_data[8])) * v163_data);
              v254_acc += ((static_cast<float>(v258_data[9])) * v164_data);
              v254_acc += ((static_cast<float>(v258_data[10])) * v165_data);
              v254_acc += ((static_cast<float>(v258_data[11])) * v166_data);
              v254_acc.copy_to(ir3 + (48));
              tensorforge::intel_esimd::simd<float, 16> v283_acc{};
              tensorforge::intel_esimd::simd<float, 16> v287_data;
              v287_data.copy_from(s1 + (48_i32));
              v283_acc += ((static_cast<float>(v287_data[0])) * v155_data);
              v283_acc += ((static_cast<float>(v287_data[1])) * v156_data);
              v283_acc += ((static_cast<float>(v287_data[2])) * v157_data);
              v283_acc += ((static_cast<float>(v287_data[3])) * v158_data);
              v283_acc += ((static_cast<float>(v287_data[4])) * v159_data);
              v283_acc += ((static_cast<float>(v287_data[5])) * v160_data);
              v283_acc += ((static_cast<float>(v287_data[6])) * v161_data);
              v283_acc += ((static_cast<float>(v287_data[7])) * v162_data);
              v283_acc += ((static_cast<float>(v287_data[8])) * v163_data);
              v283_acc += ((static_cast<float>(v287_data[9])) * v164_data);
              v283_acc += ((static_cast<float>(v287_data[10])) * v165_data);
              v283_acc += ((static_cast<float>(v287_data[11])) * v166_data);
              v283_acc.copy_to(ir3 + (64));
              tensorforge::intel_esimd::simd<float, 16> v312_acc{};
              tensorforge::intel_esimd::simd<float, 16> v316_data;
              v316_data.copy_from(s1 + (60_i32));
              v312_acc += ((static_cast<float>(v316_data[0])) * v155_data);
              v312_acc += ((static_cast<float>(v316_data[1])) * v156_data);
              v312_acc += ((static_cast<float>(v316_data[2])) * v157_data);
              v312_acc += ((static_cast<float>(v316_data[3])) * v158_data);
              v312_acc += ((static_cast<float>(v316_data[4])) * v159_data);
              v312_acc += ((static_cast<float>(v316_data[5])) * v160_data);
              v312_acc += ((static_cast<float>(v316_data[6])) * v161_data);
              v312_acc += ((static_cast<float>(v316_data[7])) * v162_data);
              v312_acc += ((static_cast<float>(v316_data[8])) * v163_data);
              v312_acc += ((static_cast<float>(v316_data[9])) * v164_data);
              v312_acc += ((static_cast<float>(v316_data[10])) * v165_data);
              v312_acc += ((static_cast<float>(v316_data[11])) * v166_data);
              v312_acc.copy_to(ir3 + (80));
              #pragma unroll
              for (int32_t v341_n1 = 0; v341_n1 < 6; ++v341_n1) {
                int32_t v342_a = v341_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v344_data;
                v344_data.copy_from(ir3 + (v342_a));
                v344_data.copy_to(r3 + (v342_a));
              }
              // glb_m2 = store{r>g}(r3);
              #pragma unroll
              for (int32_t v347_i1 = 0; v347_i1 < 6; ++v347_i1) {
                tensorforge::intel_esimd::simd<float, 12> v350_data;
                v350_data.copy_from(r3 + ((v347_i1 * 16)));
                v350_data.copy_to(glb_m2 + ((v347_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

