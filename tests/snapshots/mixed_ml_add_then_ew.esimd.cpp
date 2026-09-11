// === base name ===
kernel_26f4420e3d69851b

// === header ===
void launcher_kernel_26f4420e3d69851b(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_26f4420e3d69851b(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_26f4420e3d69851b(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_26f4420e3d69851b(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
        // m4 8×8(8×8) {0..8}×{0..8} strided
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] += m2 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m3 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // C = abs(TMP)
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[80 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          float * __restrict__ s0 = &localShrMem0[0];
          float * __restrict__ s2 = &localShrMem0[0];
          float * __restrict__ s1 = &localShrMem0[0];
          for (size_t v5_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v5_batchId0 < numElements0; v5_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v6_ahead1 = v5_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v8_batchId1 = (v6_ahead1 < numElements0) ? v6_ahead1 : v5_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v5_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v5_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v5_batchId0 * 64 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v5_batchId0 * 64 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v5_batchId0 * 64 + 0 + m3_extraOffset];
              float *const __restrict__ glb_m4 = &m4[v5_batchId0 * 64 + 0 + m4_extraOffset];
              float r0[128]{};
              // r0 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v19_i1 = 0; v19_i1 < 8; ++v19_i1) {
                tensorforge::intel_esimd::simd<float, 8> v24_data;
                v24_data.copy_from(glb_m0 + ((v19_i1 * 8)));
                v24_data.copy_to(r0 + ((v19_i1 * 16)));
              }
              // s0 = load{g>s}(glb_m1[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v27_ld;
              v27_ld.copy_from(glb_m1 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v27_ld.copy_to(s0 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              // wait(r0 = load{g>r}(glb_m0););
              float r2[128]{};
              // r2 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v29_i1 = 0; v29_i1 < 8; ++v29_i1) {
                tensorforge::intel_esimd::simd<float, 8> v34_data;
                v34_data.copy_from(glb_m2 + ((v29_i1 * 8)));
                v34_data.copy_to(r2 + ((v29_i1 * 16)));
              }
              // wait(s0 = load{g>s}(glb_m1[0, 1]));
              float r1[128]{};
              // r1 = +(r0 * s0) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              tensorforge::intel_esimd::simd<float, 16> v38_data;
              v38_data.copy_from(r0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v39_data;
              v39_data.copy_from(r0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v40_data;
              v40_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v41_data;
              v41_data.copy_from(r0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v42_data;
              v42_data.copy_from(r0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v43_data;
              v43_data.copy_from(r0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v44_data;
              v44_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v45_data;
              v45_data.copy_from(r0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v46_acc{};
              tensorforge::intel_esimd::simd<float, 16> v50_data;
              v50_data.copy_from(s0 + (0_i32));
              v46_acc += ((static_cast<float>(v50_data[0])) * v38_data);
              v46_acc += ((static_cast<float>(v50_data[1])) * v39_data);
              v46_acc += ((static_cast<float>(v50_data[2])) * v40_data);
              v46_acc += ((static_cast<float>(v50_data[3])) * v41_data);
              v46_acc += ((static_cast<float>(v50_data[4])) * v42_data);
              v46_acc += ((static_cast<float>(v50_data[5])) * v43_data);
              v46_acc += ((static_cast<float>(v50_data[6])) * v44_data);
              v46_acc += ((static_cast<float>(v50_data[7])) * v45_data);
              v46_acc.copy_to(r1 + (0));
              tensorforge::intel_esimd::simd<float, 16> v67_acc{};
              tensorforge::intel_esimd::simd<float, 16> v71_data;
              v71_data.copy_from(s0 + (8_i32));
              v67_acc += ((static_cast<float>(v71_data[0])) * v38_data);
              v67_acc += ((static_cast<float>(v71_data[1])) * v39_data);
              v67_acc += ((static_cast<float>(v71_data[2])) * v40_data);
              v67_acc += ((static_cast<float>(v71_data[3])) * v41_data);
              v67_acc += ((static_cast<float>(v71_data[4])) * v42_data);
              v67_acc += ((static_cast<float>(v71_data[5])) * v43_data);
              v67_acc += ((static_cast<float>(v71_data[6])) * v44_data);
              v67_acc += ((static_cast<float>(v71_data[7])) * v45_data);
              v67_acc.copy_to(r1 + (16));
              tensorforge::intel_esimd::simd<float, 16> v88_acc{};
              tensorforge::intel_esimd::simd<float, 16> v92_data;
              v92_data.copy_from(s0 + (16_i32));
              v88_acc += ((static_cast<float>(v92_data[0])) * v38_data);
              v88_acc += ((static_cast<float>(v92_data[1])) * v39_data);
              v88_acc += ((static_cast<float>(v92_data[2])) * v40_data);
              v88_acc += ((static_cast<float>(v92_data[3])) * v41_data);
              v88_acc += ((static_cast<float>(v92_data[4])) * v42_data);
              v88_acc += ((static_cast<float>(v92_data[5])) * v43_data);
              v88_acc += ((static_cast<float>(v92_data[6])) * v44_data);
              v88_acc += ((static_cast<float>(v92_data[7])) * v45_data);
              v88_acc.copy_to(r1 + (32));
              tensorforge::intel_esimd::simd<float, 16> v109_acc{};
              tensorforge::intel_esimd::simd<float, 16> v113_data;
              v113_data.copy_from(s0 + (24_i32));
              v109_acc += ((static_cast<float>(v113_data[0])) * v38_data);
              v109_acc += ((static_cast<float>(v113_data[1])) * v39_data);
              v109_acc += ((static_cast<float>(v113_data[2])) * v40_data);
              v109_acc += ((static_cast<float>(v113_data[3])) * v41_data);
              v109_acc += ((static_cast<float>(v113_data[4])) * v42_data);
              v109_acc += ((static_cast<float>(v113_data[5])) * v43_data);
              v109_acc += ((static_cast<float>(v113_data[6])) * v44_data);
              v109_acc += ((static_cast<float>(v113_data[7])) * v45_data);
              v109_acc.copy_to(r1 + (48));
              tensorforge::intel_esimd::simd<float, 16> v130_acc{};
              tensorforge::intel_esimd::simd<float, 16> v134_data;
              v134_data.copy_from(s0 + (32_i32));
              v130_acc += ((static_cast<float>(v134_data[0])) * v38_data);
              v130_acc += ((static_cast<float>(v134_data[1])) * v39_data);
              v130_acc += ((static_cast<float>(v134_data[2])) * v40_data);
              v130_acc += ((static_cast<float>(v134_data[3])) * v41_data);
              v130_acc += ((static_cast<float>(v134_data[4])) * v42_data);
              v130_acc += ((static_cast<float>(v134_data[5])) * v43_data);
              v130_acc += ((static_cast<float>(v134_data[6])) * v44_data);
              v130_acc += ((static_cast<float>(v134_data[7])) * v45_data);
              v130_acc.copy_to(r1 + (64));
              tensorforge::intel_esimd::simd<float, 16> v151_acc{};
              tensorforge::intel_esimd::simd<float, 16> v155_data;
              v155_data.copy_from(s0 + (40_i32));
              v151_acc += ((static_cast<float>(v155_data[0])) * v38_data);
              v151_acc += ((static_cast<float>(v155_data[1])) * v39_data);
              v151_acc += ((static_cast<float>(v155_data[2])) * v40_data);
              v151_acc += ((static_cast<float>(v155_data[3])) * v41_data);
              v151_acc += ((static_cast<float>(v155_data[4])) * v42_data);
              v151_acc += ((static_cast<float>(v155_data[5])) * v43_data);
              v151_acc += ((static_cast<float>(v155_data[6])) * v44_data);
              v151_acc += ((static_cast<float>(v155_data[7])) * v45_data);
              v151_acc.copy_to(r1 + (80));
              tensorforge::intel_esimd::simd<float, 16> v172_acc{};
              tensorforge::intel_esimd::simd<float, 16> v176_data;
              v176_data.copy_from(s0 + (48_i32));
              v172_acc += ((static_cast<float>(v176_data[0])) * v38_data);
              v172_acc += ((static_cast<float>(v176_data[1])) * v39_data);
              v172_acc += ((static_cast<float>(v176_data[2])) * v40_data);
              v172_acc += ((static_cast<float>(v176_data[3])) * v41_data);
              v172_acc += ((static_cast<float>(v176_data[4])) * v42_data);
              v172_acc += ((static_cast<float>(v176_data[5])) * v43_data);
              v172_acc += ((static_cast<float>(v176_data[6])) * v44_data);
              v172_acc += ((static_cast<float>(v176_data[7])) * v45_data);
              v172_acc.copy_to(r1 + (96));
              tensorforge::intel_esimd::simd<float, 16> v193_acc{};
              tensorforge::intel_esimd::simd<float, 16> v197_data;
              v197_data.copy_from(s0 + (56_i32));
              v193_acc += ((static_cast<float>(v197_data[0])) * v38_data);
              v193_acc += ((static_cast<float>(v197_data[1])) * v39_data);
              v193_acc += ((static_cast<float>(v197_data[2])) * v40_data);
              v193_acc += ((static_cast<float>(v197_data[3])) * v41_data);
              v193_acc += ((static_cast<float>(v197_data[4])) * v42_data);
              v193_acc += ((static_cast<float>(v197_data[5])) * v43_data);
              v193_acc += ((static_cast<float>(v197_data[6])) * v44_data);
              v193_acc += ((static_cast<float>(v197_data[7])) * v45_data);
              v193_acc.copy_to(r1 + (112));
              // s2 = load{g>s}(glb_m3[0, 1])
              tensorforge::intel_esimd::simd<float, 64> v214_ld;
              v214_ld.copy_from(glb_m3 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              v214_ld.copy_to(s2 + (0 + 0 + 4 * item.get_local_id(0) + 0));
              // wait(r2 = load{g>r}(glb_m2););
              // wait(s2 = load{g>s}(glb_m3[0, 1]));
              float r3[128]{};
              // r3 = +(r2 * s2) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 8), (0, 8)] [(0, 8)]
              float ir3[128]{};
              tensorforge::intel_esimd::simd<float, 16> v217_data;
              v217_data.copy_from(r2 + (0));
              tensorforge::intel_esimd::simd<float, 16> v218_data;
              v218_data.copy_from(r2 + (16));
              tensorforge::intel_esimd::simd<float, 16> v219_data;
              v219_data.copy_from(r2 + (32));
              tensorforge::intel_esimd::simd<float, 16> v220_data;
              v220_data.copy_from(r2 + (48));
              tensorforge::intel_esimd::simd<float, 16> v221_data;
              v221_data.copy_from(r2 + (64));
              tensorforge::intel_esimd::simd<float, 16> v222_data;
              v222_data.copy_from(r2 + (80));
              tensorforge::intel_esimd::simd<float, 16> v223_data;
              v223_data.copy_from(r2 + (96));
              tensorforge::intel_esimd::simd<float, 16> v224_data;
              v224_data.copy_from(r2 + (112));
              tensorforge::intel_esimd::simd<float, 16> v225_acc{};
              tensorforge::intel_esimd::simd<float, 16> v229_data;
              v229_data.copy_from(s2 + (0_i32));
              v225_acc += ((static_cast<float>(v229_data[0])) * v217_data);
              v225_acc += ((static_cast<float>(v229_data[1])) * v218_data);
              v225_acc += ((static_cast<float>(v229_data[2])) * v219_data);
              v225_acc += ((static_cast<float>(v229_data[3])) * v220_data);
              v225_acc += ((static_cast<float>(v229_data[4])) * v221_data);
              v225_acc += ((static_cast<float>(v229_data[5])) * v222_data);
              v225_acc += ((static_cast<float>(v229_data[6])) * v223_data);
              v225_acc += ((static_cast<float>(v229_data[7])) * v224_data);
              v225_acc.copy_to(ir3 + (0));
              tensorforge::intel_esimd::simd<float, 16> v246_acc{};
              tensorforge::intel_esimd::simd<float, 16> v250_data;
              v250_data.copy_from(s2 + (8_i32));
              v246_acc += ((static_cast<float>(v250_data[0])) * v217_data);
              v246_acc += ((static_cast<float>(v250_data[1])) * v218_data);
              v246_acc += ((static_cast<float>(v250_data[2])) * v219_data);
              v246_acc += ((static_cast<float>(v250_data[3])) * v220_data);
              v246_acc += ((static_cast<float>(v250_data[4])) * v221_data);
              v246_acc += ((static_cast<float>(v250_data[5])) * v222_data);
              v246_acc += ((static_cast<float>(v250_data[6])) * v223_data);
              v246_acc += ((static_cast<float>(v250_data[7])) * v224_data);
              v246_acc.copy_to(ir3 + (16));
              tensorforge::intel_esimd::simd<float, 16> v267_acc{};
              tensorforge::intel_esimd::simd<float, 16> v271_data;
              v271_data.copy_from(s2 + (16_i32));
              v267_acc += ((static_cast<float>(v271_data[0])) * v217_data);
              v267_acc += ((static_cast<float>(v271_data[1])) * v218_data);
              v267_acc += ((static_cast<float>(v271_data[2])) * v219_data);
              v267_acc += ((static_cast<float>(v271_data[3])) * v220_data);
              v267_acc += ((static_cast<float>(v271_data[4])) * v221_data);
              v267_acc += ((static_cast<float>(v271_data[5])) * v222_data);
              v267_acc += ((static_cast<float>(v271_data[6])) * v223_data);
              v267_acc += ((static_cast<float>(v271_data[7])) * v224_data);
              v267_acc.copy_to(ir3 + (32));
              tensorforge::intel_esimd::simd<float, 16> v288_acc{};
              tensorforge::intel_esimd::simd<float, 16> v292_data;
              v292_data.copy_from(s2 + (24_i32));
              v288_acc += ((static_cast<float>(v292_data[0])) * v217_data);
              v288_acc += ((static_cast<float>(v292_data[1])) * v218_data);
              v288_acc += ((static_cast<float>(v292_data[2])) * v219_data);
              v288_acc += ((static_cast<float>(v292_data[3])) * v220_data);
              v288_acc += ((static_cast<float>(v292_data[4])) * v221_data);
              v288_acc += ((static_cast<float>(v292_data[5])) * v222_data);
              v288_acc += ((static_cast<float>(v292_data[6])) * v223_data);
              v288_acc += ((static_cast<float>(v292_data[7])) * v224_data);
              v288_acc.copy_to(ir3 + (48));
              tensorforge::intel_esimd::simd<float, 16> v309_acc{};
              tensorforge::intel_esimd::simd<float, 16> v313_data;
              v313_data.copy_from(s2 + (32_i32));
              v309_acc += ((static_cast<float>(v313_data[0])) * v217_data);
              v309_acc += ((static_cast<float>(v313_data[1])) * v218_data);
              v309_acc += ((static_cast<float>(v313_data[2])) * v219_data);
              v309_acc += ((static_cast<float>(v313_data[3])) * v220_data);
              v309_acc += ((static_cast<float>(v313_data[4])) * v221_data);
              v309_acc += ((static_cast<float>(v313_data[5])) * v222_data);
              v309_acc += ((static_cast<float>(v313_data[6])) * v223_data);
              v309_acc += ((static_cast<float>(v313_data[7])) * v224_data);
              v309_acc.copy_to(ir3 + (64));
              tensorforge::intel_esimd::simd<float, 16> v330_acc{};
              tensorforge::intel_esimd::simd<float, 16> v334_data;
              v334_data.copy_from(s2 + (40_i32));
              v330_acc += ((static_cast<float>(v334_data[0])) * v217_data);
              v330_acc += ((static_cast<float>(v334_data[1])) * v218_data);
              v330_acc += ((static_cast<float>(v334_data[2])) * v219_data);
              v330_acc += ((static_cast<float>(v334_data[3])) * v220_data);
              v330_acc += ((static_cast<float>(v334_data[4])) * v221_data);
              v330_acc += ((static_cast<float>(v334_data[5])) * v222_data);
              v330_acc += ((static_cast<float>(v334_data[6])) * v223_data);
              v330_acc += ((static_cast<float>(v334_data[7])) * v224_data);
              v330_acc.copy_to(ir3 + (80));
              tensorforge::intel_esimd::simd<float, 16> v351_acc{};
              tensorforge::intel_esimd::simd<float, 16> v355_data;
              v355_data.copy_from(s2 + (48_i32));
              v351_acc += ((static_cast<float>(v355_data[0])) * v217_data);
              v351_acc += ((static_cast<float>(v355_data[1])) * v218_data);
              v351_acc += ((static_cast<float>(v355_data[2])) * v219_data);
              v351_acc += ((static_cast<float>(v355_data[3])) * v220_data);
              v351_acc += ((static_cast<float>(v355_data[4])) * v221_data);
              v351_acc += ((static_cast<float>(v355_data[5])) * v222_data);
              v351_acc += ((static_cast<float>(v355_data[6])) * v223_data);
              v351_acc += ((static_cast<float>(v355_data[7])) * v224_data);
              v351_acc.copy_to(ir3 + (96));
              tensorforge::intel_esimd::simd<float, 16> v372_acc{};
              tensorforge::intel_esimd::simd<float, 16> v376_data;
              v376_data.copy_from(s2 + (56_i32));
              v372_acc += ((static_cast<float>(v376_data[0])) * v217_data);
              v372_acc += ((static_cast<float>(v376_data[1])) * v218_data);
              v372_acc += ((static_cast<float>(v376_data[2])) * v219_data);
              v372_acc += ((static_cast<float>(v376_data[3])) * v220_data);
              v372_acc += ((static_cast<float>(v376_data[4])) * v221_data);
              v372_acc += ((static_cast<float>(v376_data[5])) * v222_data);
              v372_acc += ((static_cast<float>(v376_data[6])) * v223_data);
              v372_acc += ((static_cast<float>(v376_data[7])) * v224_data);
              v372_acc.copy_to(ir3 + (112));
              #pragma unroll
              for (int32_t v393_n1 = 0; v393_n1 < 8; ++v393_n1) {
                int32_t v394_a = v393_n1 * 16;
                tensorforge::intel_esimd::simd<float, 8> v396_data;
                v396_data.copy_from(ir3 + (v394_a));
                tensorforge::intel_esimd::simd<float, 8> v399_data;
                v399_data.copy_from(r1 + (v394_a));
                (v399_data + v396_data).copy_to(r3 + (v394_a));
              }
              // s1 = store{r>s}(localShrMem0, r3);
              #pragma unroll
              for (int32_t v403_i1 = 0; v403_i1 < 8; ++v403_i1) {
                tensorforge::intel_esimd::simd<float, 8> v406_data;
                v406_data.copy_from(r3 + ((v403_i1 * 16)));
                v406_data.copy_to(s1 + ((v403_i1 * 8)));
              }
              // glb_m4 = abs(s1)
              #pragma unroll
              for (int32_t v411_k1 = 0; v411_k1 < 8; ++v411_k1) {
                int32_t v414_a = v411_k1 * 8;
                tensorforge::intel_esimd::simd<float, 8> v416_data;
                v416_data.copy_from(s1 + (v414_a));
                (tensorforge::intel_esimd::abs(v416_data)).copy_to(glb_m4 + (v414_a));
              }
            }
          }
        }
      });
    }
  });
}

