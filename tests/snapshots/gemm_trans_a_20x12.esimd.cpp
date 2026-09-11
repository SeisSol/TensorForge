// === base name ===
kernel_6e1d8103419a1770

// === header ===
void launcher_kernel_6e1d8103419a1770(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_6e1d8103419a1770(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_6e1d8103419a1770(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_6e1d8103419a1770(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (9472, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 12×16(12×16) {0..12}×{0..16} strided
        // m1 20×12(20×12) {0..20}×{0..12} strided
        // m2 20×16(20×16) {0..20}×{0..16} strided
        // m0 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, 1] = m1 20×12(20×12) {0..20}×{0..12} strided({0..20}×{0..12})[-1, 0]×m2 20×16(20×16) {0..20}×{0..16} strided({0..20}×{0..16})[-1, 1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[592 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[576];
          float * __restrict__ s0 = &localShrMem0[320];
          float * __restrict__ s1 = &localShrMem0[0];
          for (size_t v4_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v4_batchId0 < numElements0; v4_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v5_ahead1 = v4_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v7_batchId1 = (v5_ahead1 < numElements0) ? v5_ahead1 : v4_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v4_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v4_batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v4_batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v4_batchId0 * 320 + 0 + m2_extraOffset];
              // s0 = load{g>s}(glb_m1[1, 0])
              #pragma unroll
              for (int32_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
                int32_t v17_lead = v15_i0 * 16;
                #pragma unroll
                for (int32_t v16_i1 = 0; v16_i1 < 12; ++v16_i1) {
                  tensorforge::intel_esimd::simd<float, 16> v21_data;
                  v21_data.copy_from(glb_m1 + ((v17_lead + (v16_i1 * 20))));
                  v21_data.copy_to(s0 + ((v17_lead + (v16_i1 * 21))));
                }
              }
              #pragma unroll
              for (int32_t v26_i1 = 0; v26_i1 < 12; ++v26_i1) {
                tensorforge::intel_esimd::simd<float, 4> v32_data;
                v32_data.copy_from(glb_m1 + ((16_i32 + (v26_i1 * 20))));
                v32_data.copy_to(s0 + ((16_i32 + (v26_i1 * 21))));
              }
              // s1 = load{g>s}(glb_m2[0, 1])
              #pragma unroll
              for (int32_t i = 0; i < 20; i += 4) {
                tensorforge::intel_esimd::simd<float, 64> v38_ld;
                v38_ld.copy_from(glb_m2 + (0 + 0 + 4 * item.get_local_id(0) + i * 16));
                v38_ld.copy_to(s1 + (0 + 0 + 4 * item.get_local_id(0) + i * 16));
              }
              // wait(s0 = load{g>s}(glb_m1[1, 0]));
              // wait(s1 = load{g>s}(glb_m2[0, 1]));
              float r0[256]{};
              // r0 = +(s0 * s1) + None
              // [(0, 12), (0, 16)] [(0, 20)]
              float ir0[256]{};
              tensorforge::intel_esimd::simd<float, 16> v45_data;
              v45_data.copy_from(s0 + (0_i32));
              tensorforge::intel_esimd::simd<float, 16> v50_data;
              v50_data.copy_from(s0 + (1_i32));
              tensorforge::intel_esimd::simd<float, 16> v55_data;
              v55_data.copy_from(s0 + (2_i32));
              tensorforge::intel_esimd::simd<float, 16> v60_data;
              v60_data.copy_from(s0 + (3_i32));
              tensorforge::intel_esimd::simd<float, 16> v65_data;
              v65_data.copy_from(s0 + (4_i32));
              tensorforge::intel_esimd::simd<float, 16> v70_data;
              v70_data.copy_from(s0 + (5_i32));
              tensorforge::intel_esimd::simd<float, 16> v75_data;
              v75_data.copy_from(s0 + (6_i32));
              tensorforge::intel_esimd::simd<float, 16> v80_data;
              v80_data.copy_from(s0 + (7_i32));
              tensorforge::intel_esimd::simd<float, 16> v85_data;
              v85_data.copy_from(s0 + (8_i32));
              tensorforge::intel_esimd::simd<float, 16> v90_data;
              v90_data.copy_from(s0 + (9_i32));
              tensorforge::intel_esimd::simd<float, 16> v95_data;
              v95_data.copy_from(s0 + (10_i32));
              tensorforge::intel_esimd::simd<float, 16> v100_data;
              v100_data.copy_from(s0 + (11_i32));
              tensorforge::intel_esimd::simd<float, 16> v105_data;
              v105_data.copy_from(s0 + (12_i32));
              tensorforge::intel_esimd::simd<float, 16> v110_data;
              v110_data.copy_from(s0 + (13_i32));
              tensorforge::intel_esimd::simd<float, 16> v115_data;
              v115_data.copy_from(s0 + (14_i32));
              tensorforge::intel_esimd::simd<float, 16> v120_data;
              v120_data.copy_from(s0 + (15_i32));
              tensorforge::intel_esimd::simd<float, 16> v125_data;
              v125_data.copy_from(s0 + (16_i32));
              tensorforge::intel_esimd::simd<float, 16> v130_data;
              v130_data.copy_from(s0 + (17_i32));
              tensorforge::intel_esimd::simd<float, 16> v135_data;
              v135_data.copy_from(s0 + (18_i32));
              tensorforge::intel_esimd::simd<float, 16> v140_data;
              v140_data.copy_from(s0 + (19_i32));
              tensorforge::intel_esimd::simd<float, 16> v141_acc{};
              tensorforge::intel_esimd::simd<float, 16> v145_data;
              v145_data.copy_from(s1 + (0_i32));
              v141_acc += ((static_cast<float>(v145_data[0])) * v45_data);
              v141_acc += ((static_cast<float>(v145_data[1])) * v50_data);
              v141_acc += ((static_cast<float>(v145_data[2])) * v55_data);
              v141_acc += ((static_cast<float>(v145_data[3])) * v60_data);
              v141_acc += ((static_cast<float>(v145_data[4])) * v65_data);
              v141_acc += ((static_cast<float>(v145_data[5])) * v70_data);
              v141_acc += ((static_cast<float>(v145_data[6])) * v75_data);
              v141_acc += ((static_cast<float>(v145_data[7])) * v80_data);
              v141_acc += ((static_cast<float>(v145_data[8])) * v85_data);
              v141_acc += ((static_cast<float>(v145_data[9])) * v90_data);
              v141_acc += ((static_cast<float>(v145_data[10])) * v95_data);
              v141_acc += ((static_cast<float>(v145_data[11])) * v100_data);
              v141_acc += ((static_cast<float>(v145_data[12])) * v105_data);
              v141_acc += ((static_cast<float>(v145_data[13])) * v110_data);
              v141_acc += ((static_cast<float>(v145_data[14])) * v115_data);
              v141_acc += ((static_cast<float>(v145_data[15])) * v120_data);
              tensorforge::intel_esimd::simd<float, 16> v181_data;
              v181_data.copy_from(s1 + (16_i32));
              v141_acc += ((static_cast<float>(v181_data[0])) * v125_data);
              v141_acc += ((static_cast<float>(v181_data[1])) * v130_data);
              v141_acc += ((static_cast<float>(v181_data[2])) * v135_data);
              v141_acc += ((static_cast<float>(v181_data[3])) * v140_data);
              v141_acc.copy_to(ir0 + (0));
              tensorforge::intel_esimd::simd<float, 16> v190_acc{};
              tensorforge::intel_esimd::simd<float, 16> v194_data;
              v194_data.copy_from(s1 + (20_i32));
              v190_acc += ((static_cast<float>(v194_data[0])) * v45_data);
              v190_acc += ((static_cast<float>(v194_data[1])) * v50_data);
              v190_acc += ((static_cast<float>(v194_data[2])) * v55_data);
              v190_acc += ((static_cast<float>(v194_data[3])) * v60_data);
              v190_acc += ((static_cast<float>(v194_data[4])) * v65_data);
              v190_acc += ((static_cast<float>(v194_data[5])) * v70_data);
              v190_acc += ((static_cast<float>(v194_data[6])) * v75_data);
              v190_acc += ((static_cast<float>(v194_data[7])) * v80_data);
              v190_acc += ((static_cast<float>(v194_data[8])) * v85_data);
              v190_acc += ((static_cast<float>(v194_data[9])) * v90_data);
              v190_acc += ((static_cast<float>(v194_data[10])) * v95_data);
              v190_acc += ((static_cast<float>(v194_data[11])) * v100_data);
              v190_acc += ((static_cast<float>(v194_data[12])) * v105_data);
              v190_acc += ((static_cast<float>(v194_data[13])) * v110_data);
              v190_acc += ((static_cast<float>(v194_data[14])) * v115_data);
              v190_acc += ((static_cast<float>(v194_data[15])) * v120_data);
              tensorforge::intel_esimd::simd<float, 16> v230_data;
              v230_data.copy_from(s1 + (36_i32));
              v190_acc += ((static_cast<float>(v230_data[0])) * v125_data);
              v190_acc += ((static_cast<float>(v230_data[1])) * v130_data);
              v190_acc += ((static_cast<float>(v230_data[2])) * v135_data);
              v190_acc += ((static_cast<float>(v230_data[3])) * v140_data);
              v190_acc.copy_to(ir0 + (16));
              tensorforge::intel_esimd::simd<float, 16> v239_acc{};
              tensorforge::intel_esimd::simd<float, 16> v243_data;
              v243_data.copy_from(s1 + (40_i32));
              v239_acc += ((static_cast<float>(v243_data[0])) * v45_data);
              v239_acc += ((static_cast<float>(v243_data[1])) * v50_data);
              v239_acc += ((static_cast<float>(v243_data[2])) * v55_data);
              v239_acc += ((static_cast<float>(v243_data[3])) * v60_data);
              v239_acc += ((static_cast<float>(v243_data[4])) * v65_data);
              v239_acc += ((static_cast<float>(v243_data[5])) * v70_data);
              v239_acc += ((static_cast<float>(v243_data[6])) * v75_data);
              v239_acc += ((static_cast<float>(v243_data[7])) * v80_data);
              v239_acc += ((static_cast<float>(v243_data[8])) * v85_data);
              v239_acc += ((static_cast<float>(v243_data[9])) * v90_data);
              v239_acc += ((static_cast<float>(v243_data[10])) * v95_data);
              v239_acc += ((static_cast<float>(v243_data[11])) * v100_data);
              v239_acc += ((static_cast<float>(v243_data[12])) * v105_data);
              v239_acc += ((static_cast<float>(v243_data[13])) * v110_data);
              v239_acc += ((static_cast<float>(v243_data[14])) * v115_data);
              v239_acc += ((static_cast<float>(v243_data[15])) * v120_data);
              tensorforge::intel_esimd::simd<float, 16> v279_data;
              v279_data.copy_from(s1 + (56_i32));
              v239_acc += ((static_cast<float>(v279_data[0])) * v125_data);
              v239_acc += ((static_cast<float>(v279_data[1])) * v130_data);
              v239_acc += ((static_cast<float>(v279_data[2])) * v135_data);
              v239_acc += ((static_cast<float>(v279_data[3])) * v140_data);
              v239_acc.copy_to(ir0 + (32));
              tensorforge::intel_esimd::simd<float, 16> v288_acc{};
              tensorforge::intel_esimd::simd<float, 16> v292_data;
              v292_data.copy_from(s1 + (60_i32));
              v288_acc += ((static_cast<float>(v292_data[0])) * v45_data);
              v288_acc += ((static_cast<float>(v292_data[1])) * v50_data);
              v288_acc += ((static_cast<float>(v292_data[2])) * v55_data);
              v288_acc += ((static_cast<float>(v292_data[3])) * v60_data);
              v288_acc += ((static_cast<float>(v292_data[4])) * v65_data);
              v288_acc += ((static_cast<float>(v292_data[5])) * v70_data);
              v288_acc += ((static_cast<float>(v292_data[6])) * v75_data);
              v288_acc += ((static_cast<float>(v292_data[7])) * v80_data);
              v288_acc += ((static_cast<float>(v292_data[8])) * v85_data);
              v288_acc += ((static_cast<float>(v292_data[9])) * v90_data);
              v288_acc += ((static_cast<float>(v292_data[10])) * v95_data);
              v288_acc += ((static_cast<float>(v292_data[11])) * v100_data);
              v288_acc += ((static_cast<float>(v292_data[12])) * v105_data);
              v288_acc += ((static_cast<float>(v292_data[13])) * v110_data);
              v288_acc += ((static_cast<float>(v292_data[14])) * v115_data);
              v288_acc += ((static_cast<float>(v292_data[15])) * v120_data);
              tensorforge::intel_esimd::simd<float, 16> v328_data;
              v328_data.copy_from(s1 + (76_i32));
              v288_acc += ((static_cast<float>(v328_data[0])) * v125_data);
              v288_acc += ((static_cast<float>(v328_data[1])) * v130_data);
              v288_acc += ((static_cast<float>(v328_data[2])) * v135_data);
              v288_acc += ((static_cast<float>(v328_data[3])) * v140_data);
              v288_acc.copy_to(ir0 + (48));
              tensorforge::intel_esimd::simd<float, 16> v337_acc{};
              tensorforge::intel_esimd::simd<float, 16> v341_data;
              v341_data.copy_from(s1 + (80_i32));
              v337_acc += ((static_cast<float>(v341_data[0])) * v45_data);
              v337_acc += ((static_cast<float>(v341_data[1])) * v50_data);
              v337_acc += ((static_cast<float>(v341_data[2])) * v55_data);
              v337_acc += ((static_cast<float>(v341_data[3])) * v60_data);
              v337_acc += ((static_cast<float>(v341_data[4])) * v65_data);
              v337_acc += ((static_cast<float>(v341_data[5])) * v70_data);
              v337_acc += ((static_cast<float>(v341_data[6])) * v75_data);
              v337_acc += ((static_cast<float>(v341_data[7])) * v80_data);
              v337_acc += ((static_cast<float>(v341_data[8])) * v85_data);
              v337_acc += ((static_cast<float>(v341_data[9])) * v90_data);
              v337_acc += ((static_cast<float>(v341_data[10])) * v95_data);
              v337_acc += ((static_cast<float>(v341_data[11])) * v100_data);
              v337_acc += ((static_cast<float>(v341_data[12])) * v105_data);
              v337_acc += ((static_cast<float>(v341_data[13])) * v110_data);
              v337_acc += ((static_cast<float>(v341_data[14])) * v115_data);
              v337_acc += ((static_cast<float>(v341_data[15])) * v120_data);
              tensorforge::intel_esimd::simd<float, 16> v377_data;
              v377_data.copy_from(s1 + (96_i32));
              v337_acc += ((static_cast<float>(v377_data[0])) * v125_data);
              v337_acc += ((static_cast<float>(v377_data[1])) * v130_data);
              v337_acc += ((static_cast<float>(v377_data[2])) * v135_data);
              v337_acc += ((static_cast<float>(v377_data[3])) * v140_data);
              v337_acc.copy_to(ir0 + (64));
              tensorforge::intel_esimd::simd<float, 16> v386_acc{};
              tensorforge::intel_esimd::simd<float, 16> v390_data;
              v390_data.copy_from(s1 + (100_i32));
              v386_acc += ((static_cast<float>(v390_data[0])) * v45_data);
              v386_acc += ((static_cast<float>(v390_data[1])) * v50_data);
              v386_acc += ((static_cast<float>(v390_data[2])) * v55_data);
              v386_acc += ((static_cast<float>(v390_data[3])) * v60_data);
              v386_acc += ((static_cast<float>(v390_data[4])) * v65_data);
              v386_acc += ((static_cast<float>(v390_data[5])) * v70_data);
              v386_acc += ((static_cast<float>(v390_data[6])) * v75_data);
              v386_acc += ((static_cast<float>(v390_data[7])) * v80_data);
              v386_acc += ((static_cast<float>(v390_data[8])) * v85_data);
              v386_acc += ((static_cast<float>(v390_data[9])) * v90_data);
              v386_acc += ((static_cast<float>(v390_data[10])) * v95_data);
              v386_acc += ((static_cast<float>(v390_data[11])) * v100_data);
              v386_acc += ((static_cast<float>(v390_data[12])) * v105_data);
              v386_acc += ((static_cast<float>(v390_data[13])) * v110_data);
              v386_acc += ((static_cast<float>(v390_data[14])) * v115_data);
              v386_acc += ((static_cast<float>(v390_data[15])) * v120_data);
              tensorforge::intel_esimd::simd<float, 16> v426_data;
              v426_data.copy_from(s1 + (116_i32));
              v386_acc += ((static_cast<float>(v426_data[0])) * v125_data);
              v386_acc += ((static_cast<float>(v426_data[1])) * v130_data);
              v386_acc += ((static_cast<float>(v426_data[2])) * v135_data);
              v386_acc += ((static_cast<float>(v426_data[3])) * v140_data);
              v386_acc.copy_to(ir0 + (80));
              tensorforge::intel_esimd::simd<float, 16> v435_acc{};
              tensorforge::intel_esimd::simd<float, 16> v439_data;
              v439_data.copy_from(s1 + (120_i32));
              v435_acc += ((static_cast<float>(v439_data[0])) * v45_data);
              v435_acc += ((static_cast<float>(v439_data[1])) * v50_data);
              v435_acc += ((static_cast<float>(v439_data[2])) * v55_data);
              v435_acc += ((static_cast<float>(v439_data[3])) * v60_data);
              v435_acc += ((static_cast<float>(v439_data[4])) * v65_data);
              v435_acc += ((static_cast<float>(v439_data[5])) * v70_data);
              v435_acc += ((static_cast<float>(v439_data[6])) * v75_data);
              v435_acc += ((static_cast<float>(v439_data[7])) * v80_data);
              v435_acc += ((static_cast<float>(v439_data[8])) * v85_data);
              v435_acc += ((static_cast<float>(v439_data[9])) * v90_data);
              v435_acc += ((static_cast<float>(v439_data[10])) * v95_data);
              v435_acc += ((static_cast<float>(v439_data[11])) * v100_data);
              v435_acc += ((static_cast<float>(v439_data[12])) * v105_data);
              v435_acc += ((static_cast<float>(v439_data[13])) * v110_data);
              v435_acc += ((static_cast<float>(v439_data[14])) * v115_data);
              v435_acc += ((static_cast<float>(v439_data[15])) * v120_data);
              tensorforge::intel_esimd::simd<float, 16> v475_data;
              v475_data.copy_from(s1 + (136_i32));
              v435_acc += ((static_cast<float>(v475_data[0])) * v125_data);
              v435_acc += ((static_cast<float>(v475_data[1])) * v130_data);
              v435_acc += ((static_cast<float>(v475_data[2])) * v135_data);
              v435_acc += ((static_cast<float>(v475_data[3])) * v140_data);
              v435_acc.copy_to(ir0 + (96));
              tensorforge::intel_esimd::simd<float, 16> v484_acc{};
              tensorforge::intel_esimd::simd<float, 16> v488_data;
              v488_data.copy_from(s1 + (140_i32));
              v484_acc += ((static_cast<float>(v488_data[0])) * v45_data);
              v484_acc += ((static_cast<float>(v488_data[1])) * v50_data);
              v484_acc += ((static_cast<float>(v488_data[2])) * v55_data);
              v484_acc += ((static_cast<float>(v488_data[3])) * v60_data);
              v484_acc += ((static_cast<float>(v488_data[4])) * v65_data);
              v484_acc += ((static_cast<float>(v488_data[5])) * v70_data);
              v484_acc += ((static_cast<float>(v488_data[6])) * v75_data);
              v484_acc += ((static_cast<float>(v488_data[7])) * v80_data);
              v484_acc += ((static_cast<float>(v488_data[8])) * v85_data);
              v484_acc += ((static_cast<float>(v488_data[9])) * v90_data);
              v484_acc += ((static_cast<float>(v488_data[10])) * v95_data);
              v484_acc += ((static_cast<float>(v488_data[11])) * v100_data);
              v484_acc += ((static_cast<float>(v488_data[12])) * v105_data);
              v484_acc += ((static_cast<float>(v488_data[13])) * v110_data);
              v484_acc += ((static_cast<float>(v488_data[14])) * v115_data);
              v484_acc += ((static_cast<float>(v488_data[15])) * v120_data);
              tensorforge::intel_esimd::simd<float, 16> v524_data;
              v524_data.copy_from(s1 + (156_i32));
              v484_acc += ((static_cast<float>(v524_data[0])) * v125_data);
              v484_acc += ((static_cast<float>(v524_data[1])) * v130_data);
              v484_acc += ((static_cast<float>(v524_data[2])) * v135_data);
              v484_acc += ((static_cast<float>(v524_data[3])) * v140_data);
              v484_acc.copy_to(ir0 + (112));
              tensorforge::intel_esimd::simd<float, 16> v533_acc{};
              tensorforge::intel_esimd::simd<float, 16> v537_data;
              v537_data.copy_from(s1 + (160_i32));
              v533_acc += ((static_cast<float>(v537_data[0])) * v45_data);
              v533_acc += ((static_cast<float>(v537_data[1])) * v50_data);
              v533_acc += ((static_cast<float>(v537_data[2])) * v55_data);
              v533_acc += ((static_cast<float>(v537_data[3])) * v60_data);
              v533_acc += ((static_cast<float>(v537_data[4])) * v65_data);
              v533_acc += ((static_cast<float>(v537_data[5])) * v70_data);
              v533_acc += ((static_cast<float>(v537_data[6])) * v75_data);
              v533_acc += ((static_cast<float>(v537_data[7])) * v80_data);
              v533_acc += ((static_cast<float>(v537_data[8])) * v85_data);
              v533_acc += ((static_cast<float>(v537_data[9])) * v90_data);
              v533_acc += ((static_cast<float>(v537_data[10])) * v95_data);
              v533_acc += ((static_cast<float>(v537_data[11])) * v100_data);
              v533_acc += ((static_cast<float>(v537_data[12])) * v105_data);
              v533_acc += ((static_cast<float>(v537_data[13])) * v110_data);
              v533_acc += ((static_cast<float>(v537_data[14])) * v115_data);
              v533_acc += ((static_cast<float>(v537_data[15])) * v120_data);
              tensorforge::intel_esimd::simd<float, 16> v573_data;
              v573_data.copy_from(s1 + (176_i32));
              v533_acc += ((static_cast<float>(v573_data[0])) * v125_data);
              v533_acc += ((static_cast<float>(v573_data[1])) * v130_data);
              v533_acc += ((static_cast<float>(v573_data[2])) * v135_data);
              v533_acc += ((static_cast<float>(v573_data[3])) * v140_data);
              v533_acc.copy_to(ir0 + (128));
              tensorforge::intel_esimd::simd<float, 16> v582_acc{};
              tensorforge::intel_esimd::simd<float, 16> v586_data;
              v586_data.copy_from(s1 + (180_i32));
              v582_acc += ((static_cast<float>(v586_data[0])) * v45_data);
              v582_acc += ((static_cast<float>(v586_data[1])) * v50_data);
              v582_acc += ((static_cast<float>(v586_data[2])) * v55_data);
              v582_acc += ((static_cast<float>(v586_data[3])) * v60_data);
              v582_acc += ((static_cast<float>(v586_data[4])) * v65_data);
              v582_acc += ((static_cast<float>(v586_data[5])) * v70_data);
              v582_acc += ((static_cast<float>(v586_data[6])) * v75_data);
              v582_acc += ((static_cast<float>(v586_data[7])) * v80_data);
              v582_acc += ((static_cast<float>(v586_data[8])) * v85_data);
              v582_acc += ((static_cast<float>(v586_data[9])) * v90_data);
              v582_acc += ((static_cast<float>(v586_data[10])) * v95_data);
              v582_acc += ((static_cast<float>(v586_data[11])) * v100_data);
              v582_acc += ((static_cast<float>(v586_data[12])) * v105_data);
              v582_acc += ((static_cast<float>(v586_data[13])) * v110_data);
              v582_acc += ((static_cast<float>(v586_data[14])) * v115_data);
              v582_acc += ((static_cast<float>(v586_data[15])) * v120_data);
              tensorforge::intel_esimd::simd<float, 16> v622_data;
              v622_data.copy_from(s1 + (196_i32));
              v582_acc += ((static_cast<float>(v622_data[0])) * v125_data);
              v582_acc += ((static_cast<float>(v622_data[1])) * v130_data);
              v582_acc += ((static_cast<float>(v622_data[2])) * v135_data);
              v582_acc += ((static_cast<float>(v622_data[3])) * v140_data);
              v582_acc.copy_to(ir0 + (144));
              tensorforge::intel_esimd::simd<float, 16> v631_acc{};
              tensorforge::intel_esimd::simd<float, 16> v635_data;
              v635_data.copy_from(s1 + (200_i32));
              v631_acc += ((static_cast<float>(v635_data[0])) * v45_data);
              v631_acc += ((static_cast<float>(v635_data[1])) * v50_data);
              v631_acc += ((static_cast<float>(v635_data[2])) * v55_data);
              v631_acc += ((static_cast<float>(v635_data[3])) * v60_data);
              v631_acc += ((static_cast<float>(v635_data[4])) * v65_data);
              v631_acc += ((static_cast<float>(v635_data[5])) * v70_data);
              v631_acc += ((static_cast<float>(v635_data[6])) * v75_data);
              v631_acc += ((static_cast<float>(v635_data[7])) * v80_data);
              v631_acc += ((static_cast<float>(v635_data[8])) * v85_data);
              v631_acc += ((static_cast<float>(v635_data[9])) * v90_data);
              v631_acc += ((static_cast<float>(v635_data[10])) * v95_data);
              v631_acc += ((static_cast<float>(v635_data[11])) * v100_data);
              v631_acc += ((static_cast<float>(v635_data[12])) * v105_data);
              v631_acc += ((static_cast<float>(v635_data[13])) * v110_data);
              v631_acc += ((static_cast<float>(v635_data[14])) * v115_data);
              v631_acc += ((static_cast<float>(v635_data[15])) * v120_data);
              tensorforge::intel_esimd::simd<float, 16> v671_data;
              v671_data.copy_from(s1 + (216_i32));
              v631_acc += ((static_cast<float>(v671_data[0])) * v125_data);
              v631_acc += ((static_cast<float>(v671_data[1])) * v130_data);
              v631_acc += ((static_cast<float>(v671_data[2])) * v135_data);
              v631_acc += ((static_cast<float>(v671_data[3])) * v140_data);
              v631_acc.copy_to(ir0 + (160));
              tensorforge::intel_esimd::simd<float, 16> v680_acc{};
              tensorforge::intel_esimd::simd<float, 16> v684_data;
              v684_data.copy_from(s1 + (220_i32));
              v680_acc += ((static_cast<float>(v684_data[0])) * v45_data);
              v680_acc += ((static_cast<float>(v684_data[1])) * v50_data);
              v680_acc += ((static_cast<float>(v684_data[2])) * v55_data);
              v680_acc += ((static_cast<float>(v684_data[3])) * v60_data);
              v680_acc += ((static_cast<float>(v684_data[4])) * v65_data);
              v680_acc += ((static_cast<float>(v684_data[5])) * v70_data);
              v680_acc += ((static_cast<float>(v684_data[6])) * v75_data);
              v680_acc += ((static_cast<float>(v684_data[7])) * v80_data);
              v680_acc += ((static_cast<float>(v684_data[8])) * v85_data);
              v680_acc += ((static_cast<float>(v684_data[9])) * v90_data);
              v680_acc += ((static_cast<float>(v684_data[10])) * v95_data);
              v680_acc += ((static_cast<float>(v684_data[11])) * v100_data);
              v680_acc += ((static_cast<float>(v684_data[12])) * v105_data);
              v680_acc += ((static_cast<float>(v684_data[13])) * v110_data);
              v680_acc += ((static_cast<float>(v684_data[14])) * v115_data);
              v680_acc += ((static_cast<float>(v684_data[15])) * v120_data);
              tensorforge::intel_esimd::simd<float, 16> v720_data;
              v720_data.copy_from(s1 + (236_i32));
              v680_acc += ((static_cast<float>(v720_data[0])) * v125_data);
              v680_acc += ((static_cast<float>(v720_data[1])) * v130_data);
              v680_acc += ((static_cast<float>(v720_data[2])) * v135_data);
              v680_acc += ((static_cast<float>(v720_data[3])) * v140_data);
              v680_acc.copy_to(ir0 + (176));
              tensorforge::intel_esimd::simd<float, 16> v729_acc{};
              tensorforge::intel_esimd::simd<float, 16> v733_data;
              v733_data.copy_from(s1 + (240_i32));
              v729_acc += ((static_cast<float>(v733_data[0])) * v45_data);
              v729_acc += ((static_cast<float>(v733_data[1])) * v50_data);
              v729_acc += ((static_cast<float>(v733_data[2])) * v55_data);
              v729_acc += ((static_cast<float>(v733_data[3])) * v60_data);
              v729_acc += ((static_cast<float>(v733_data[4])) * v65_data);
              v729_acc += ((static_cast<float>(v733_data[5])) * v70_data);
              v729_acc += ((static_cast<float>(v733_data[6])) * v75_data);
              v729_acc += ((static_cast<float>(v733_data[7])) * v80_data);
              v729_acc += ((static_cast<float>(v733_data[8])) * v85_data);
              v729_acc += ((static_cast<float>(v733_data[9])) * v90_data);
              v729_acc += ((static_cast<float>(v733_data[10])) * v95_data);
              v729_acc += ((static_cast<float>(v733_data[11])) * v100_data);
              v729_acc += ((static_cast<float>(v733_data[12])) * v105_data);
              v729_acc += ((static_cast<float>(v733_data[13])) * v110_data);
              v729_acc += ((static_cast<float>(v733_data[14])) * v115_data);
              v729_acc += ((static_cast<float>(v733_data[15])) * v120_data);
              tensorforge::intel_esimd::simd<float, 16> v769_data;
              v769_data.copy_from(s1 + (256_i32));
              v729_acc += ((static_cast<float>(v769_data[0])) * v125_data);
              v729_acc += ((static_cast<float>(v769_data[1])) * v130_data);
              v729_acc += ((static_cast<float>(v769_data[2])) * v135_data);
              v729_acc += ((static_cast<float>(v769_data[3])) * v140_data);
              v729_acc.copy_to(ir0 + (192));
              tensorforge::intel_esimd::simd<float, 16> v778_acc{};
              tensorforge::intel_esimd::simd<float, 16> v782_data;
              v782_data.copy_from(s1 + (260_i32));
              v778_acc += ((static_cast<float>(v782_data[0])) * v45_data);
              v778_acc += ((static_cast<float>(v782_data[1])) * v50_data);
              v778_acc += ((static_cast<float>(v782_data[2])) * v55_data);
              v778_acc += ((static_cast<float>(v782_data[3])) * v60_data);
              v778_acc += ((static_cast<float>(v782_data[4])) * v65_data);
              v778_acc += ((static_cast<float>(v782_data[5])) * v70_data);
              v778_acc += ((static_cast<float>(v782_data[6])) * v75_data);
              v778_acc += ((static_cast<float>(v782_data[7])) * v80_data);
              v778_acc += ((static_cast<float>(v782_data[8])) * v85_data);
              v778_acc += ((static_cast<float>(v782_data[9])) * v90_data);
              v778_acc += ((static_cast<float>(v782_data[10])) * v95_data);
              v778_acc += ((static_cast<float>(v782_data[11])) * v100_data);
              v778_acc += ((static_cast<float>(v782_data[12])) * v105_data);
              v778_acc += ((static_cast<float>(v782_data[13])) * v110_data);
              v778_acc += ((static_cast<float>(v782_data[14])) * v115_data);
              v778_acc += ((static_cast<float>(v782_data[15])) * v120_data);
              tensorforge::intel_esimd::simd<float, 16> v818_data;
              v818_data.copy_from(s1 + (276_i32));
              v778_acc += ((static_cast<float>(v818_data[0])) * v125_data);
              v778_acc += ((static_cast<float>(v818_data[1])) * v130_data);
              v778_acc += ((static_cast<float>(v818_data[2])) * v135_data);
              v778_acc += ((static_cast<float>(v818_data[3])) * v140_data);
              v778_acc.copy_to(ir0 + (208));
              tensorforge::intel_esimd::simd<float, 16> v827_acc{};
              tensorforge::intel_esimd::simd<float, 16> v831_data;
              v831_data.copy_from(s1 + (280_i32));
              v827_acc += ((static_cast<float>(v831_data[0])) * v45_data);
              v827_acc += ((static_cast<float>(v831_data[1])) * v50_data);
              v827_acc += ((static_cast<float>(v831_data[2])) * v55_data);
              v827_acc += ((static_cast<float>(v831_data[3])) * v60_data);
              v827_acc += ((static_cast<float>(v831_data[4])) * v65_data);
              v827_acc += ((static_cast<float>(v831_data[5])) * v70_data);
              v827_acc += ((static_cast<float>(v831_data[6])) * v75_data);
              v827_acc += ((static_cast<float>(v831_data[7])) * v80_data);
              v827_acc += ((static_cast<float>(v831_data[8])) * v85_data);
              v827_acc += ((static_cast<float>(v831_data[9])) * v90_data);
              v827_acc += ((static_cast<float>(v831_data[10])) * v95_data);
              v827_acc += ((static_cast<float>(v831_data[11])) * v100_data);
              v827_acc += ((static_cast<float>(v831_data[12])) * v105_data);
              v827_acc += ((static_cast<float>(v831_data[13])) * v110_data);
              v827_acc += ((static_cast<float>(v831_data[14])) * v115_data);
              v827_acc += ((static_cast<float>(v831_data[15])) * v120_data);
              tensorforge::intel_esimd::simd<float, 16> v867_data;
              v867_data.copy_from(s1 + (296_i32));
              v827_acc += ((static_cast<float>(v867_data[0])) * v125_data);
              v827_acc += ((static_cast<float>(v867_data[1])) * v130_data);
              v827_acc += ((static_cast<float>(v867_data[2])) * v135_data);
              v827_acc += ((static_cast<float>(v867_data[3])) * v140_data);
              v827_acc.copy_to(ir0 + (224));
              tensorforge::intel_esimd::simd<float, 16> v876_acc{};
              tensorforge::intel_esimd::simd<float, 16> v880_data;
              v880_data.copy_from(s1 + (300_i32));
              v876_acc += ((static_cast<float>(v880_data[0])) * v45_data);
              v876_acc += ((static_cast<float>(v880_data[1])) * v50_data);
              v876_acc += ((static_cast<float>(v880_data[2])) * v55_data);
              v876_acc += ((static_cast<float>(v880_data[3])) * v60_data);
              v876_acc += ((static_cast<float>(v880_data[4])) * v65_data);
              v876_acc += ((static_cast<float>(v880_data[5])) * v70_data);
              v876_acc += ((static_cast<float>(v880_data[6])) * v75_data);
              v876_acc += ((static_cast<float>(v880_data[7])) * v80_data);
              v876_acc += ((static_cast<float>(v880_data[8])) * v85_data);
              v876_acc += ((static_cast<float>(v880_data[9])) * v90_data);
              v876_acc += ((static_cast<float>(v880_data[10])) * v95_data);
              v876_acc += ((static_cast<float>(v880_data[11])) * v100_data);
              v876_acc += ((static_cast<float>(v880_data[12])) * v105_data);
              v876_acc += ((static_cast<float>(v880_data[13])) * v110_data);
              v876_acc += ((static_cast<float>(v880_data[14])) * v115_data);
              v876_acc += ((static_cast<float>(v880_data[15])) * v120_data);
              tensorforge::intel_esimd::simd<float, 16> v916_data;
              v916_data.copy_from(s1 + (316_i32));
              v876_acc += ((static_cast<float>(v916_data[0])) * v125_data);
              v876_acc += ((static_cast<float>(v916_data[1])) * v130_data);
              v876_acc += ((static_cast<float>(v916_data[2])) * v135_data);
              v876_acc += ((static_cast<float>(v916_data[3])) * v140_data);
              v876_acc.copy_to(ir0 + (240));
              #pragma unroll
              for (int32_t v925_n1 = 0; v925_n1 < 16; ++v925_n1) {
                int32_t v926_a = v925_n1 * 16;
                tensorforge::intel_esimd::simd<float, 12> v928_data;
                v928_data.copy_from(ir0 + (v926_a));
                v928_data.copy_to(r0 + (v926_a));
              }
              // glb_m0 = store{r>g}(r0);
              #pragma unroll
              for (int32_t v931_i1 = 0; v931_i1 < 16; ++v931_i1) {
                tensorforge::intel_esimd::simd<float, 12> v934_data;
                v934_data.copy_from(r0 + ((v931_i1 * 16)));
                v934_data.copy_to(glb_m0 + ((v931_i1 * 12)));
              }
            }
          }
        }
      });
    }
  });
}

