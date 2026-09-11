// === base name ===
kernel_bc8f6cd7a9d5289e

// === header ===
void launcher_kernel_bc8f6cd7a9d5289e(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_bc8f6cd7a9d5289e(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_bc8f6cd7a9d5289e(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_bc8f6cd7a9d5289e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, sycl::ext::oneapi::experimental::properties{sycl::ext::intel::experimental::grf_size<256>}, [=](sycl::nd_item<3> item) [[intel::sycl_explicit_simd]] [[intel::kernel_args_restrict]] {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 35×4(35×4) {0..35}×{0..4} strided
        // m1 35×8(35×8) {0..35}×{0..8} strided
        // m2 8×4(8×4) {0..8}×{0..4} strided
        // m0 35×4(35×4) {0..35}×{0..4} strided({0..35}×{0..4})[0, 1] = m1 35×8(35×8) {0..35}×{0..8} strided({0..35}×{0..8})[0, -1]×m2 8×4(8×4) {0..8}×{0..4} strided({0..8}×{0..4})[-1, 1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[32 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[32];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 140 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 280 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 32 + 0 + m2_extraOffset];
              float r0[512]{};
              // r0 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v15_i0 = 0; v15_i0 < 1; ++v15_i0) {
                int32_t v17_lead = v15_i0 * 32;
                #pragma unroll
                for (int32_t v16_i1 = 0; v16_i1 < 8; ++v16_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v21_data;
                  v21_data.copy_from(glb_m1 + ((v17_lead + (v16_i1 * 35))));
                  v21_data.copy_to(r0 + ((v17_lead + (v16_i1 * 64))));
                }
              }
              #pragma unroll
              for (int32_t v25_i1 = 0; v25_i1 < 8; ++v25_i1) {
                tensorforge::intel_esimd::simd<float, 3> v31_data;
                v31_data.copy_from(glb_m1 + ((32_i32 + (v25_i1 * 35))));
                v31_data.copy_to(r0 + ((32 + (v25_i1 * 64))));
              }
              // s0 = load{g>s}(glb_m2[0, 1])
              tensorforge::intel_esimd::simd<float, 32> v34_ld;
              v34_ld.copy_from(glb_m2 + (0 + 0 + 1 * item.get_local_id(0) + 0));
              v34_ld.copy_to(s0 + (0 + 0 + 1 * item.get_local_id(0) + 0));
              // wait(r0 = load{g>r}(glb_m1););
              // wait(s0 = load{g>s}(glb_m2[0, 1]));
              float r1[256]{};
              // r1 = +(r0 * s0) + None
              // [(0, 35), (0, 4)] [(0, 8)]
              float ir1[256]{};
              tensorforge::intel_esimd::simd<float, 32> v37_data;
              v37_data.copy_from(r0 + (0));
              float v38_data = s0[0];
              tensorforge::intel_esimd::simd<float, 32> v40_data;
              v40_data.copy_from(ir1 + (0));
              (v40_data + (v37_data * v38_data)).copy_to(ir1 + (0));
              float v43_data = s0[8];
              tensorforge::intel_esimd::simd<float, 32> v45_data;
              v45_data.copy_from(ir1 + (64));
              (v45_data + (v37_data * v43_data)).copy_to(ir1 + (64));
              float v48_data = s0[16];
              tensorforge::intel_esimd::simd<float, 32> v50_data;
              v50_data.copy_from(ir1 + (128));
              (v50_data + (v37_data * v48_data)).copy_to(ir1 + (128));
              float v53_data = s0[24];
              tensorforge::intel_esimd::simd<float, 32> v55_data;
              v55_data.copy_from(ir1 + (192));
              (v55_data + (v37_data * v53_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v57_data;
              v57_data.copy_from(r0 + (32));
              tensorforge::intel_esimd::simd<float, 3> v60_data;
              v60_data.copy_from(ir1 + (32));
              (v60_data + (v57_data * v38_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v65_data;
              v65_data.copy_from(ir1 + (96));
              (v65_data + (v57_data * v43_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v70_data;
              v70_data.copy_from(ir1 + (160));
              (v70_data + (v57_data * v48_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v75_data;
              v75_data.copy_from(ir1 + (224));
              (v75_data + (v57_data * v53_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v77_data;
              v77_data.copy_from(r0 + (64));
              float v78_data = s0[1];
              tensorforge::intel_esimd::simd<float, 32> v80_data;
              v80_data.copy_from(ir1 + (0));
              (v80_data + (v77_data * v78_data)).copy_to(ir1 + (0));
              float v83_data = s0[9];
              tensorforge::intel_esimd::simd<float, 32> v85_data;
              v85_data.copy_from(ir1 + (64));
              (v85_data + (v77_data * v83_data)).copy_to(ir1 + (64));
              float v88_data = s0[17];
              tensorforge::intel_esimd::simd<float, 32> v90_data;
              v90_data.copy_from(ir1 + (128));
              (v90_data + (v77_data * v88_data)).copy_to(ir1 + (128));
              float v93_data = s0[25];
              tensorforge::intel_esimd::simd<float, 32> v95_data;
              v95_data.copy_from(ir1 + (192));
              (v95_data + (v77_data * v93_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v97_data;
              v97_data.copy_from(r0 + (96));
              tensorforge::intel_esimd::simd<float, 3> v100_data;
              v100_data.copy_from(ir1 + (32));
              (v100_data + (v97_data * v78_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v105_data;
              v105_data.copy_from(ir1 + (96));
              (v105_data + (v97_data * v83_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v110_data;
              v110_data.copy_from(ir1 + (160));
              (v110_data + (v97_data * v88_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v115_data;
              v115_data.copy_from(ir1 + (224));
              (v115_data + (v97_data * v93_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v117_data;
              v117_data.copy_from(r0 + (128));
              float v118_data = s0[2];
              tensorforge::intel_esimd::simd<float, 32> v120_data;
              v120_data.copy_from(ir1 + (0));
              (v120_data + (v117_data * v118_data)).copy_to(ir1 + (0));
              float v123_data = s0[10];
              tensorforge::intel_esimd::simd<float, 32> v125_data;
              v125_data.copy_from(ir1 + (64));
              (v125_data + (v117_data * v123_data)).copy_to(ir1 + (64));
              float v128_data = s0[18];
              tensorforge::intel_esimd::simd<float, 32> v130_data;
              v130_data.copy_from(ir1 + (128));
              (v130_data + (v117_data * v128_data)).copy_to(ir1 + (128));
              float v133_data = s0[26];
              tensorforge::intel_esimd::simd<float, 32> v135_data;
              v135_data.copy_from(ir1 + (192));
              (v135_data + (v117_data * v133_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v137_data;
              v137_data.copy_from(r0 + (160));
              tensorforge::intel_esimd::simd<float, 3> v140_data;
              v140_data.copy_from(ir1 + (32));
              (v140_data + (v137_data * v118_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v145_data;
              v145_data.copy_from(ir1 + (96));
              (v145_data + (v137_data * v123_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v150_data;
              v150_data.copy_from(ir1 + (160));
              (v150_data + (v137_data * v128_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v155_data;
              v155_data.copy_from(ir1 + (224));
              (v155_data + (v137_data * v133_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v157_data;
              v157_data.copy_from(r0 + (192));
              float v158_data = s0[3];
              tensorforge::intel_esimd::simd<float, 32> v160_data;
              v160_data.copy_from(ir1 + (0));
              (v160_data + (v157_data * v158_data)).copy_to(ir1 + (0));
              float v163_data = s0[11];
              tensorforge::intel_esimd::simd<float, 32> v165_data;
              v165_data.copy_from(ir1 + (64));
              (v165_data + (v157_data * v163_data)).copy_to(ir1 + (64));
              float v168_data = s0[19];
              tensorforge::intel_esimd::simd<float, 32> v170_data;
              v170_data.copy_from(ir1 + (128));
              (v170_data + (v157_data * v168_data)).copy_to(ir1 + (128));
              float v173_data = s0[27];
              tensorforge::intel_esimd::simd<float, 32> v175_data;
              v175_data.copy_from(ir1 + (192));
              (v175_data + (v157_data * v173_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v177_data;
              v177_data.copy_from(r0 + (224));
              tensorforge::intel_esimd::simd<float, 3> v180_data;
              v180_data.copy_from(ir1 + (32));
              (v180_data + (v177_data * v158_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v185_data;
              v185_data.copy_from(ir1 + (96));
              (v185_data + (v177_data * v163_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v190_data;
              v190_data.copy_from(ir1 + (160));
              (v190_data + (v177_data * v168_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v195_data;
              v195_data.copy_from(ir1 + (224));
              (v195_data + (v177_data * v173_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v197_data;
              v197_data.copy_from(r0 + (256));
              float v198_data = s0[4];
              tensorforge::intel_esimd::simd<float, 32> v200_data;
              v200_data.copy_from(ir1 + (0));
              (v200_data + (v197_data * v198_data)).copy_to(ir1 + (0));
              float v203_data = s0[12];
              tensorforge::intel_esimd::simd<float, 32> v205_data;
              v205_data.copy_from(ir1 + (64));
              (v205_data + (v197_data * v203_data)).copy_to(ir1 + (64));
              float v208_data = s0[20];
              tensorforge::intel_esimd::simd<float, 32> v210_data;
              v210_data.copy_from(ir1 + (128));
              (v210_data + (v197_data * v208_data)).copy_to(ir1 + (128));
              float v213_data = s0[28];
              tensorforge::intel_esimd::simd<float, 32> v215_data;
              v215_data.copy_from(ir1 + (192));
              (v215_data + (v197_data * v213_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v217_data;
              v217_data.copy_from(r0 + (288));
              tensorforge::intel_esimd::simd<float, 3> v220_data;
              v220_data.copy_from(ir1 + (32));
              (v220_data + (v217_data * v198_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v225_data;
              v225_data.copy_from(ir1 + (96));
              (v225_data + (v217_data * v203_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v230_data;
              v230_data.copy_from(ir1 + (160));
              (v230_data + (v217_data * v208_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v235_data;
              v235_data.copy_from(ir1 + (224));
              (v235_data + (v217_data * v213_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v237_data;
              v237_data.copy_from(r0 + (320));
              float v238_data = s0[5];
              tensorforge::intel_esimd::simd<float, 32> v240_data;
              v240_data.copy_from(ir1 + (0));
              (v240_data + (v237_data * v238_data)).copy_to(ir1 + (0));
              float v243_data = s0[13];
              tensorforge::intel_esimd::simd<float, 32> v245_data;
              v245_data.copy_from(ir1 + (64));
              (v245_data + (v237_data * v243_data)).copy_to(ir1 + (64));
              float v248_data = s0[21];
              tensorforge::intel_esimd::simd<float, 32> v250_data;
              v250_data.copy_from(ir1 + (128));
              (v250_data + (v237_data * v248_data)).copy_to(ir1 + (128));
              float v253_data = s0[29];
              tensorforge::intel_esimd::simd<float, 32> v255_data;
              v255_data.copy_from(ir1 + (192));
              (v255_data + (v237_data * v253_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v257_data;
              v257_data.copy_from(r0 + (352));
              tensorforge::intel_esimd::simd<float, 3> v260_data;
              v260_data.copy_from(ir1 + (32));
              (v260_data + (v257_data * v238_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v265_data;
              v265_data.copy_from(ir1 + (96));
              (v265_data + (v257_data * v243_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v270_data;
              v270_data.copy_from(ir1 + (160));
              (v270_data + (v257_data * v248_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v275_data;
              v275_data.copy_from(ir1 + (224));
              (v275_data + (v257_data * v253_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v277_data;
              v277_data.copy_from(r0 + (384));
              float v278_data = s0[6];
              tensorforge::intel_esimd::simd<float, 32> v280_data;
              v280_data.copy_from(ir1 + (0));
              (v280_data + (v277_data * v278_data)).copy_to(ir1 + (0));
              float v283_data = s0[14];
              tensorforge::intel_esimd::simd<float, 32> v285_data;
              v285_data.copy_from(ir1 + (64));
              (v285_data + (v277_data * v283_data)).copy_to(ir1 + (64));
              float v288_data = s0[22];
              tensorforge::intel_esimd::simd<float, 32> v290_data;
              v290_data.copy_from(ir1 + (128));
              (v290_data + (v277_data * v288_data)).copy_to(ir1 + (128));
              float v293_data = s0[30];
              tensorforge::intel_esimd::simd<float, 32> v295_data;
              v295_data.copy_from(ir1 + (192));
              (v295_data + (v277_data * v293_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v297_data;
              v297_data.copy_from(r0 + (416));
              tensorforge::intel_esimd::simd<float, 3> v300_data;
              v300_data.copy_from(ir1 + (32));
              (v300_data + (v297_data * v278_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v305_data;
              v305_data.copy_from(ir1 + (96));
              (v305_data + (v297_data * v283_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v310_data;
              v310_data.copy_from(ir1 + (160));
              (v310_data + (v297_data * v288_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v315_data;
              v315_data.copy_from(ir1 + (224));
              (v315_data + (v297_data * v293_data)).copy_to(ir1 + (224));
              tensorforge::intel_esimd::simd<float, 32> v317_data;
              v317_data.copy_from(r0 + (448));
              float v318_data = s0[7];
              tensorforge::intel_esimd::simd<float, 32> v320_data;
              v320_data.copy_from(ir1 + (0));
              (v320_data + (v317_data * v318_data)).copy_to(ir1 + (0));
              float v323_data = s0[15];
              tensorforge::intel_esimd::simd<float, 32> v325_data;
              v325_data.copy_from(ir1 + (64));
              (v325_data + (v317_data * v323_data)).copy_to(ir1 + (64));
              float v328_data = s0[23];
              tensorforge::intel_esimd::simd<float, 32> v330_data;
              v330_data.copy_from(ir1 + (128));
              (v330_data + (v317_data * v328_data)).copy_to(ir1 + (128));
              float v333_data = s0[31];
              tensorforge::intel_esimd::simd<float, 32> v335_data;
              v335_data.copy_from(ir1 + (192));
              (v335_data + (v317_data * v333_data)).copy_to(ir1 + (192));
              tensorforge::intel_esimd::simd<float, 3> v337_data;
              v337_data.copy_from(r0 + (480));
              tensorforge::intel_esimd::simd<float, 3> v340_data;
              v340_data.copy_from(ir1 + (32));
              (v340_data + (v337_data * v318_data)).copy_to(ir1 + (32));
              tensorforge::intel_esimd::simd<float, 3> v345_data;
              v345_data.copy_from(ir1 + (96));
              (v345_data + (v337_data * v323_data)).copy_to(ir1 + (96));
              tensorforge::intel_esimd::simd<float, 3> v350_data;
              v350_data.copy_from(ir1 + (160));
              (v350_data + (v337_data * v328_data)).copy_to(ir1 + (160));
              tensorforge::intel_esimd::simd<float, 3> v355_data;
              v355_data.copy_from(ir1 + (224));
              (v355_data + (v337_data * v333_data)).copy_to(ir1 + (224));
              #pragma unroll
              for (int32_t v357_n0 = 0; v357_n0 < 1; ++v357_n0) {
                int32_t v359_a = v357_n0 * 32;
                #pragma unroll
                for (int32_t v358_n1 = 0; v358_n1 < 4; ++v358_n1) {
                  int32_t v361_a = v359_a + (v358_n1 * 64);
                  tensorforge::intel_esimd::simd<float, 32> v362_data;
                  v362_data.copy_from(ir1 + (v361_a));
                  v362_data.copy_to(r1 + (v361_a));
                }
              }
              #pragma unroll
              for (int32_t v366_n1 = 0; v366_n1 < 4; ++v366_n1) {
                int32_t v368_a = 32 + (v366_n1 * 64);
                tensorforge::intel_esimd::simd<float, 3> v369_data;
                v369_data.copy_from(ir1 + (v368_a));
                v369_data.copy_to(r1 + (v368_a));
              }
              // glb_m0 = store{r>g}(r1);
              #pragma unroll
              for (int32_t v372_i0 = 0; v372_i0 < 1; ++v372_i0) {
                int32_t v374_a = v372_i0 * 32;
                #pragma unroll
                for (int32_t v373_i1 = 0; v373_i1 < 4; ++v373_i1) {
                  tensorforge::intel_esimd::simd<float, 32> v377_data;
                  v377_data.copy_from(r1 + ((v374_a + (v373_i1 * 64))));
                  v377_data.copy_to(glb_m0 + ((v374_a + (v373_i1 * 35))));
                }
              }
              #pragma unroll
              for (int32_t v382_i1 = 0; v382_i1 < 4; ++v382_i1) {
                tensorforge::intel_esimd::simd<float, 3> v385_data;
                v385_data.copy_from(r1 + ((32 + (v382_i1 * 64))));
                v385_data.copy_to(glb_m0 + ((32_i32 + (v382_i1 * 35))));
              }
            }
          }
        }
      });
    }
  });
}

