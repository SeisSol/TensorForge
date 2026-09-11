// === base name ===
kernel_aaf39a8409299473

// === header ===
void launcher_kernel_aaf39a8409299473(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_aaf39a8409299473(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 1, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_aaf39a8409299473(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  m5,  m5_extraOffset,  m6,  m6_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_aaf39a8409299473(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, const float* m5, size_t m5_extraOffset, const float* m6, size_t m6_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 32×16(32×16) {0..32}×{0..16} strided
        // m1 32×12(32×12) {0..32}×{0..12} strided
        // m2 12×16(12×16) {0..12}×{0..16} strided
        // m3 32×12(32×12) {0..32}×{0..12} strided
        // m4 12×8(12×8) {0..12}×{0..8} strided
        // m5 32×12(32×12) {0..32}×{0..12} strided
        // m6 12×8(12×8) {0..12}×{0..8} strided
        // m0 32×16(32×16) {0..32}×{0..16} strided({0..32}×{0..16})[0, 1] = m1 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m2 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[-1, 1]
        // m0 32×16(32×16) {0..32}×{0..16} strided({0..32}×{0..8})[0, 1] += m3 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m4 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        // m0 32×16(32×16) {0..32}×{0..16} strided({0..32}×{0..8})[0, 1] += m5 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m6 12×8(12×8) {0..12}×{0..8} strided({0..12}×{0..8})[-1, 1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t v0_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v0_batchId0 < numElements0; v0_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v1_ahead1 = v0_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v3_batchId1 = (v1_ahead1 < numElements0) ? v1_ahead1 : v0_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v0_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v0_batchId0 * 512 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v0_batchId0 * 384 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v0_batchId0 * 192 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v0_batchId0 * 384 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v0_batchId0 * 96 + 0 + m4_extraOffset];
              const float *const __restrict__ glb_m5 = &m5[v0_batchId0 * 384 + 0 + m5_extraOffset];
              const float *const __restrict__ glb_m6 = &m6[v0_batchId0 * 96 + 0 + m6_extraOffset];
              float r0[12]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v18_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v19_i0 = 0; v19_i0 < 1; ++v19_i0) {
                int32_t v25_lead = v18_lead + (v19_i0 * 32);
                #pragma unroll
                for (int32_t v20_i1 = 0; v20_i1 < 12; ++v20_i1) {
                  float v28_data = glb_m1[(v25_lead + (v20_i1 * 32))];
                  r0[(v19_i0 + v20_i1)] = v28_data;
                }
              }
              float r1[16]{};
              // r1 = load{g>r}(glb_m2);
              if (v18_lead < 12) {
                #pragma unroll
                for (int32_t v35_i1 = 0; v35_i1 < 16; ++v35_i1) {
                  float v43_data = glb_m2[(v18_lead + (v35_i1 * 12))];
                  r1[v35_i1] = v43_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              #pragma unroll
              for (int32_t v49_i0 = 0; v49_i0 < 1; ++v49_i0) {
                int32_t v55_lead = v18_lead + (v49_i0 * 32);
                #pragma unroll
                for (int32_t v50_i1 = 0; v50_i1 < 12; ++v50_i1) {
                  float v58_data = glb_m3[(v55_lead + (v50_i1 * 32))];
                  r3[(v49_i0 + v50_i1)] = v58_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m2););
              float r2[16]{};
              // r2 = +(r0 * r1) + None
              // [(0, 32), (0, 16)] [(0, 12)]
              float ir2[16]{};
              float v65_data = r0[0];
              float v66_data = r1[0];
              float v69_data = ir2[0];
              ir2[0] = (v69_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v66_data, 0))));
              float v72_data = r1[1];
              float v75_data = ir2[1];
              ir2[1] = (v75_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v72_data, 0))));
              float v78_data = r1[2];
              float v81_data = ir2[2];
              ir2[2] = (v81_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v78_data, 0))));
              float v84_data = r1[3];
              float v87_data = ir2[3];
              ir2[3] = (v87_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v84_data, 0))));
              float v90_data = r1[4];
              float v93_data = ir2[4];
              ir2[4] = (v93_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v90_data, 0))));
              float v96_data = r1[5];
              float v99_data = ir2[5];
              ir2[5] = (v99_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v96_data, 0))));
              float v102_data = r1[6];
              float v105_data = ir2[6];
              ir2[6] = (v105_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v102_data, 0))));
              float v108_data = r1[7];
              float v111_data = ir2[7];
              ir2[7] = (v111_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 0))));
              float v114_data = r1[8];
              float v117_data = ir2[8];
              ir2[8] = (v117_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 0))));
              float v120_data = r1[9];
              float v123_data = ir2[9];
              ir2[9] = (v123_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 0))));
              float v126_data = r1[10];
              float v129_data = ir2[10];
              ir2[10] = (v129_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 0))));
              float v132_data = r1[11];
              float v135_data = ir2[11];
              ir2[11] = (v135_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 0))));
              float v138_data = r1[12];
              float v141_data = ir2[12];
              ir2[12] = (v141_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 0))));
              float v144_data = r1[13];
              float v147_data = ir2[13];
              ir2[13] = (v147_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 0))));
              float v150_data = r1[14];
              float v153_data = ir2[14];
              ir2[14] = (v153_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 0))));
              float v156_data = r1[15];
              float v159_data = ir2[15];
              ir2[15] = (v159_data + (v65_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 0))));
              float v164_data = r0[1];
              float v168_data = ir2[0];
              ir2[0] = (v168_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v66_data, 1))));
              float v174_data = ir2[1];
              ir2[1] = (v174_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v72_data, 1))));
              float v180_data = ir2[2];
              ir2[2] = (v180_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v78_data, 1))));
              float v186_data = ir2[3];
              ir2[3] = (v186_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v84_data, 1))));
              float v192_data = ir2[4];
              ir2[4] = (v192_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v90_data, 1))));
              float v198_data = ir2[5];
              ir2[5] = (v198_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v96_data, 1))));
              float v204_data = ir2[6];
              ir2[6] = (v204_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v102_data, 1))));
              float v210_data = ir2[7];
              ir2[7] = (v210_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 1))));
              float v216_data = ir2[8];
              ir2[8] = (v216_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 1))));
              float v222_data = ir2[9];
              ir2[9] = (v222_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 1))));
              float v228_data = ir2[10];
              ir2[10] = (v228_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 1))));
              float v234_data = ir2[11];
              ir2[11] = (v234_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 1))));
              float v240_data = ir2[12];
              ir2[12] = (v240_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 1))));
              float v246_data = ir2[13];
              ir2[13] = (v246_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 1))));
              float v252_data = ir2[14];
              ir2[14] = (v252_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 1))));
              float v258_data = ir2[15];
              ir2[15] = (v258_data + (v164_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 1))));
              float v263_data = r0[2];
              float v267_data = ir2[0];
              ir2[0] = (v267_data + (v263_data * (sycl::group_broadcast(item.get_sub_group(), v66_data, 2))));
              float v273_data = ir2[1];
              ir2[1] = (v273_data + (v263_data * (sycl::group_broadcast(item.get_sub_group(), v72_data, 2))));
              float v279_data = ir2[2];
              ir2[2] = (v279_data + (v263_data * (sycl::group_broadcast(item.get_sub_group(), v78_data, 2))));
              float v285_data = ir2[3];
              ir2[3] = (v285_data + (v263_data * (sycl::group_broadcast(item.get_sub_group(), v84_data, 2))));
              float v291_data = ir2[4];
              ir2[4] = (v291_data + (v263_data * (sycl::group_broadcast(item.get_sub_group(), v90_data, 2))));
              float v297_data = ir2[5];
              ir2[5] = (v297_data + (v263_data * (sycl::group_broadcast(item.get_sub_group(), v96_data, 2))));
              float v303_data = ir2[6];
              ir2[6] = (v303_data + (v263_data * (sycl::group_broadcast(item.get_sub_group(), v102_data, 2))));
              float v309_data = ir2[7];
              ir2[7] = (v309_data + (v263_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 2))));
              float v315_data = ir2[8];
              ir2[8] = (v315_data + (v263_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 2))));
              float v321_data = ir2[9];
              ir2[9] = (v321_data + (v263_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 2))));
              float v327_data = ir2[10];
              ir2[10] = (v327_data + (v263_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 2))));
              float v333_data = ir2[11];
              ir2[11] = (v333_data + (v263_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 2))));
              float v339_data = ir2[12];
              ir2[12] = (v339_data + (v263_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 2))));
              float v345_data = ir2[13];
              ir2[13] = (v345_data + (v263_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 2))));
              float v351_data = ir2[14];
              ir2[14] = (v351_data + (v263_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 2))));
              float v357_data = ir2[15];
              ir2[15] = (v357_data + (v263_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 2))));
              float v362_data = r0[3];
              float v366_data = ir2[0];
              ir2[0] = (v366_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v66_data, 3))));
              float v372_data = ir2[1];
              ir2[1] = (v372_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v72_data, 3))));
              float v378_data = ir2[2];
              ir2[2] = (v378_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v78_data, 3))));
              float v384_data = ir2[3];
              ir2[3] = (v384_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v84_data, 3))));
              float v390_data = ir2[4];
              ir2[4] = (v390_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v90_data, 3))));
              float v396_data = ir2[5];
              ir2[5] = (v396_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v96_data, 3))));
              float v402_data = ir2[6];
              ir2[6] = (v402_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v102_data, 3))));
              float v408_data = ir2[7];
              ir2[7] = (v408_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 3))));
              float v414_data = ir2[8];
              ir2[8] = (v414_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 3))));
              float v420_data = ir2[9];
              ir2[9] = (v420_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 3))));
              float v426_data = ir2[10];
              ir2[10] = (v426_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 3))));
              float v432_data = ir2[11];
              ir2[11] = (v432_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 3))));
              float v438_data = ir2[12];
              ir2[12] = (v438_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 3))));
              float v444_data = ir2[13];
              ir2[13] = (v444_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 3))));
              float v450_data = ir2[14];
              ir2[14] = (v450_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 3))));
              float v456_data = ir2[15];
              ir2[15] = (v456_data + (v362_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 3))));
              float v461_data = r0[4];
              float v465_data = ir2[0];
              ir2[0] = (v465_data + (v461_data * (sycl::group_broadcast(item.get_sub_group(), v66_data, 4))));
              float v471_data = ir2[1];
              ir2[1] = (v471_data + (v461_data * (sycl::group_broadcast(item.get_sub_group(), v72_data, 4))));
              float v477_data = ir2[2];
              ir2[2] = (v477_data + (v461_data * (sycl::group_broadcast(item.get_sub_group(), v78_data, 4))));
              float v483_data = ir2[3];
              ir2[3] = (v483_data + (v461_data * (sycl::group_broadcast(item.get_sub_group(), v84_data, 4))));
              float v489_data = ir2[4];
              ir2[4] = (v489_data + (v461_data * (sycl::group_broadcast(item.get_sub_group(), v90_data, 4))));
              float v495_data = ir2[5];
              ir2[5] = (v495_data + (v461_data * (sycl::group_broadcast(item.get_sub_group(), v96_data, 4))));
              float v501_data = ir2[6];
              ir2[6] = (v501_data + (v461_data * (sycl::group_broadcast(item.get_sub_group(), v102_data, 4))));
              float v507_data = ir2[7];
              ir2[7] = (v507_data + (v461_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 4))));
              float v513_data = ir2[8];
              ir2[8] = (v513_data + (v461_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 4))));
              float v519_data = ir2[9];
              ir2[9] = (v519_data + (v461_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 4))));
              float v525_data = ir2[10];
              ir2[10] = (v525_data + (v461_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 4))));
              float v531_data = ir2[11];
              ir2[11] = (v531_data + (v461_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 4))));
              float v537_data = ir2[12];
              ir2[12] = (v537_data + (v461_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 4))));
              float v543_data = ir2[13];
              ir2[13] = (v543_data + (v461_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 4))));
              float v549_data = ir2[14];
              ir2[14] = (v549_data + (v461_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 4))));
              float v555_data = ir2[15];
              ir2[15] = (v555_data + (v461_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 4))));
              float v560_data = r0[5];
              float v564_data = ir2[0];
              ir2[0] = (v564_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v66_data, 5))));
              float v570_data = ir2[1];
              ir2[1] = (v570_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v72_data, 5))));
              float v576_data = ir2[2];
              ir2[2] = (v576_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v78_data, 5))));
              float v582_data = ir2[3];
              ir2[3] = (v582_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v84_data, 5))));
              float v588_data = ir2[4];
              ir2[4] = (v588_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v90_data, 5))));
              float v594_data = ir2[5];
              ir2[5] = (v594_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v96_data, 5))));
              float v600_data = ir2[6];
              ir2[6] = (v600_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v102_data, 5))));
              float v606_data = ir2[7];
              ir2[7] = (v606_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 5))));
              float v612_data = ir2[8];
              ir2[8] = (v612_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 5))));
              float v618_data = ir2[9];
              ir2[9] = (v618_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 5))));
              float v624_data = ir2[10];
              ir2[10] = (v624_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 5))));
              float v630_data = ir2[11];
              ir2[11] = (v630_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 5))));
              float v636_data = ir2[12];
              ir2[12] = (v636_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 5))));
              float v642_data = ir2[13];
              ir2[13] = (v642_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 5))));
              float v648_data = ir2[14];
              ir2[14] = (v648_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 5))));
              float v654_data = ir2[15];
              ir2[15] = (v654_data + (v560_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 5))));
              float v659_data = r0[6];
              float v663_data = ir2[0];
              ir2[0] = (v663_data + (v659_data * (sycl::group_broadcast(item.get_sub_group(), v66_data, 6))));
              float v669_data = ir2[1];
              ir2[1] = (v669_data + (v659_data * (sycl::group_broadcast(item.get_sub_group(), v72_data, 6))));
              float v675_data = ir2[2];
              ir2[2] = (v675_data + (v659_data * (sycl::group_broadcast(item.get_sub_group(), v78_data, 6))));
              float v681_data = ir2[3];
              ir2[3] = (v681_data + (v659_data * (sycl::group_broadcast(item.get_sub_group(), v84_data, 6))));
              float v687_data = ir2[4];
              ir2[4] = (v687_data + (v659_data * (sycl::group_broadcast(item.get_sub_group(), v90_data, 6))));
              float v693_data = ir2[5];
              ir2[5] = (v693_data + (v659_data * (sycl::group_broadcast(item.get_sub_group(), v96_data, 6))));
              float v699_data = ir2[6];
              ir2[6] = (v699_data + (v659_data * (sycl::group_broadcast(item.get_sub_group(), v102_data, 6))));
              float v705_data = ir2[7];
              ir2[7] = (v705_data + (v659_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 6))));
              float v711_data = ir2[8];
              ir2[8] = (v711_data + (v659_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 6))));
              float v717_data = ir2[9];
              ir2[9] = (v717_data + (v659_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 6))));
              float v723_data = ir2[10];
              ir2[10] = (v723_data + (v659_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 6))));
              float v729_data = ir2[11];
              ir2[11] = (v729_data + (v659_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 6))));
              float v735_data = ir2[12];
              ir2[12] = (v735_data + (v659_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 6))));
              float v741_data = ir2[13];
              ir2[13] = (v741_data + (v659_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 6))));
              float v747_data = ir2[14];
              ir2[14] = (v747_data + (v659_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 6))));
              float v753_data = ir2[15];
              ir2[15] = (v753_data + (v659_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 6))));
              float v758_data = r0[7];
              float v762_data = ir2[0];
              ir2[0] = (v762_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v66_data, 7))));
              float v768_data = ir2[1];
              ir2[1] = (v768_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v72_data, 7))));
              float v774_data = ir2[2];
              ir2[2] = (v774_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v78_data, 7))));
              float v780_data = ir2[3];
              ir2[3] = (v780_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v84_data, 7))));
              float v786_data = ir2[4];
              ir2[4] = (v786_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v90_data, 7))));
              float v792_data = ir2[5];
              ir2[5] = (v792_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v96_data, 7))));
              float v798_data = ir2[6];
              ir2[6] = (v798_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v102_data, 7))));
              float v804_data = ir2[7];
              ir2[7] = (v804_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 7))));
              float v810_data = ir2[8];
              ir2[8] = (v810_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 7))));
              float v816_data = ir2[9];
              ir2[9] = (v816_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 7))));
              float v822_data = ir2[10];
              ir2[10] = (v822_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 7))));
              float v828_data = ir2[11];
              ir2[11] = (v828_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 7))));
              float v834_data = ir2[12];
              ir2[12] = (v834_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 7))));
              float v840_data = ir2[13];
              ir2[13] = (v840_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 7))));
              float v846_data = ir2[14];
              ir2[14] = (v846_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 7))));
              float v852_data = ir2[15];
              ir2[15] = (v852_data + (v758_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 7))));
              float v857_data = r0[8];
              float v861_data = ir2[0];
              ir2[0] = (v861_data + (v857_data * (sycl::group_broadcast(item.get_sub_group(), v66_data, 8))));
              float v867_data = ir2[1];
              ir2[1] = (v867_data + (v857_data * (sycl::group_broadcast(item.get_sub_group(), v72_data, 8))));
              float v873_data = ir2[2];
              ir2[2] = (v873_data + (v857_data * (sycl::group_broadcast(item.get_sub_group(), v78_data, 8))));
              float v879_data = ir2[3];
              ir2[3] = (v879_data + (v857_data * (sycl::group_broadcast(item.get_sub_group(), v84_data, 8))));
              float v885_data = ir2[4];
              ir2[4] = (v885_data + (v857_data * (sycl::group_broadcast(item.get_sub_group(), v90_data, 8))));
              float v891_data = ir2[5];
              ir2[5] = (v891_data + (v857_data * (sycl::group_broadcast(item.get_sub_group(), v96_data, 8))));
              float v897_data = ir2[6];
              ir2[6] = (v897_data + (v857_data * (sycl::group_broadcast(item.get_sub_group(), v102_data, 8))));
              float v903_data = ir2[7];
              ir2[7] = (v903_data + (v857_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 8))));
              float v909_data = ir2[8];
              ir2[8] = (v909_data + (v857_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 8))));
              float v915_data = ir2[9];
              ir2[9] = (v915_data + (v857_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 8))));
              float v921_data = ir2[10];
              ir2[10] = (v921_data + (v857_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 8))));
              float v927_data = ir2[11];
              ir2[11] = (v927_data + (v857_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 8))));
              float v933_data = ir2[12];
              ir2[12] = (v933_data + (v857_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 8))));
              float v939_data = ir2[13];
              ir2[13] = (v939_data + (v857_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 8))));
              float v945_data = ir2[14];
              ir2[14] = (v945_data + (v857_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 8))));
              float v951_data = ir2[15];
              ir2[15] = (v951_data + (v857_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 8))));
              float v956_data = r0[9];
              float v960_data = ir2[0];
              ir2[0] = (v960_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v66_data, 9))));
              float v966_data = ir2[1];
              ir2[1] = (v966_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v72_data, 9))));
              float v972_data = ir2[2];
              ir2[2] = (v972_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v78_data, 9))));
              float v978_data = ir2[3];
              ir2[3] = (v978_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v84_data, 9))));
              float v984_data = ir2[4];
              ir2[4] = (v984_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v90_data, 9))));
              float v990_data = ir2[5];
              ir2[5] = (v990_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v96_data, 9))));
              float v996_data = ir2[6];
              ir2[6] = (v996_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v102_data, 9))));
              float v1002_data = ir2[7];
              ir2[7] = (v1002_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 9))));
              float v1008_data = ir2[8];
              ir2[8] = (v1008_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 9))));
              float v1014_data = ir2[9];
              ir2[9] = (v1014_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 9))));
              float v1020_data = ir2[10];
              ir2[10] = (v1020_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 9))));
              float v1026_data = ir2[11];
              ir2[11] = (v1026_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 9))));
              float v1032_data = ir2[12];
              ir2[12] = (v1032_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 9))));
              float v1038_data = ir2[13];
              ir2[13] = (v1038_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 9))));
              float v1044_data = ir2[14];
              ir2[14] = (v1044_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 9))));
              float v1050_data = ir2[15];
              ir2[15] = (v1050_data + (v956_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 9))));
              float v1055_data = r0[10];
              float v1059_data = ir2[0];
              ir2[0] = (v1059_data + (v1055_data * (sycl::group_broadcast(item.get_sub_group(), v66_data, 10))));
              float v1065_data = ir2[1];
              ir2[1] = (v1065_data + (v1055_data * (sycl::group_broadcast(item.get_sub_group(), v72_data, 10))));
              float v1071_data = ir2[2];
              ir2[2] = (v1071_data + (v1055_data * (sycl::group_broadcast(item.get_sub_group(), v78_data, 10))));
              float v1077_data = ir2[3];
              ir2[3] = (v1077_data + (v1055_data * (sycl::group_broadcast(item.get_sub_group(), v84_data, 10))));
              float v1083_data = ir2[4];
              ir2[4] = (v1083_data + (v1055_data * (sycl::group_broadcast(item.get_sub_group(), v90_data, 10))));
              float v1089_data = ir2[5];
              ir2[5] = (v1089_data + (v1055_data * (sycl::group_broadcast(item.get_sub_group(), v96_data, 10))));
              float v1095_data = ir2[6];
              ir2[6] = (v1095_data + (v1055_data * (sycl::group_broadcast(item.get_sub_group(), v102_data, 10))));
              float v1101_data = ir2[7];
              ir2[7] = (v1101_data + (v1055_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 10))));
              float v1107_data = ir2[8];
              ir2[8] = (v1107_data + (v1055_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 10))));
              float v1113_data = ir2[9];
              ir2[9] = (v1113_data + (v1055_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 10))));
              float v1119_data = ir2[10];
              ir2[10] = (v1119_data + (v1055_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 10))));
              float v1125_data = ir2[11];
              ir2[11] = (v1125_data + (v1055_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 10))));
              float v1131_data = ir2[12];
              ir2[12] = (v1131_data + (v1055_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 10))));
              float v1137_data = ir2[13];
              ir2[13] = (v1137_data + (v1055_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 10))));
              float v1143_data = ir2[14];
              ir2[14] = (v1143_data + (v1055_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 10))));
              float v1149_data = ir2[15];
              ir2[15] = (v1149_data + (v1055_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 10))));
              float v1154_data = r0[11];
              float v1158_data = ir2[0];
              ir2[0] = (v1158_data + (v1154_data * (sycl::group_broadcast(item.get_sub_group(), v66_data, 11))));
              float v1164_data = ir2[1];
              ir2[1] = (v1164_data + (v1154_data * (sycl::group_broadcast(item.get_sub_group(), v72_data, 11))));
              float v1170_data = ir2[2];
              ir2[2] = (v1170_data + (v1154_data * (sycl::group_broadcast(item.get_sub_group(), v78_data, 11))));
              float v1176_data = ir2[3];
              ir2[3] = (v1176_data + (v1154_data * (sycl::group_broadcast(item.get_sub_group(), v84_data, 11))));
              float v1182_data = ir2[4];
              ir2[4] = (v1182_data + (v1154_data * (sycl::group_broadcast(item.get_sub_group(), v90_data, 11))));
              float v1188_data = ir2[5];
              ir2[5] = (v1188_data + (v1154_data * (sycl::group_broadcast(item.get_sub_group(), v96_data, 11))));
              float v1194_data = ir2[6];
              ir2[6] = (v1194_data + (v1154_data * (sycl::group_broadcast(item.get_sub_group(), v102_data, 11))));
              float v1200_data = ir2[7];
              ir2[7] = (v1200_data + (v1154_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 11))));
              float v1206_data = ir2[8];
              ir2[8] = (v1206_data + (v1154_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 11))));
              float v1212_data = ir2[9];
              ir2[9] = (v1212_data + (v1154_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 11))));
              float v1218_data = ir2[10];
              ir2[10] = (v1218_data + (v1154_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 11))));
              float v1224_data = ir2[11];
              ir2[11] = (v1224_data + (v1154_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 11))));
              float v1230_data = ir2[12];
              ir2[12] = (v1230_data + (v1154_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 11))));
              float v1236_data = ir2[13];
              ir2[13] = (v1236_data + (v1154_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 11))));
              float v1242_data = ir2[14];
              ir2[14] = (v1242_data + (v1154_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 11))));
              float v1248_data = ir2[15];
              ir2[15] = (v1248_data + (v1154_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 11))));
              #pragma unroll
              for (int32_t v1253_n0 = 0; v1253_n0 < 1; ++v1253_n0) {
                #pragma unroll
                for (int32_t v1254_n1 = 0; v1254_n1 < 16; ++v1254_n1) {
                  int32_t v1255_a = v1253_n0 + v1254_n1;
                  float v1256_data = ir2[v1255_a];
                  r2[v1255_a] = v1256_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v1261_i0 = 0; v1261_i0 < 1; ++v1261_i0) {
                int32_t v1269_lead = v18_lead + (v1261_i0 * 32);
                #pragma unroll
                for (int32_t v1262_i1 = 0; v1262_i1 < 16; ++v1262_i1) {
                  float v1264_data = r2[(v1261_i0 + v1262_i1)];
                  glb_m0[(v1269_lead + (v1262_i1 * 32))] = v1264_data;
                }
              }
              float r4[8]{};
              // r4 = load{g>r}(glb_m4);
              if (v18_lead < 12) {
                #pragma unroll
                for (int32_t v1277_i1 = 0; v1277_i1 < 8; ++v1277_i1) {
                  float v1285_data = glb_m4[(v18_lead + (v1277_i1 * 12))];
                  r4[v1277_i1] = v1285_data;
                }
              }
              // wait(r3 = load{g>r}(glb_m3););
              float r5[8]{};
              // r5 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v1291_i0 = 0; v1291_i0 < 1; ++v1291_i0) {
                int32_t v1297_lead = v18_lead + (v1291_i0 * 32);
                #pragma unroll
                for (int32_t v1292_i1 = 0; v1292_i1 < 8; ++v1292_i1) {
                  float v1300_data = glb_m0[(v1297_lead + (v1292_i1 * 32))];
                  r5[(v1291_i0 + v1292_i1)] = v1300_data;
                }
              }
              // wait(r4 = load{g>r}(glb_m4););
              float r7[12]{};
              // r7 = load{g>r}(glb_m5);
              #pragma unroll
              for (int32_t v1306_i0 = 0; v1306_i0 < 1; ++v1306_i0) {
                int32_t v1312_lead = v18_lead + (v1306_i0 * 32);
                #pragma unroll
                for (int32_t v1307_i1 = 0; v1307_i1 < 12; ++v1307_i1) {
                  float v1315_data = glb_m5[(v1312_lead + (v1307_i1 * 32))];
                  r7[(v1306_i0 + v1307_i1)] = v1315_data;
                }
              }
              // wait(r5 = load{g>r}(glb_m0););
              float r6[8]{};
              // r6 = +(r3 * r4) + name: r5, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 8)] [(0, 12)]
              float ir6[8]{};
              float v1322_data = r3[0];
              float v1323_data = r4[0];
              float v1326_data = ir6[0];
              ir6[0] = (v1326_data + (v1322_data * (sycl::group_broadcast(item.get_sub_group(), v1323_data, 0))));
              float v1329_data = r4[1];
              float v1332_data = ir6[1];
              ir6[1] = (v1332_data + (v1322_data * (sycl::group_broadcast(item.get_sub_group(), v1329_data, 0))));
              float v1335_data = r4[2];
              float v1338_data = ir6[2];
              ir6[2] = (v1338_data + (v1322_data * (sycl::group_broadcast(item.get_sub_group(), v1335_data, 0))));
              float v1341_data = r4[3];
              float v1344_data = ir6[3];
              ir6[3] = (v1344_data + (v1322_data * (sycl::group_broadcast(item.get_sub_group(), v1341_data, 0))));
              float v1347_data = r4[4];
              float v1350_data = ir6[4];
              ir6[4] = (v1350_data + (v1322_data * (sycl::group_broadcast(item.get_sub_group(), v1347_data, 0))));
              float v1353_data = r4[5];
              float v1356_data = ir6[5];
              ir6[5] = (v1356_data + (v1322_data * (sycl::group_broadcast(item.get_sub_group(), v1353_data, 0))));
              float v1359_data = r4[6];
              float v1362_data = ir6[6];
              ir6[6] = (v1362_data + (v1322_data * (sycl::group_broadcast(item.get_sub_group(), v1359_data, 0))));
              float v1365_data = r4[7];
              float v1368_data = ir6[7];
              ir6[7] = (v1368_data + (v1322_data * (sycl::group_broadcast(item.get_sub_group(), v1365_data, 0))));
              float v1373_data = r3[1];
              float v1377_data = ir6[0];
              ir6[0] = (v1377_data + (v1373_data * (sycl::group_broadcast(item.get_sub_group(), v1323_data, 1))));
              float v1383_data = ir6[1];
              ir6[1] = (v1383_data + (v1373_data * (sycl::group_broadcast(item.get_sub_group(), v1329_data, 1))));
              float v1389_data = ir6[2];
              ir6[2] = (v1389_data + (v1373_data * (sycl::group_broadcast(item.get_sub_group(), v1335_data, 1))));
              float v1395_data = ir6[3];
              ir6[3] = (v1395_data + (v1373_data * (sycl::group_broadcast(item.get_sub_group(), v1341_data, 1))));
              float v1401_data = ir6[4];
              ir6[4] = (v1401_data + (v1373_data * (sycl::group_broadcast(item.get_sub_group(), v1347_data, 1))));
              float v1407_data = ir6[5];
              ir6[5] = (v1407_data + (v1373_data * (sycl::group_broadcast(item.get_sub_group(), v1353_data, 1))));
              float v1413_data = ir6[6];
              ir6[6] = (v1413_data + (v1373_data * (sycl::group_broadcast(item.get_sub_group(), v1359_data, 1))));
              float v1419_data = ir6[7];
              ir6[7] = (v1419_data + (v1373_data * (sycl::group_broadcast(item.get_sub_group(), v1365_data, 1))));
              float v1424_data = r3[2];
              float v1428_data = ir6[0];
              ir6[0] = (v1428_data + (v1424_data * (sycl::group_broadcast(item.get_sub_group(), v1323_data, 2))));
              float v1434_data = ir6[1];
              ir6[1] = (v1434_data + (v1424_data * (sycl::group_broadcast(item.get_sub_group(), v1329_data, 2))));
              float v1440_data = ir6[2];
              ir6[2] = (v1440_data + (v1424_data * (sycl::group_broadcast(item.get_sub_group(), v1335_data, 2))));
              float v1446_data = ir6[3];
              ir6[3] = (v1446_data + (v1424_data * (sycl::group_broadcast(item.get_sub_group(), v1341_data, 2))));
              float v1452_data = ir6[4];
              ir6[4] = (v1452_data + (v1424_data * (sycl::group_broadcast(item.get_sub_group(), v1347_data, 2))));
              float v1458_data = ir6[5];
              ir6[5] = (v1458_data + (v1424_data * (sycl::group_broadcast(item.get_sub_group(), v1353_data, 2))));
              float v1464_data = ir6[6];
              ir6[6] = (v1464_data + (v1424_data * (sycl::group_broadcast(item.get_sub_group(), v1359_data, 2))));
              float v1470_data = ir6[7];
              ir6[7] = (v1470_data + (v1424_data * (sycl::group_broadcast(item.get_sub_group(), v1365_data, 2))));
              float v1475_data = r3[3];
              float v1479_data = ir6[0];
              ir6[0] = (v1479_data + (v1475_data * (sycl::group_broadcast(item.get_sub_group(), v1323_data, 3))));
              float v1485_data = ir6[1];
              ir6[1] = (v1485_data + (v1475_data * (sycl::group_broadcast(item.get_sub_group(), v1329_data, 3))));
              float v1491_data = ir6[2];
              ir6[2] = (v1491_data + (v1475_data * (sycl::group_broadcast(item.get_sub_group(), v1335_data, 3))));
              float v1497_data = ir6[3];
              ir6[3] = (v1497_data + (v1475_data * (sycl::group_broadcast(item.get_sub_group(), v1341_data, 3))));
              float v1503_data = ir6[4];
              ir6[4] = (v1503_data + (v1475_data * (sycl::group_broadcast(item.get_sub_group(), v1347_data, 3))));
              float v1509_data = ir6[5];
              ir6[5] = (v1509_data + (v1475_data * (sycl::group_broadcast(item.get_sub_group(), v1353_data, 3))));
              float v1515_data = ir6[6];
              ir6[6] = (v1515_data + (v1475_data * (sycl::group_broadcast(item.get_sub_group(), v1359_data, 3))));
              float v1521_data = ir6[7];
              ir6[7] = (v1521_data + (v1475_data * (sycl::group_broadcast(item.get_sub_group(), v1365_data, 3))));
              float v1526_data = r3[4];
              float v1530_data = ir6[0];
              ir6[0] = (v1530_data + (v1526_data * (sycl::group_broadcast(item.get_sub_group(), v1323_data, 4))));
              float v1536_data = ir6[1];
              ir6[1] = (v1536_data + (v1526_data * (sycl::group_broadcast(item.get_sub_group(), v1329_data, 4))));
              float v1542_data = ir6[2];
              ir6[2] = (v1542_data + (v1526_data * (sycl::group_broadcast(item.get_sub_group(), v1335_data, 4))));
              float v1548_data = ir6[3];
              ir6[3] = (v1548_data + (v1526_data * (sycl::group_broadcast(item.get_sub_group(), v1341_data, 4))));
              float v1554_data = ir6[4];
              ir6[4] = (v1554_data + (v1526_data * (sycl::group_broadcast(item.get_sub_group(), v1347_data, 4))));
              float v1560_data = ir6[5];
              ir6[5] = (v1560_data + (v1526_data * (sycl::group_broadcast(item.get_sub_group(), v1353_data, 4))));
              float v1566_data = ir6[6];
              ir6[6] = (v1566_data + (v1526_data * (sycl::group_broadcast(item.get_sub_group(), v1359_data, 4))));
              float v1572_data = ir6[7];
              ir6[7] = (v1572_data + (v1526_data * (sycl::group_broadcast(item.get_sub_group(), v1365_data, 4))));
              float v1577_data = r3[5];
              float v1581_data = ir6[0];
              ir6[0] = (v1581_data + (v1577_data * (sycl::group_broadcast(item.get_sub_group(), v1323_data, 5))));
              float v1587_data = ir6[1];
              ir6[1] = (v1587_data + (v1577_data * (sycl::group_broadcast(item.get_sub_group(), v1329_data, 5))));
              float v1593_data = ir6[2];
              ir6[2] = (v1593_data + (v1577_data * (sycl::group_broadcast(item.get_sub_group(), v1335_data, 5))));
              float v1599_data = ir6[3];
              ir6[3] = (v1599_data + (v1577_data * (sycl::group_broadcast(item.get_sub_group(), v1341_data, 5))));
              float v1605_data = ir6[4];
              ir6[4] = (v1605_data + (v1577_data * (sycl::group_broadcast(item.get_sub_group(), v1347_data, 5))));
              float v1611_data = ir6[5];
              ir6[5] = (v1611_data + (v1577_data * (sycl::group_broadcast(item.get_sub_group(), v1353_data, 5))));
              float v1617_data = ir6[6];
              ir6[6] = (v1617_data + (v1577_data * (sycl::group_broadcast(item.get_sub_group(), v1359_data, 5))));
              float v1623_data = ir6[7];
              ir6[7] = (v1623_data + (v1577_data * (sycl::group_broadcast(item.get_sub_group(), v1365_data, 5))));
              float v1628_data = r3[6];
              float v1632_data = ir6[0];
              ir6[0] = (v1632_data + (v1628_data * (sycl::group_broadcast(item.get_sub_group(), v1323_data, 6))));
              float v1638_data = ir6[1];
              ir6[1] = (v1638_data + (v1628_data * (sycl::group_broadcast(item.get_sub_group(), v1329_data, 6))));
              float v1644_data = ir6[2];
              ir6[2] = (v1644_data + (v1628_data * (sycl::group_broadcast(item.get_sub_group(), v1335_data, 6))));
              float v1650_data = ir6[3];
              ir6[3] = (v1650_data + (v1628_data * (sycl::group_broadcast(item.get_sub_group(), v1341_data, 6))));
              float v1656_data = ir6[4];
              ir6[4] = (v1656_data + (v1628_data * (sycl::group_broadcast(item.get_sub_group(), v1347_data, 6))));
              float v1662_data = ir6[5];
              ir6[5] = (v1662_data + (v1628_data * (sycl::group_broadcast(item.get_sub_group(), v1353_data, 6))));
              float v1668_data = ir6[6];
              ir6[6] = (v1668_data + (v1628_data * (sycl::group_broadcast(item.get_sub_group(), v1359_data, 6))));
              float v1674_data = ir6[7];
              ir6[7] = (v1674_data + (v1628_data * (sycl::group_broadcast(item.get_sub_group(), v1365_data, 6))));
              float v1679_data = r3[7];
              float v1683_data = ir6[0];
              ir6[0] = (v1683_data + (v1679_data * (sycl::group_broadcast(item.get_sub_group(), v1323_data, 7))));
              float v1689_data = ir6[1];
              ir6[1] = (v1689_data + (v1679_data * (sycl::group_broadcast(item.get_sub_group(), v1329_data, 7))));
              float v1695_data = ir6[2];
              ir6[2] = (v1695_data + (v1679_data * (sycl::group_broadcast(item.get_sub_group(), v1335_data, 7))));
              float v1701_data = ir6[3];
              ir6[3] = (v1701_data + (v1679_data * (sycl::group_broadcast(item.get_sub_group(), v1341_data, 7))));
              float v1707_data = ir6[4];
              ir6[4] = (v1707_data + (v1679_data * (sycl::group_broadcast(item.get_sub_group(), v1347_data, 7))));
              float v1713_data = ir6[5];
              ir6[5] = (v1713_data + (v1679_data * (sycl::group_broadcast(item.get_sub_group(), v1353_data, 7))));
              float v1719_data = ir6[6];
              ir6[6] = (v1719_data + (v1679_data * (sycl::group_broadcast(item.get_sub_group(), v1359_data, 7))));
              float v1725_data = ir6[7];
              ir6[7] = (v1725_data + (v1679_data * (sycl::group_broadcast(item.get_sub_group(), v1365_data, 7))));
              float v1730_data = r3[8];
              float v1734_data = ir6[0];
              ir6[0] = (v1734_data + (v1730_data * (sycl::group_broadcast(item.get_sub_group(), v1323_data, 8))));
              float v1740_data = ir6[1];
              ir6[1] = (v1740_data + (v1730_data * (sycl::group_broadcast(item.get_sub_group(), v1329_data, 8))));
              float v1746_data = ir6[2];
              ir6[2] = (v1746_data + (v1730_data * (sycl::group_broadcast(item.get_sub_group(), v1335_data, 8))));
              float v1752_data = ir6[3];
              ir6[3] = (v1752_data + (v1730_data * (sycl::group_broadcast(item.get_sub_group(), v1341_data, 8))));
              float v1758_data = ir6[4];
              ir6[4] = (v1758_data + (v1730_data * (sycl::group_broadcast(item.get_sub_group(), v1347_data, 8))));
              float v1764_data = ir6[5];
              ir6[5] = (v1764_data + (v1730_data * (sycl::group_broadcast(item.get_sub_group(), v1353_data, 8))));
              float v1770_data = ir6[6];
              ir6[6] = (v1770_data + (v1730_data * (sycl::group_broadcast(item.get_sub_group(), v1359_data, 8))));
              float v1776_data = ir6[7];
              ir6[7] = (v1776_data + (v1730_data * (sycl::group_broadcast(item.get_sub_group(), v1365_data, 8))));
              float v1781_data = r3[9];
              float v1785_data = ir6[0];
              ir6[0] = (v1785_data + (v1781_data * (sycl::group_broadcast(item.get_sub_group(), v1323_data, 9))));
              float v1791_data = ir6[1];
              ir6[1] = (v1791_data + (v1781_data * (sycl::group_broadcast(item.get_sub_group(), v1329_data, 9))));
              float v1797_data = ir6[2];
              ir6[2] = (v1797_data + (v1781_data * (sycl::group_broadcast(item.get_sub_group(), v1335_data, 9))));
              float v1803_data = ir6[3];
              ir6[3] = (v1803_data + (v1781_data * (sycl::group_broadcast(item.get_sub_group(), v1341_data, 9))));
              float v1809_data = ir6[4];
              ir6[4] = (v1809_data + (v1781_data * (sycl::group_broadcast(item.get_sub_group(), v1347_data, 9))));
              float v1815_data = ir6[5];
              ir6[5] = (v1815_data + (v1781_data * (sycl::group_broadcast(item.get_sub_group(), v1353_data, 9))));
              float v1821_data = ir6[6];
              ir6[6] = (v1821_data + (v1781_data * (sycl::group_broadcast(item.get_sub_group(), v1359_data, 9))));
              float v1827_data = ir6[7];
              ir6[7] = (v1827_data + (v1781_data * (sycl::group_broadcast(item.get_sub_group(), v1365_data, 9))));
              float v1832_data = r3[10];
              float v1836_data = ir6[0];
              ir6[0] = (v1836_data + (v1832_data * (sycl::group_broadcast(item.get_sub_group(), v1323_data, 10))));
              float v1842_data = ir6[1];
              ir6[1] = (v1842_data + (v1832_data * (sycl::group_broadcast(item.get_sub_group(), v1329_data, 10))));
              float v1848_data = ir6[2];
              ir6[2] = (v1848_data + (v1832_data * (sycl::group_broadcast(item.get_sub_group(), v1335_data, 10))));
              float v1854_data = ir6[3];
              ir6[3] = (v1854_data + (v1832_data * (sycl::group_broadcast(item.get_sub_group(), v1341_data, 10))));
              float v1860_data = ir6[4];
              ir6[4] = (v1860_data + (v1832_data * (sycl::group_broadcast(item.get_sub_group(), v1347_data, 10))));
              float v1866_data = ir6[5];
              ir6[5] = (v1866_data + (v1832_data * (sycl::group_broadcast(item.get_sub_group(), v1353_data, 10))));
              float v1872_data = ir6[6];
              ir6[6] = (v1872_data + (v1832_data * (sycl::group_broadcast(item.get_sub_group(), v1359_data, 10))));
              float v1878_data = ir6[7];
              ir6[7] = (v1878_data + (v1832_data * (sycl::group_broadcast(item.get_sub_group(), v1365_data, 10))));
              float v1883_data = r3[11];
              float v1887_data = ir6[0];
              ir6[0] = (v1887_data + (v1883_data * (sycl::group_broadcast(item.get_sub_group(), v1323_data, 11))));
              float v1893_data = ir6[1];
              ir6[1] = (v1893_data + (v1883_data * (sycl::group_broadcast(item.get_sub_group(), v1329_data, 11))));
              float v1899_data = ir6[2];
              ir6[2] = (v1899_data + (v1883_data * (sycl::group_broadcast(item.get_sub_group(), v1335_data, 11))));
              float v1905_data = ir6[3];
              ir6[3] = (v1905_data + (v1883_data * (sycl::group_broadcast(item.get_sub_group(), v1341_data, 11))));
              float v1911_data = ir6[4];
              ir6[4] = (v1911_data + (v1883_data * (sycl::group_broadcast(item.get_sub_group(), v1347_data, 11))));
              float v1917_data = ir6[5];
              ir6[5] = (v1917_data + (v1883_data * (sycl::group_broadcast(item.get_sub_group(), v1353_data, 11))));
              float v1923_data = ir6[6];
              ir6[6] = (v1923_data + (v1883_data * (sycl::group_broadcast(item.get_sub_group(), v1359_data, 11))));
              float v1929_data = ir6[7];
              ir6[7] = (v1929_data + (v1883_data * (sycl::group_broadcast(item.get_sub_group(), v1365_data, 11))));
              #pragma unroll
              for (int32_t v1934_n0 = 0; v1934_n0 < 1; ++v1934_n0) {
                #pragma unroll
                for (int32_t v1935_n1 = 0; v1935_n1 < 8; ++v1935_n1) {
                  int32_t v1936_a = v1934_n0 + v1935_n1;
                  float v1937_data = ir6[v1936_a];
                  float v1939_data = r5[v1936_a];
                  r6[v1936_a] = (v1939_data + v1937_data);
                }
              }
              // glb_m0 = store{r>g}(r6);
              #pragma unroll
              for (int32_t v1945_i0 = 0; v1945_i0 < 1; ++v1945_i0) {
                int32_t v1953_lead = v18_lead + (v1945_i0 * 32);
                #pragma unroll
                for (int32_t v1946_i1 = 0; v1946_i1 < 8; ++v1946_i1) {
                  float v1948_data = r6[(v1945_i0 + v1946_i1)];
                  glb_m0[(v1953_lead + (v1946_i1 * 32))] = v1948_data;
                }
              }
              float r8[8]{};
              // r8 = load{g>r}(glb_m6);
              if (v18_lead < 12) {
                #pragma unroll
                for (int32_t v1961_i1 = 0; v1961_i1 < 8; ++v1961_i1) {
                  float v1969_data = glb_m6[(v18_lead + (v1961_i1 * 12))];
                  r8[v1961_i1] = v1969_data;
                }
              }
              // wait(r7 = load{g>r}(glb_m5););
              float r9[8]{};
              // r9 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v1975_i0 = 0; v1975_i0 < 1; ++v1975_i0) {
                int32_t v1981_lead = v18_lead + (v1975_i0 * 32);
                #pragma unroll
                for (int32_t v1976_i1 = 0; v1976_i1 < 8; ++v1976_i1) {
                  float v1985_data = glb_m0[(v1981_lead + ((v1976_i1 + 8) * 32))];
                  r9[(v1975_i0 + v1976_i1)] = v1985_data;
                }
              }
              // wait(r8 = load{g>r}(glb_m6););
              // wait(r9 = load{g>r}(glb_m0););
              float r10[8]{};
              // r10 = +(r7 * r8) + name: r9, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 8)] [(0, 12)]
              float ir10[8]{};
              float v1992_data = r7[0];
              float v1993_data = r8[0];
              float v1996_data = ir10[0];
              ir10[0] = (v1996_data + (v1992_data * (sycl::group_broadcast(item.get_sub_group(), v1993_data, 0))));
              float v1999_data = r8[1];
              float v2002_data = ir10[1];
              ir10[1] = (v2002_data + (v1992_data * (sycl::group_broadcast(item.get_sub_group(), v1999_data, 0))));
              float v2005_data = r8[2];
              float v2008_data = ir10[2];
              ir10[2] = (v2008_data + (v1992_data * (sycl::group_broadcast(item.get_sub_group(), v2005_data, 0))));
              float v2011_data = r8[3];
              float v2014_data = ir10[3];
              ir10[3] = (v2014_data + (v1992_data * (sycl::group_broadcast(item.get_sub_group(), v2011_data, 0))));
              float v2017_data = r8[4];
              float v2020_data = ir10[4];
              ir10[4] = (v2020_data + (v1992_data * (sycl::group_broadcast(item.get_sub_group(), v2017_data, 0))));
              float v2023_data = r8[5];
              float v2026_data = ir10[5];
              ir10[5] = (v2026_data + (v1992_data * (sycl::group_broadcast(item.get_sub_group(), v2023_data, 0))));
              float v2029_data = r8[6];
              float v2032_data = ir10[6];
              ir10[6] = (v2032_data + (v1992_data * (sycl::group_broadcast(item.get_sub_group(), v2029_data, 0))));
              float v2035_data = r8[7];
              float v2038_data = ir10[7];
              ir10[7] = (v2038_data + (v1992_data * (sycl::group_broadcast(item.get_sub_group(), v2035_data, 0))));
              float v2043_data = r7[1];
              float v2047_data = ir10[0];
              ir10[0] = (v2047_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v1993_data, 1))));
              float v2053_data = ir10[1];
              ir10[1] = (v2053_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v1999_data, 1))));
              float v2059_data = ir10[2];
              ir10[2] = (v2059_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v2005_data, 1))));
              float v2065_data = ir10[3];
              ir10[3] = (v2065_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v2011_data, 1))));
              float v2071_data = ir10[4];
              ir10[4] = (v2071_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v2017_data, 1))));
              float v2077_data = ir10[5];
              ir10[5] = (v2077_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v2023_data, 1))));
              float v2083_data = ir10[6];
              ir10[6] = (v2083_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v2029_data, 1))));
              float v2089_data = ir10[7];
              ir10[7] = (v2089_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v2035_data, 1))));
              float v2094_data = r7[2];
              float v2098_data = ir10[0];
              ir10[0] = (v2098_data + (v2094_data * (sycl::group_broadcast(item.get_sub_group(), v1993_data, 2))));
              float v2104_data = ir10[1];
              ir10[1] = (v2104_data + (v2094_data * (sycl::group_broadcast(item.get_sub_group(), v1999_data, 2))));
              float v2110_data = ir10[2];
              ir10[2] = (v2110_data + (v2094_data * (sycl::group_broadcast(item.get_sub_group(), v2005_data, 2))));
              float v2116_data = ir10[3];
              ir10[3] = (v2116_data + (v2094_data * (sycl::group_broadcast(item.get_sub_group(), v2011_data, 2))));
              float v2122_data = ir10[4];
              ir10[4] = (v2122_data + (v2094_data * (sycl::group_broadcast(item.get_sub_group(), v2017_data, 2))));
              float v2128_data = ir10[5];
              ir10[5] = (v2128_data + (v2094_data * (sycl::group_broadcast(item.get_sub_group(), v2023_data, 2))));
              float v2134_data = ir10[6];
              ir10[6] = (v2134_data + (v2094_data * (sycl::group_broadcast(item.get_sub_group(), v2029_data, 2))));
              float v2140_data = ir10[7];
              ir10[7] = (v2140_data + (v2094_data * (sycl::group_broadcast(item.get_sub_group(), v2035_data, 2))));
              float v2145_data = r7[3];
              float v2149_data = ir10[0];
              ir10[0] = (v2149_data + (v2145_data * (sycl::group_broadcast(item.get_sub_group(), v1993_data, 3))));
              float v2155_data = ir10[1];
              ir10[1] = (v2155_data + (v2145_data * (sycl::group_broadcast(item.get_sub_group(), v1999_data, 3))));
              float v2161_data = ir10[2];
              ir10[2] = (v2161_data + (v2145_data * (sycl::group_broadcast(item.get_sub_group(), v2005_data, 3))));
              float v2167_data = ir10[3];
              ir10[3] = (v2167_data + (v2145_data * (sycl::group_broadcast(item.get_sub_group(), v2011_data, 3))));
              float v2173_data = ir10[4];
              ir10[4] = (v2173_data + (v2145_data * (sycl::group_broadcast(item.get_sub_group(), v2017_data, 3))));
              float v2179_data = ir10[5];
              ir10[5] = (v2179_data + (v2145_data * (sycl::group_broadcast(item.get_sub_group(), v2023_data, 3))));
              float v2185_data = ir10[6];
              ir10[6] = (v2185_data + (v2145_data * (sycl::group_broadcast(item.get_sub_group(), v2029_data, 3))));
              float v2191_data = ir10[7];
              ir10[7] = (v2191_data + (v2145_data * (sycl::group_broadcast(item.get_sub_group(), v2035_data, 3))));
              float v2196_data = r7[4];
              float v2200_data = ir10[0];
              ir10[0] = (v2200_data + (v2196_data * (sycl::group_broadcast(item.get_sub_group(), v1993_data, 4))));
              float v2206_data = ir10[1];
              ir10[1] = (v2206_data + (v2196_data * (sycl::group_broadcast(item.get_sub_group(), v1999_data, 4))));
              float v2212_data = ir10[2];
              ir10[2] = (v2212_data + (v2196_data * (sycl::group_broadcast(item.get_sub_group(), v2005_data, 4))));
              float v2218_data = ir10[3];
              ir10[3] = (v2218_data + (v2196_data * (sycl::group_broadcast(item.get_sub_group(), v2011_data, 4))));
              float v2224_data = ir10[4];
              ir10[4] = (v2224_data + (v2196_data * (sycl::group_broadcast(item.get_sub_group(), v2017_data, 4))));
              float v2230_data = ir10[5];
              ir10[5] = (v2230_data + (v2196_data * (sycl::group_broadcast(item.get_sub_group(), v2023_data, 4))));
              float v2236_data = ir10[6];
              ir10[6] = (v2236_data + (v2196_data * (sycl::group_broadcast(item.get_sub_group(), v2029_data, 4))));
              float v2242_data = ir10[7];
              ir10[7] = (v2242_data + (v2196_data * (sycl::group_broadcast(item.get_sub_group(), v2035_data, 4))));
              float v2247_data = r7[5];
              float v2251_data = ir10[0];
              ir10[0] = (v2251_data + (v2247_data * (sycl::group_broadcast(item.get_sub_group(), v1993_data, 5))));
              float v2257_data = ir10[1];
              ir10[1] = (v2257_data + (v2247_data * (sycl::group_broadcast(item.get_sub_group(), v1999_data, 5))));
              float v2263_data = ir10[2];
              ir10[2] = (v2263_data + (v2247_data * (sycl::group_broadcast(item.get_sub_group(), v2005_data, 5))));
              float v2269_data = ir10[3];
              ir10[3] = (v2269_data + (v2247_data * (sycl::group_broadcast(item.get_sub_group(), v2011_data, 5))));
              float v2275_data = ir10[4];
              ir10[4] = (v2275_data + (v2247_data * (sycl::group_broadcast(item.get_sub_group(), v2017_data, 5))));
              float v2281_data = ir10[5];
              ir10[5] = (v2281_data + (v2247_data * (sycl::group_broadcast(item.get_sub_group(), v2023_data, 5))));
              float v2287_data = ir10[6];
              ir10[6] = (v2287_data + (v2247_data * (sycl::group_broadcast(item.get_sub_group(), v2029_data, 5))));
              float v2293_data = ir10[7];
              ir10[7] = (v2293_data + (v2247_data * (sycl::group_broadcast(item.get_sub_group(), v2035_data, 5))));
              float v2298_data = r7[6];
              float v2302_data = ir10[0];
              ir10[0] = (v2302_data + (v2298_data * (sycl::group_broadcast(item.get_sub_group(), v1993_data, 6))));
              float v2308_data = ir10[1];
              ir10[1] = (v2308_data + (v2298_data * (sycl::group_broadcast(item.get_sub_group(), v1999_data, 6))));
              float v2314_data = ir10[2];
              ir10[2] = (v2314_data + (v2298_data * (sycl::group_broadcast(item.get_sub_group(), v2005_data, 6))));
              float v2320_data = ir10[3];
              ir10[3] = (v2320_data + (v2298_data * (sycl::group_broadcast(item.get_sub_group(), v2011_data, 6))));
              float v2326_data = ir10[4];
              ir10[4] = (v2326_data + (v2298_data * (sycl::group_broadcast(item.get_sub_group(), v2017_data, 6))));
              float v2332_data = ir10[5];
              ir10[5] = (v2332_data + (v2298_data * (sycl::group_broadcast(item.get_sub_group(), v2023_data, 6))));
              float v2338_data = ir10[6];
              ir10[6] = (v2338_data + (v2298_data * (sycl::group_broadcast(item.get_sub_group(), v2029_data, 6))));
              float v2344_data = ir10[7];
              ir10[7] = (v2344_data + (v2298_data * (sycl::group_broadcast(item.get_sub_group(), v2035_data, 6))));
              float v2349_data = r7[7];
              float v2353_data = ir10[0];
              ir10[0] = (v2353_data + (v2349_data * (sycl::group_broadcast(item.get_sub_group(), v1993_data, 7))));
              float v2359_data = ir10[1];
              ir10[1] = (v2359_data + (v2349_data * (sycl::group_broadcast(item.get_sub_group(), v1999_data, 7))));
              float v2365_data = ir10[2];
              ir10[2] = (v2365_data + (v2349_data * (sycl::group_broadcast(item.get_sub_group(), v2005_data, 7))));
              float v2371_data = ir10[3];
              ir10[3] = (v2371_data + (v2349_data * (sycl::group_broadcast(item.get_sub_group(), v2011_data, 7))));
              float v2377_data = ir10[4];
              ir10[4] = (v2377_data + (v2349_data * (sycl::group_broadcast(item.get_sub_group(), v2017_data, 7))));
              float v2383_data = ir10[5];
              ir10[5] = (v2383_data + (v2349_data * (sycl::group_broadcast(item.get_sub_group(), v2023_data, 7))));
              float v2389_data = ir10[6];
              ir10[6] = (v2389_data + (v2349_data * (sycl::group_broadcast(item.get_sub_group(), v2029_data, 7))));
              float v2395_data = ir10[7];
              ir10[7] = (v2395_data + (v2349_data * (sycl::group_broadcast(item.get_sub_group(), v2035_data, 7))));
              float v2400_data = r7[8];
              float v2404_data = ir10[0];
              ir10[0] = (v2404_data + (v2400_data * (sycl::group_broadcast(item.get_sub_group(), v1993_data, 8))));
              float v2410_data = ir10[1];
              ir10[1] = (v2410_data + (v2400_data * (sycl::group_broadcast(item.get_sub_group(), v1999_data, 8))));
              float v2416_data = ir10[2];
              ir10[2] = (v2416_data + (v2400_data * (sycl::group_broadcast(item.get_sub_group(), v2005_data, 8))));
              float v2422_data = ir10[3];
              ir10[3] = (v2422_data + (v2400_data * (sycl::group_broadcast(item.get_sub_group(), v2011_data, 8))));
              float v2428_data = ir10[4];
              ir10[4] = (v2428_data + (v2400_data * (sycl::group_broadcast(item.get_sub_group(), v2017_data, 8))));
              float v2434_data = ir10[5];
              ir10[5] = (v2434_data + (v2400_data * (sycl::group_broadcast(item.get_sub_group(), v2023_data, 8))));
              float v2440_data = ir10[6];
              ir10[6] = (v2440_data + (v2400_data * (sycl::group_broadcast(item.get_sub_group(), v2029_data, 8))));
              float v2446_data = ir10[7];
              ir10[7] = (v2446_data + (v2400_data * (sycl::group_broadcast(item.get_sub_group(), v2035_data, 8))));
              float v2451_data = r7[9];
              float v2455_data = ir10[0];
              ir10[0] = (v2455_data + (v2451_data * (sycl::group_broadcast(item.get_sub_group(), v1993_data, 9))));
              float v2461_data = ir10[1];
              ir10[1] = (v2461_data + (v2451_data * (sycl::group_broadcast(item.get_sub_group(), v1999_data, 9))));
              float v2467_data = ir10[2];
              ir10[2] = (v2467_data + (v2451_data * (sycl::group_broadcast(item.get_sub_group(), v2005_data, 9))));
              float v2473_data = ir10[3];
              ir10[3] = (v2473_data + (v2451_data * (sycl::group_broadcast(item.get_sub_group(), v2011_data, 9))));
              float v2479_data = ir10[4];
              ir10[4] = (v2479_data + (v2451_data * (sycl::group_broadcast(item.get_sub_group(), v2017_data, 9))));
              float v2485_data = ir10[5];
              ir10[5] = (v2485_data + (v2451_data * (sycl::group_broadcast(item.get_sub_group(), v2023_data, 9))));
              float v2491_data = ir10[6];
              ir10[6] = (v2491_data + (v2451_data * (sycl::group_broadcast(item.get_sub_group(), v2029_data, 9))));
              float v2497_data = ir10[7];
              ir10[7] = (v2497_data + (v2451_data * (sycl::group_broadcast(item.get_sub_group(), v2035_data, 9))));
              float v2502_data = r7[10];
              float v2506_data = ir10[0];
              ir10[0] = (v2506_data + (v2502_data * (sycl::group_broadcast(item.get_sub_group(), v1993_data, 10))));
              float v2512_data = ir10[1];
              ir10[1] = (v2512_data + (v2502_data * (sycl::group_broadcast(item.get_sub_group(), v1999_data, 10))));
              float v2518_data = ir10[2];
              ir10[2] = (v2518_data + (v2502_data * (sycl::group_broadcast(item.get_sub_group(), v2005_data, 10))));
              float v2524_data = ir10[3];
              ir10[3] = (v2524_data + (v2502_data * (sycl::group_broadcast(item.get_sub_group(), v2011_data, 10))));
              float v2530_data = ir10[4];
              ir10[4] = (v2530_data + (v2502_data * (sycl::group_broadcast(item.get_sub_group(), v2017_data, 10))));
              float v2536_data = ir10[5];
              ir10[5] = (v2536_data + (v2502_data * (sycl::group_broadcast(item.get_sub_group(), v2023_data, 10))));
              float v2542_data = ir10[6];
              ir10[6] = (v2542_data + (v2502_data * (sycl::group_broadcast(item.get_sub_group(), v2029_data, 10))));
              float v2548_data = ir10[7];
              ir10[7] = (v2548_data + (v2502_data * (sycl::group_broadcast(item.get_sub_group(), v2035_data, 10))));
              float v2553_data = r7[11];
              float v2557_data = ir10[0];
              ir10[0] = (v2557_data + (v2553_data * (sycl::group_broadcast(item.get_sub_group(), v1993_data, 11))));
              float v2563_data = ir10[1];
              ir10[1] = (v2563_data + (v2553_data * (sycl::group_broadcast(item.get_sub_group(), v1999_data, 11))));
              float v2569_data = ir10[2];
              ir10[2] = (v2569_data + (v2553_data * (sycl::group_broadcast(item.get_sub_group(), v2005_data, 11))));
              float v2575_data = ir10[3];
              ir10[3] = (v2575_data + (v2553_data * (sycl::group_broadcast(item.get_sub_group(), v2011_data, 11))));
              float v2581_data = ir10[4];
              ir10[4] = (v2581_data + (v2553_data * (sycl::group_broadcast(item.get_sub_group(), v2017_data, 11))));
              float v2587_data = ir10[5];
              ir10[5] = (v2587_data + (v2553_data * (sycl::group_broadcast(item.get_sub_group(), v2023_data, 11))));
              float v2593_data = ir10[6];
              ir10[6] = (v2593_data + (v2553_data * (sycl::group_broadcast(item.get_sub_group(), v2029_data, 11))));
              float v2599_data = ir10[7];
              ir10[7] = (v2599_data + (v2553_data * (sycl::group_broadcast(item.get_sub_group(), v2035_data, 11))));
              #pragma unroll
              for (int32_t v2604_n0 = 0; v2604_n0 < 1; ++v2604_n0) {
                #pragma unroll
                for (int32_t v2605_n1 = 0; v2605_n1 < 8; ++v2605_n1) {
                  int32_t v2606_a = v2604_n0 + v2605_n1;
                  float v2607_data = ir10[v2606_a];
                  float v2609_data = r9[v2606_a];
                  r10[v2606_a] = (v2609_data + v2607_data);
                }
              }
              // glb_m0 = store{r>g}(r10);
              #pragma unroll
              for (int32_t v2615_i0 = 0; v2615_i0 < 1; ++v2615_i0) {
                int32_t v2623_lead = v18_lead + (v2615_i0 * 32);
                #pragma unroll
                for (int32_t v2616_i1 = 0; v2616_i1 < 8; ++v2616_i1) {
                  float v2618_data = r10[(v2615_i0 + v2616_i1)];
                  glb_m0[(v2623_lead + ((v2616_i1 + 8) * 32))] = v2618_data;
                }
              }
              item.barrier();
            }
          }
        }
      });
    }
  });
}

