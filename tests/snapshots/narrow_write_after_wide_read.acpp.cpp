// === base name ===
kernel_056595506a2a751e

// === header ===
void launcher_kernel_056595506a2a751e(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_056595506a2a751e(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_056595506a2a751e(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_056595506a2a751e(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (0, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 32×13(32×13) {0..32}×{0..13} strided
        // m1 32×12(32×12) {0..32}×{0..12} strided
        // m2 12×13(12×13) {0..12}×{0..13} strided
        // m3 32×13(32×13) {0..32}×{0..13} strided
        // m4 13×13(13×13) {0..13}×{0..13} strided
        // t0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1]
        // t0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] += m1 32×12(32×12) {0..32}×{0..12} strided({0..32}×{0..12})[0, -1]×m2 12×13(12×13) {0..12}×{0..13} strided({0..12}×{0..13})[-1, 1]
        // m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1] = t0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..1})[0, 1]
        // m3 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, 1] = m0 32×13(32×13) {0..32}×{0..13} strided({0..32}×{0..13})[0, -1]×m4 13×13(13×13) {0..13}×{0..13} strided({0..13}×{0..13})[-1, 1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t v0_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v0_batchId0 < numElements0; v0_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v1_ahead1 = v0_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v3_batchId1 = (v1_ahead1 < numElements0) ? v1_ahead1 : v0_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v0_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v0_batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v0_batchId0 * 384 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v0_batchId0 * 156 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[v0_batchId0 * 416 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[v0_batchId0 * 169 + 0 + m4_extraOffset];
              float r0[13]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v16_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
                int32_t v23_lead = v16_lead + (v17_i0 * 32);
                #pragma unroll
                for (int32_t v18_i1 = 0; v18_i1 < 13; ++v18_i1) {
                  float v26_data = glb_m0[(v23_lead + (v18_i1 * 32))];
                  r0[(v17_i0 + v18_i1)] = v26_data;
                }
              }
              float r2[12]{};
              // r2 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v32_i0 = 0; v32_i0 < 1; ++v32_i0) {
                int32_t v38_lead = v16_lead + (v32_i0 * 32);
                #pragma unroll
                for (int32_t v33_i1 = 0; v33_i1 < 12; ++v33_i1) {
                  float v41_data = glb_m1[(v38_lead + (v33_i1 * 32))];
                  r2[(v32_i0 + v33_i1)] = v41_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[13]{};
              // r1 = +(r0) + None
              // [(0, 32), (0, 13)] []
              float v47_data = r0[0];
              float v48_data = r1[0];
              r1[0] = (v48_data + v47_data);
              float v50_data = r0[1];
              float v51_data = r1[1];
              r1[1] = (v51_data + v50_data);
              float v53_data = r0[2];
              float v54_data = r1[2];
              r1[2] = (v54_data + v53_data);
              float v56_data = r0[3];
              float v57_data = r1[3];
              r1[3] = (v57_data + v56_data);
              float v59_data = r0[4];
              float v60_data = r1[4];
              r1[4] = (v60_data + v59_data);
              float v62_data = r0[5];
              float v63_data = r1[5];
              r1[5] = (v63_data + v62_data);
              float v65_data = r0[6];
              float v66_data = r1[6];
              r1[6] = (v66_data + v65_data);
              float v68_data = r0[7];
              float v69_data = r1[7];
              r1[7] = (v69_data + v68_data);
              float v71_data = r0[8];
              float v72_data = r1[8];
              r1[8] = (v72_data + v71_data);
              float v74_data = r0[9];
              float v75_data = r1[9];
              r1[9] = (v75_data + v74_data);
              float v77_data = r0[10];
              float v78_data = r1[10];
              r1[10] = (v78_data + v77_data);
              float v80_data = r0[11];
              float v81_data = r1[11];
              r1[11] = (v81_data + v80_data);
              float v83_data = r0[12];
              float v84_data = r1[12];
              r1[12] = (v84_data + v83_data);
              float r3[13]{};
              // r3 = load{g>r}(glb_m2);
              if (v16_lead < 12) {
                #pragma unroll
                for (int32_t v91_i1 = 0; v91_i1 < 13; ++v91_i1) {
                  float v99_data = glb_m2[(v16_lead + (v91_i1 * 12))];
                  r3[v91_i1] = v99_data;
                }
              }
              // wait(r2 = load{g>r}(glb_m1););
              // wait(r3 = load{g>r}(glb_m2););
              float r4[13]{};
              // r4 = +(r2 * r3) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 13)] [(0, 12)]
              float ir4[13]{};
              float v106_data = r2[0];
              float v107_data = r3[0];
              float v110_data = ir4[0];
              ir4[0] = (v110_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 0))));
              float v113_data = r3[1];
              float v116_data = ir4[1];
              ir4[1] = (v116_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 0))));
              float v119_data = r3[2];
              float v122_data = ir4[2];
              ir4[2] = (v122_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 0))));
              float v125_data = r3[3];
              float v128_data = ir4[3];
              ir4[3] = (v128_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 0))));
              float v131_data = r3[4];
              float v134_data = ir4[4];
              ir4[4] = (v134_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 0))));
              float v137_data = r3[5];
              float v140_data = ir4[5];
              ir4[5] = (v140_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 0))));
              float v143_data = r3[6];
              float v146_data = ir4[6];
              ir4[6] = (v146_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 0))));
              float v149_data = r3[7];
              float v152_data = ir4[7];
              ir4[7] = (v152_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 0))));
              float v155_data = r3[8];
              float v158_data = ir4[8];
              ir4[8] = (v158_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 0))));
              float v161_data = r3[9];
              float v164_data = ir4[9];
              ir4[9] = (v164_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 0))));
              float v167_data = r3[10];
              float v170_data = ir4[10];
              ir4[10] = (v170_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 0))));
              float v173_data = r3[11];
              float v176_data = ir4[11];
              ir4[11] = (v176_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 0))));
              float v179_data = r3[12];
              float v182_data = ir4[12];
              ir4[12] = (v182_data + (v106_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 0))));
              float v187_data = r2[1];
              float v191_data = ir4[0];
              ir4[0] = (v191_data + (v187_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 1))));
              float v197_data = ir4[1];
              ir4[1] = (v197_data + (v187_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 1))));
              float v203_data = ir4[2];
              ir4[2] = (v203_data + (v187_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 1))));
              float v209_data = ir4[3];
              ir4[3] = (v209_data + (v187_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 1))));
              float v215_data = ir4[4];
              ir4[4] = (v215_data + (v187_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 1))));
              float v221_data = ir4[5];
              ir4[5] = (v221_data + (v187_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 1))));
              float v227_data = ir4[6];
              ir4[6] = (v227_data + (v187_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 1))));
              float v233_data = ir4[7];
              ir4[7] = (v233_data + (v187_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 1))));
              float v239_data = ir4[8];
              ir4[8] = (v239_data + (v187_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 1))));
              float v245_data = ir4[9];
              ir4[9] = (v245_data + (v187_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 1))));
              float v251_data = ir4[10];
              ir4[10] = (v251_data + (v187_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 1))));
              float v257_data = ir4[11];
              ir4[11] = (v257_data + (v187_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 1))));
              float v263_data = ir4[12];
              ir4[12] = (v263_data + (v187_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 1))));
              float v268_data = r2[2];
              float v272_data = ir4[0];
              ir4[0] = (v272_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 2))));
              float v278_data = ir4[1];
              ir4[1] = (v278_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 2))));
              float v284_data = ir4[2];
              ir4[2] = (v284_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 2))));
              float v290_data = ir4[3];
              ir4[3] = (v290_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 2))));
              float v296_data = ir4[4];
              ir4[4] = (v296_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 2))));
              float v302_data = ir4[5];
              ir4[5] = (v302_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 2))));
              float v308_data = ir4[6];
              ir4[6] = (v308_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 2))));
              float v314_data = ir4[7];
              ir4[7] = (v314_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 2))));
              float v320_data = ir4[8];
              ir4[8] = (v320_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 2))));
              float v326_data = ir4[9];
              ir4[9] = (v326_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 2))));
              float v332_data = ir4[10];
              ir4[10] = (v332_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 2))));
              float v338_data = ir4[11];
              ir4[11] = (v338_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 2))));
              float v344_data = ir4[12];
              ir4[12] = (v344_data + (v268_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 2))));
              float v349_data = r2[3];
              float v353_data = ir4[0];
              ir4[0] = (v353_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 3))));
              float v359_data = ir4[1];
              ir4[1] = (v359_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 3))));
              float v365_data = ir4[2];
              ir4[2] = (v365_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 3))));
              float v371_data = ir4[3];
              ir4[3] = (v371_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 3))));
              float v377_data = ir4[4];
              ir4[4] = (v377_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 3))));
              float v383_data = ir4[5];
              ir4[5] = (v383_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 3))));
              float v389_data = ir4[6];
              ir4[6] = (v389_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 3))));
              float v395_data = ir4[7];
              ir4[7] = (v395_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 3))));
              float v401_data = ir4[8];
              ir4[8] = (v401_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 3))));
              float v407_data = ir4[9];
              ir4[9] = (v407_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 3))));
              float v413_data = ir4[10];
              ir4[10] = (v413_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 3))));
              float v419_data = ir4[11];
              ir4[11] = (v419_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 3))));
              float v425_data = ir4[12];
              ir4[12] = (v425_data + (v349_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 3))));
              float v430_data = r2[4];
              float v434_data = ir4[0];
              ir4[0] = (v434_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 4))));
              float v440_data = ir4[1];
              ir4[1] = (v440_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 4))));
              float v446_data = ir4[2];
              ir4[2] = (v446_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 4))));
              float v452_data = ir4[3];
              ir4[3] = (v452_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 4))));
              float v458_data = ir4[4];
              ir4[4] = (v458_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 4))));
              float v464_data = ir4[5];
              ir4[5] = (v464_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 4))));
              float v470_data = ir4[6];
              ir4[6] = (v470_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 4))));
              float v476_data = ir4[7];
              ir4[7] = (v476_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 4))));
              float v482_data = ir4[8];
              ir4[8] = (v482_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 4))));
              float v488_data = ir4[9];
              ir4[9] = (v488_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 4))));
              float v494_data = ir4[10];
              ir4[10] = (v494_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 4))));
              float v500_data = ir4[11];
              ir4[11] = (v500_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 4))));
              float v506_data = ir4[12];
              ir4[12] = (v506_data + (v430_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 4))));
              float v511_data = r2[5];
              float v515_data = ir4[0];
              ir4[0] = (v515_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 5))));
              float v521_data = ir4[1];
              ir4[1] = (v521_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 5))));
              float v527_data = ir4[2];
              ir4[2] = (v527_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 5))));
              float v533_data = ir4[3];
              ir4[3] = (v533_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 5))));
              float v539_data = ir4[4];
              ir4[4] = (v539_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 5))));
              float v545_data = ir4[5];
              ir4[5] = (v545_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 5))));
              float v551_data = ir4[6];
              ir4[6] = (v551_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 5))));
              float v557_data = ir4[7];
              ir4[7] = (v557_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 5))));
              float v563_data = ir4[8];
              ir4[8] = (v563_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 5))));
              float v569_data = ir4[9];
              ir4[9] = (v569_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 5))));
              float v575_data = ir4[10];
              ir4[10] = (v575_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 5))));
              float v581_data = ir4[11];
              ir4[11] = (v581_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 5))));
              float v587_data = ir4[12];
              ir4[12] = (v587_data + (v511_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 5))));
              float v592_data = r2[6];
              float v596_data = ir4[0];
              ir4[0] = (v596_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 6))));
              float v602_data = ir4[1];
              ir4[1] = (v602_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 6))));
              float v608_data = ir4[2];
              ir4[2] = (v608_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 6))));
              float v614_data = ir4[3];
              ir4[3] = (v614_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 6))));
              float v620_data = ir4[4];
              ir4[4] = (v620_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 6))));
              float v626_data = ir4[5];
              ir4[5] = (v626_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 6))));
              float v632_data = ir4[6];
              ir4[6] = (v632_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 6))));
              float v638_data = ir4[7];
              ir4[7] = (v638_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 6))));
              float v644_data = ir4[8];
              ir4[8] = (v644_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 6))));
              float v650_data = ir4[9];
              ir4[9] = (v650_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 6))));
              float v656_data = ir4[10];
              ir4[10] = (v656_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 6))));
              float v662_data = ir4[11];
              ir4[11] = (v662_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 6))));
              float v668_data = ir4[12];
              ir4[12] = (v668_data + (v592_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 6))));
              float v673_data = r2[7];
              float v677_data = ir4[0];
              ir4[0] = (v677_data + (v673_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 7))));
              float v683_data = ir4[1];
              ir4[1] = (v683_data + (v673_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 7))));
              float v689_data = ir4[2];
              ir4[2] = (v689_data + (v673_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 7))));
              float v695_data = ir4[3];
              ir4[3] = (v695_data + (v673_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 7))));
              float v701_data = ir4[4];
              ir4[4] = (v701_data + (v673_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 7))));
              float v707_data = ir4[5];
              ir4[5] = (v707_data + (v673_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 7))));
              float v713_data = ir4[6];
              ir4[6] = (v713_data + (v673_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 7))));
              float v719_data = ir4[7];
              ir4[7] = (v719_data + (v673_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 7))));
              float v725_data = ir4[8];
              ir4[8] = (v725_data + (v673_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 7))));
              float v731_data = ir4[9];
              ir4[9] = (v731_data + (v673_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 7))));
              float v737_data = ir4[10];
              ir4[10] = (v737_data + (v673_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 7))));
              float v743_data = ir4[11];
              ir4[11] = (v743_data + (v673_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 7))));
              float v749_data = ir4[12];
              ir4[12] = (v749_data + (v673_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 7))));
              float v754_data = r2[8];
              float v758_data = ir4[0];
              ir4[0] = (v758_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 8))));
              float v764_data = ir4[1];
              ir4[1] = (v764_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 8))));
              float v770_data = ir4[2];
              ir4[2] = (v770_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 8))));
              float v776_data = ir4[3];
              ir4[3] = (v776_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 8))));
              float v782_data = ir4[4];
              ir4[4] = (v782_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 8))));
              float v788_data = ir4[5];
              ir4[5] = (v788_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 8))));
              float v794_data = ir4[6];
              ir4[6] = (v794_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 8))));
              float v800_data = ir4[7];
              ir4[7] = (v800_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 8))));
              float v806_data = ir4[8];
              ir4[8] = (v806_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 8))));
              float v812_data = ir4[9];
              ir4[9] = (v812_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 8))));
              float v818_data = ir4[10];
              ir4[10] = (v818_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 8))));
              float v824_data = ir4[11];
              ir4[11] = (v824_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 8))));
              float v830_data = ir4[12];
              ir4[12] = (v830_data + (v754_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 8))));
              float v835_data = r2[9];
              float v839_data = ir4[0];
              ir4[0] = (v839_data + (v835_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 9))));
              float v845_data = ir4[1];
              ir4[1] = (v845_data + (v835_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 9))));
              float v851_data = ir4[2];
              ir4[2] = (v851_data + (v835_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 9))));
              float v857_data = ir4[3];
              ir4[3] = (v857_data + (v835_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 9))));
              float v863_data = ir4[4];
              ir4[4] = (v863_data + (v835_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 9))));
              float v869_data = ir4[5];
              ir4[5] = (v869_data + (v835_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 9))));
              float v875_data = ir4[6];
              ir4[6] = (v875_data + (v835_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 9))));
              float v881_data = ir4[7];
              ir4[7] = (v881_data + (v835_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 9))));
              float v887_data = ir4[8];
              ir4[8] = (v887_data + (v835_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 9))));
              float v893_data = ir4[9];
              ir4[9] = (v893_data + (v835_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 9))));
              float v899_data = ir4[10];
              ir4[10] = (v899_data + (v835_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 9))));
              float v905_data = ir4[11];
              ir4[11] = (v905_data + (v835_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 9))));
              float v911_data = ir4[12];
              ir4[12] = (v911_data + (v835_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 9))));
              float v916_data = r2[10];
              float v920_data = ir4[0];
              ir4[0] = (v920_data + (v916_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 10))));
              float v926_data = ir4[1];
              ir4[1] = (v926_data + (v916_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 10))));
              float v932_data = ir4[2];
              ir4[2] = (v932_data + (v916_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 10))));
              float v938_data = ir4[3];
              ir4[3] = (v938_data + (v916_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 10))));
              float v944_data = ir4[4];
              ir4[4] = (v944_data + (v916_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 10))));
              float v950_data = ir4[5];
              ir4[5] = (v950_data + (v916_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 10))));
              float v956_data = ir4[6];
              ir4[6] = (v956_data + (v916_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 10))));
              float v962_data = ir4[7];
              ir4[7] = (v962_data + (v916_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 10))));
              float v968_data = ir4[8];
              ir4[8] = (v968_data + (v916_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 10))));
              float v974_data = ir4[9];
              ir4[9] = (v974_data + (v916_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 10))));
              float v980_data = ir4[10];
              ir4[10] = (v980_data + (v916_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 10))));
              float v986_data = ir4[11];
              ir4[11] = (v986_data + (v916_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 10))));
              float v992_data = ir4[12];
              ir4[12] = (v992_data + (v916_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 10))));
              float v997_data = r2[11];
              float v1001_data = ir4[0];
              ir4[0] = (v1001_data + (v997_data * (sycl::group_broadcast(item.get_sub_group(), v107_data, 11))));
              float v1007_data = ir4[1];
              ir4[1] = (v1007_data + (v997_data * (sycl::group_broadcast(item.get_sub_group(), v113_data, 11))));
              float v1013_data = ir4[2];
              ir4[2] = (v1013_data + (v997_data * (sycl::group_broadcast(item.get_sub_group(), v119_data, 11))));
              float v1019_data = ir4[3];
              ir4[3] = (v1019_data + (v997_data * (sycl::group_broadcast(item.get_sub_group(), v125_data, 11))));
              float v1025_data = ir4[4];
              ir4[4] = (v1025_data + (v997_data * (sycl::group_broadcast(item.get_sub_group(), v131_data, 11))));
              float v1031_data = ir4[5];
              ir4[5] = (v1031_data + (v997_data * (sycl::group_broadcast(item.get_sub_group(), v137_data, 11))));
              float v1037_data = ir4[6];
              ir4[6] = (v1037_data + (v997_data * (sycl::group_broadcast(item.get_sub_group(), v143_data, 11))));
              float v1043_data = ir4[7];
              ir4[7] = (v1043_data + (v997_data * (sycl::group_broadcast(item.get_sub_group(), v149_data, 11))));
              float v1049_data = ir4[8];
              ir4[8] = (v1049_data + (v997_data * (sycl::group_broadcast(item.get_sub_group(), v155_data, 11))));
              float v1055_data = ir4[9];
              ir4[9] = (v1055_data + (v997_data * (sycl::group_broadcast(item.get_sub_group(), v161_data, 11))));
              float v1061_data = ir4[10];
              ir4[10] = (v1061_data + (v997_data * (sycl::group_broadcast(item.get_sub_group(), v167_data, 11))));
              float v1067_data = ir4[11];
              ir4[11] = (v1067_data + (v997_data * (sycl::group_broadcast(item.get_sub_group(), v173_data, 11))));
              float v1073_data = ir4[12];
              ir4[12] = (v1073_data + (v997_data * (sycl::group_broadcast(item.get_sub_group(), v179_data, 11))));
              #pragma unroll
              for (int32_t v1078_n0 = 0; v1078_n0 < 1; ++v1078_n0) {
                #pragma unroll
                for (int32_t v1079_n1 = 0; v1079_n1 < 13; ++v1079_n1) {
                  int32_t v1080_a = v1078_n0 + v1079_n1;
                  float v1081_data = ir4[v1080_a];
                  float v1083_data = r1[v1080_a];
                  r4[v1080_a] = (v1083_data + v1081_data);
                }
              }
              float r5[1]{};
              // r5 = +(r4) + None
              // [(0, 32), (0, 1)] []
              float ir5[1]{};
              float v1091_data = r4[4];
              float v1092_data = ir5[0];
              ir5[0] = (v1092_data + v1091_data);
              #pragma unroll
              for (int32_t v1097_n0 = 0; v1097_n0 < 1; ++v1097_n0) {
                #pragma unroll
                for (int32_t v1098_n1 = 0; v1098_n1 < 1; ++v1098_n1) {
                  int32_t v1099_a = v1097_n0 + v1098_n1;
                  float v1100_data = ir5[v1099_a];
                  r5[v1099_a] = v1100_data;
                }
              }
              // glb_m0 = store{r>g}(r5);
              #pragma unroll
              for (int32_t v1105_i0 = 0; v1105_i0 < 1; ++v1105_i0) {
                int32_t v1113_lead = v16_lead + (v1105_i0 * 32);
                #pragma unroll
                for (int32_t v1106_i1 = 0; v1106_i1 < 1; ++v1106_i1) {
                  float v1108_data = r5[(v1105_i0 + v1106_i1)];
                  glb_m0[(v1113_lead + ((v1106_i1 + 4) * 32))] = v1108_data;
                }
              }
              float r6[13]{};
              // r6 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v1121_i0 = 0; v1121_i0 < 1; ++v1121_i0) {
                int32_t v1127_lead = v16_lead + (v1121_i0 * 32);
                #pragma unroll
                for (int32_t v1122_i1 = 0; v1122_i1 < 13; ++v1122_i1) {
                  float v1130_data = glb_m0[(v1127_lead + (v1122_i1 * 32))];
                  r6[(v1121_i0 + v1122_i1)] = v1130_data;
                }
              }
              float r7[13]{};
              // r7 = load{g>r}(glb_m4);
              if (v16_lead < 13) {
                #pragma unroll
                for (int32_t v1137_i1 = 0; v1137_i1 < 13; ++v1137_i1) {
                  float v1145_data = glb_m4[(v16_lead + (v1137_i1 * 13))];
                  r7[v1137_i1] = v1145_data;
                }
              }
              // wait(r6 = load{g>r}(glb_m0););
              // wait(r7 = load{g>r}(glb_m4););
              float r8[13]{};
              // r8 = +(r6 * r7) + None
              // [(0, 32), (0, 13)] [(0, 13)]
              float ir8[13]{};
              float v1152_data = r6[0];
              float v1153_data = r7[0];
              float v1156_data = ir8[0];
              ir8[0] = (v1156_data + (v1152_data * (sycl::group_broadcast(item.get_sub_group(), v1153_data, 0))));
              float v1159_data = r7[1];
              float v1162_data = ir8[1];
              ir8[1] = (v1162_data + (v1152_data * (sycl::group_broadcast(item.get_sub_group(), v1159_data, 0))));
              float v1165_data = r7[2];
              float v1168_data = ir8[2];
              ir8[2] = (v1168_data + (v1152_data * (sycl::group_broadcast(item.get_sub_group(), v1165_data, 0))));
              float v1171_data = r7[3];
              float v1174_data = ir8[3];
              ir8[3] = (v1174_data + (v1152_data * (sycl::group_broadcast(item.get_sub_group(), v1171_data, 0))));
              float v1177_data = r7[4];
              float v1180_data = ir8[4];
              ir8[4] = (v1180_data + (v1152_data * (sycl::group_broadcast(item.get_sub_group(), v1177_data, 0))));
              float v1183_data = r7[5];
              float v1186_data = ir8[5];
              ir8[5] = (v1186_data + (v1152_data * (sycl::group_broadcast(item.get_sub_group(), v1183_data, 0))));
              float v1189_data = r7[6];
              float v1192_data = ir8[6];
              ir8[6] = (v1192_data + (v1152_data * (sycl::group_broadcast(item.get_sub_group(), v1189_data, 0))));
              float v1195_data = r7[7];
              float v1198_data = ir8[7];
              ir8[7] = (v1198_data + (v1152_data * (sycl::group_broadcast(item.get_sub_group(), v1195_data, 0))));
              float v1201_data = r7[8];
              float v1204_data = ir8[8];
              ir8[8] = (v1204_data + (v1152_data * (sycl::group_broadcast(item.get_sub_group(), v1201_data, 0))));
              float v1207_data = r7[9];
              float v1210_data = ir8[9];
              ir8[9] = (v1210_data + (v1152_data * (sycl::group_broadcast(item.get_sub_group(), v1207_data, 0))));
              float v1213_data = r7[10];
              float v1216_data = ir8[10];
              ir8[10] = (v1216_data + (v1152_data * (sycl::group_broadcast(item.get_sub_group(), v1213_data, 0))));
              float v1219_data = r7[11];
              float v1222_data = ir8[11];
              ir8[11] = (v1222_data + (v1152_data * (sycl::group_broadcast(item.get_sub_group(), v1219_data, 0))));
              float v1225_data = r7[12];
              float v1228_data = ir8[12];
              ir8[12] = (v1228_data + (v1152_data * (sycl::group_broadcast(item.get_sub_group(), v1225_data, 0))));
              float v1233_data = r6[1];
              float v1237_data = ir8[0];
              ir8[0] = (v1237_data + (v1233_data * (sycl::group_broadcast(item.get_sub_group(), v1153_data, 1))));
              float v1243_data = ir8[1];
              ir8[1] = (v1243_data + (v1233_data * (sycl::group_broadcast(item.get_sub_group(), v1159_data, 1))));
              float v1249_data = ir8[2];
              ir8[2] = (v1249_data + (v1233_data * (sycl::group_broadcast(item.get_sub_group(), v1165_data, 1))));
              float v1255_data = ir8[3];
              ir8[3] = (v1255_data + (v1233_data * (sycl::group_broadcast(item.get_sub_group(), v1171_data, 1))));
              float v1261_data = ir8[4];
              ir8[4] = (v1261_data + (v1233_data * (sycl::group_broadcast(item.get_sub_group(), v1177_data, 1))));
              float v1267_data = ir8[5];
              ir8[5] = (v1267_data + (v1233_data * (sycl::group_broadcast(item.get_sub_group(), v1183_data, 1))));
              float v1273_data = ir8[6];
              ir8[6] = (v1273_data + (v1233_data * (sycl::group_broadcast(item.get_sub_group(), v1189_data, 1))));
              float v1279_data = ir8[7];
              ir8[7] = (v1279_data + (v1233_data * (sycl::group_broadcast(item.get_sub_group(), v1195_data, 1))));
              float v1285_data = ir8[8];
              ir8[8] = (v1285_data + (v1233_data * (sycl::group_broadcast(item.get_sub_group(), v1201_data, 1))));
              float v1291_data = ir8[9];
              ir8[9] = (v1291_data + (v1233_data * (sycl::group_broadcast(item.get_sub_group(), v1207_data, 1))));
              float v1297_data = ir8[10];
              ir8[10] = (v1297_data + (v1233_data * (sycl::group_broadcast(item.get_sub_group(), v1213_data, 1))));
              float v1303_data = ir8[11];
              ir8[11] = (v1303_data + (v1233_data * (sycl::group_broadcast(item.get_sub_group(), v1219_data, 1))));
              float v1309_data = ir8[12];
              ir8[12] = (v1309_data + (v1233_data * (sycl::group_broadcast(item.get_sub_group(), v1225_data, 1))));
              float v1314_data = r6[2];
              float v1318_data = ir8[0];
              ir8[0] = (v1318_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1153_data, 2))));
              float v1324_data = ir8[1];
              ir8[1] = (v1324_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1159_data, 2))));
              float v1330_data = ir8[2];
              ir8[2] = (v1330_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1165_data, 2))));
              float v1336_data = ir8[3];
              ir8[3] = (v1336_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1171_data, 2))));
              float v1342_data = ir8[4];
              ir8[4] = (v1342_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1177_data, 2))));
              float v1348_data = ir8[5];
              ir8[5] = (v1348_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1183_data, 2))));
              float v1354_data = ir8[6];
              ir8[6] = (v1354_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1189_data, 2))));
              float v1360_data = ir8[7];
              ir8[7] = (v1360_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1195_data, 2))));
              float v1366_data = ir8[8];
              ir8[8] = (v1366_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1201_data, 2))));
              float v1372_data = ir8[9];
              ir8[9] = (v1372_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1207_data, 2))));
              float v1378_data = ir8[10];
              ir8[10] = (v1378_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1213_data, 2))));
              float v1384_data = ir8[11];
              ir8[11] = (v1384_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1219_data, 2))));
              float v1390_data = ir8[12];
              ir8[12] = (v1390_data + (v1314_data * (sycl::group_broadcast(item.get_sub_group(), v1225_data, 2))));
              float v1395_data = r6[3];
              float v1399_data = ir8[0];
              ir8[0] = (v1399_data + (v1395_data * (sycl::group_broadcast(item.get_sub_group(), v1153_data, 3))));
              float v1405_data = ir8[1];
              ir8[1] = (v1405_data + (v1395_data * (sycl::group_broadcast(item.get_sub_group(), v1159_data, 3))));
              float v1411_data = ir8[2];
              ir8[2] = (v1411_data + (v1395_data * (sycl::group_broadcast(item.get_sub_group(), v1165_data, 3))));
              float v1417_data = ir8[3];
              ir8[3] = (v1417_data + (v1395_data * (sycl::group_broadcast(item.get_sub_group(), v1171_data, 3))));
              float v1423_data = ir8[4];
              ir8[4] = (v1423_data + (v1395_data * (sycl::group_broadcast(item.get_sub_group(), v1177_data, 3))));
              float v1429_data = ir8[5];
              ir8[5] = (v1429_data + (v1395_data * (sycl::group_broadcast(item.get_sub_group(), v1183_data, 3))));
              float v1435_data = ir8[6];
              ir8[6] = (v1435_data + (v1395_data * (sycl::group_broadcast(item.get_sub_group(), v1189_data, 3))));
              float v1441_data = ir8[7];
              ir8[7] = (v1441_data + (v1395_data * (sycl::group_broadcast(item.get_sub_group(), v1195_data, 3))));
              float v1447_data = ir8[8];
              ir8[8] = (v1447_data + (v1395_data * (sycl::group_broadcast(item.get_sub_group(), v1201_data, 3))));
              float v1453_data = ir8[9];
              ir8[9] = (v1453_data + (v1395_data * (sycl::group_broadcast(item.get_sub_group(), v1207_data, 3))));
              float v1459_data = ir8[10];
              ir8[10] = (v1459_data + (v1395_data * (sycl::group_broadcast(item.get_sub_group(), v1213_data, 3))));
              float v1465_data = ir8[11];
              ir8[11] = (v1465_data + (v1395_data * (sycl::group_broadcast(item.get_sub_group(), v1219_data, 3))));
              float v1471_data = ir8[12];
              ir8[12] = (v1471_data + (v1395_data * (sycl::group_broadcast(item.get_sub_group(), v1225_data, 3))));
              float v1476_data = r6[4];
              float v1480_data = ir8[0];
              ir8[0] = (v1480_data + (v1476_data * (sycl::group_broadcast(item.get_sub_group(), v1153_data, 4))));
              float v1486_data = ir8[1];
              ir8[1] = (v1486_data + (v1476_data * (sycl::group_broadcast(item.get_sub_group(), v1159_data, 4))));
              float v1492_data = ir8[2];
              ir8[2] = (v1492_data + (v1476_data * (sycl::group_broadcast(item.get_sub_group(), v1165_data, 4))));
              float v1498_data = ir8[3];
              ir8[3] = (v1498_data + (v1476_data * (sycl::group_broadcast(item.get_sub_group(), v1171_data, 4))));
              float v1504_data = ir8[4];
              ir8[4] = (v1504_data + (v1476_data * (sycl::group_broadcast(item.get_sub_group(), v1177_data, 4))));
              float v1510_data = ir8[5];
              ir8[5] = (v1510_data + (v1476_data * (sycl::group_broadcast(item.get_sub_group(), v1183_data, 4))));
              float v1516_data = ir8[6];
              ir8[6] = (v1516_data + (v1476_data * (sycl::group_broadcast(item.get_sub_group(), v1189_data, 4))));
              float v1522_data = ir8[7];
              ir8[7] = (v1522_data + (v1476_data * (sycl::group_broadcast(item.get_sub_group(), v1195_data, 4))));
              float v1528_data = ir8[8];
              ir8[8] = (v1528_data + (v1476_data * (sycl::group_broadcast(item.get_sub_group(), v1201_data, 4))));
              float v1534_data = ir8[9];
              ir8[9] = (v1534_data + (v1476_data * (sycl::group_broadcast(item.get_sub_group(), v1207_data, 4))));
              float v1540_data = ir8[10];
              ir8[10] = (v1540_data + (v1476_data * (sycl::group_broadcast(item.get_sub_group(), v1213_data, 4))));
              float v1546_data = ir8[11];
              ir8[11] = (v1546_data + (v1476_data * (sycl::group_broadcast(item.get_sub_group(), v1219_data, 4))));
              float v1552_data = ir8[12];
              ir8[12] = (v1552_data + (v1476_data * (sycl::group_broadcast(item.get_sub_group(), v1225_data, 4))));
              float v1557_data = r6[5];
              float v1561_data = ir8[0];
              ir8[0] = (v1561_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1153_data, 5))));
              float v1567_data = ir8[1];
              ir8[1] = (v1567_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1159_data, 5))));
              float v1573_data = ir8[2];
              ir8[2] = (v1573_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1165_data, 5))));
              float v1579_data = ir8[3];
              ir8[3] = (v1579_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1171_data, 5))));
              float v1585_data = ir8[4];
              ir8[4] = (v1585_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1177_data, 5))));
              float v1591_data = ir8[5];
              ir8[5] = (v1591_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1183_data, 5))));
              float v1597_data = ir8[6];
              ir8[6] = (v1597_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1189_data, 5))));
              float v1603_data = ir8[7];
              ir8[7] = (v1603_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1195_data, 5))));
              float v1609_data = ir8[8];
              ir8[8] = (v1609_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1201_data, 5))));
              float v1615_data = ir8[9];
              ir8[9] = (v1615_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1207_data, 5))));
              float v1621_data = ir8[10];
              ir8[10] = (v1621_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1213_data, 5))));
              float v1627_data = ir8[11];
              ir8[11] = (v1627_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1219_data, 5))));
              float v1633_data = ir8[12];
              ir8[12] = (v1633_data + (v1557_data * (sycl::group_broadcast(item.get_sub_group(), v1225_data, 5))));
              float v1638_data = r6[6];
              float v1642_data = ir8[0];
              ir8[0] = (v1642_data + (v1638_data * (sycl::group_broadcast(item.get_sub_group(), v1153_data, 6))));
              float v1648_data = ir8[1];
              ir8[1] = (v1648_data + (v1638_data * (sycl::group_broadcast(item.get_sub_group(), v1159_data, 6))));
              float v1654_data = ir8[2];
              ir8[2] = (v1654_data + (v1638_data * (sycl::group_broadcast(item.get_sub_group(), v1165_data, 6))));
              float v1660_data = ir8[3];
              ir8[3] = (v1660_data + (v1638_data * (sycl::group_broadcast(item.get_sub_group(), v1171_data, 6))));
              float v1666_data = ir8[4];
              ir8[4] = (v1666_data + (v1638_data * (sycl::group_broadcast(item.get_sub_group(), v1177_data, 6))));
              float v1672_data = ir8[5];
              ir8[5] = (v1672_data + (v1638_data * (sycl::group_broadcast(item.get_sub_group(), v1183_data, 6))));
              float v1678_data = ir8[6];
              ir8[6] = (v1678_data + (v1638_data * (sycl::group_broadcast(item.get_sub_group(), v1189_data, 6))));
              float v1684_data = ir8[7];
              ir8[7] = (v1684_data + (v1638_data * (sycl::group_broadcast(item.get_sub_group(), v1195_data, 6))));
              float v1690_data = ir8[8];
              ir8[8] = (v1690_data + (v1638_data * (sycl::group_broadcast(item.get_sub_group(), v1201_data, 6))));
              float v1696_data = ir8[9];
              ir8[9] = (v1696_data + (v1638_data * (sycl::group_broadcast(item.get_sub_group(), v1207_data, 6))));
              float v1702_data = ir8[10];
              ir8[10] = (v1702_data + (v1638_data * (sycl::group_broadcast(item.get_sub_group(), v1213_data, 6))));
              float v1708_data = ir8[11];
              ir8[11] = (v1708_data + (v1638_data * (sycl::group_broadcast(item.get_sub_group(), v1219_data, 6))));
              float v1714_data = ir8[12];
              ir8[12] = (v1714_data + (v1638_data * (sycl::group_broadcast(item.get_sub_group(), v1225_data, 6))));
              float v1719_data = r6[7];
              float v1723_data = ir8[0];
              ir8[0] = (v1723_data + (v1719_data * (sycl::group_broadcast(item.get_sub_group(), v1153_data, 7))));
              float v1729_data = ir8[1];
              ir8[1] = (v1729_data + (v1719_data * (sycl::group_broadcast(item.get_sub_group(), v1159_data, 7))));
              float v1735_data = ir8[2];
              ir8[2] = (v1735_data + (v1719_data * (sycl::group_broadcast(item.get_sub_group(), v1165_data, 7))));
              float v1741_data = ir8[3];
              ir8[3] = (v1741_data + (v1719_data * (sycl::group_broadcast(item.get_sub_group(), v1171_data, 7))));
              float v1747_data = ir8[4];
              ir8[4] = (v1747_data + (v1719_data * (sycl::group_broadcast(item.get_sub_group(), v1177_data, 7))));
              float v1753_data = ir8[5];
              ir8[5] = (v1753_data + (v1719_data * (sycl::group_broadcast(item.get_sub_group(), v1183_data, 7))));
              float v1759_data = ir8[6];
              ir8[6] = (v1759_data + (v1719_data * (sycl::group_broadcast(item.get_sub_group(), v1189_data, 7))));
              float v1765_data = ir8[7];
              ir8[7] = (v1765_data + (v1719_data * (sycl::group_broadcast(item.get_sub_group(), v1195_data, 7))));
              float v1771_data = ir8[8];
              ir8[8] = (v1771_data + (v1719_data * (sycl::group_broadcast(item.get_sub_group(), v1201_data, 7))));
              float v1777_data = ir8[9];
              ir8[9] = (v1777_data + (v1719_data * (sycl::group_broadcast(item.get_sub_group(), v1207_data, 7))));
              float v1783_data = ir8[10];
              ir8[10] = (v1783_data + (v1719_data * (sycl::group_broadcast(item.get_sub_group(), v1213_data, 7))));
              float v1789_data = ir8[11];
              ir8[11] = (v1789_data + (v1719_data * (sycl::group_broadcast(item.get_sub_group(), v1219_data, 7))));
              float v1795_data = ir8[12];
              ir8[12] = (v1795_data + (v1719_data * (sycl::group_broadcast(item.get_sub_group(), v1225_data, 7))));
              float v1800_data = r6[8];
              float v1804_data = ir8[0];
              ir8[0] = (v1804_data + (v1800_data * (sycl::group_broadcast(item.get_sub_group(), v1153_data, 8))));
              float v1810_data = ir8[1];
              ir8[1] = (v1810_data + (v1800_data * (sycl::group_broadcast(item.get_sub_group(), v1159_data, 8))));
              float v1816_data = ir8[2];
              ir8[2] = (v1816_data + (v1800_data * (sycl::group_broadcast(item.get_sub_group(), v1165_data, 8))));
              float v1822_data = ir8[3];
              ir8[3] = (v1822_data + (v1800_data * (sycl::group_broadcast(item.get_sub_group(), v1171_data, 8))));
              float v1828_data = ir8[4];
              ir8[4] = (v1828_data + (v1800_data * (sycl::group_broadcast(item.get_sub_group(), v1177_data, 8))));
              float v1834_data = ir8[5];
              ir8[5] = (v1834_data + (v1800_data * (sycl::group_broadcast(item.get_sub_group(), v1183_data, 8))));
              float v1840_data = ir8[6];
              ir8[6] = (v1840_data + (v1800_data * (sycl::group_broadcast(item.get_sub_group(), v1189_data, 8))));
              float v1846_data = ir8[7];
              ir8[7] = (v1846_data + (v1800_data * (sycl::group_broadcast(item.get_sub_group(), v1195_data, 8))));
              float v1852_data = ir8[8];
              ir8[8] = (v1852_data + (v1800_data * (sycl::group_broadcast(item.get_sub_group(), v1201_data, 8))));
              float v1858_data = ir8[9];
              ir8[9] = (v1858_data + (v1800_data * (sycl::group_broadcast(item.get_sub_group(), v1207_data, 8))));
              float v1864_data = ir8[10];
              ir8[10] = (v1864_data + (v1800_data * (sycl::group_broadcast(item.get_sub_group(), v1213_data, 8))));
              float v1870_data = ir8[11];
              ir8[11] = (v1870_data + (v1800_data * (sycl::group_broadcast(item.get_sub_group(), v1219_data, 8))));
              float v1876_data = ir8[12];
              ir8[12] = (v1876_data + (v1800_data * (sycl::group_broadcast(item.get_sub_group(), v1225_data, 8))));
              float v1881_data = r6[9];
              float v1885_data = ir8[0];
              ir8[0] = (v1885_data + (v1881_data * (sycl::group_broadcast(item.get_sub_group(), v1153_data, 9))));
              float v1891_data = ir8[1];
              ir8[1] = (v1891_data + (v1881_data * (sycl::group_broadcast(item.get_sub_group(), v1159_data, 9))));
              float v1897_data = ir8[2];
              ir8[2] = (v1897_data + (v1881_data * (sycl::group_broadcast(item.get_sub_group(), v1165_data, 9))));
              float v1903_data = ir8[3];
              ir8[3] = (v1903_data + (v1881_data * (sycl::group_broadcast(item.get_sub_group(), v1171_data, 9))));
              float v1909_data = ir8[4];
              ir8[4] = (v1909_data + (v1881_data * (sycl::group_broadcast(item.get_sub_group(), v1177_data, 9))));
              float v1915_data = ir8[5];
              ir8[5] = (v1915_data + (v1881_data * (sycl::group_broadcast(item.get_sub_group(), v1183_data, 9))));
              float v1921_data = ir8[6];
              ir8[6] = (v1921_data + (v1881_data * (sycl::group_broadcast(item.get_sub_group(), v1189_data, 9))));
              float v1927_data = ir8[7];
              ir8[7] = (v1927_data + (v1881_data * (sycl::group_broadcast(item.get_sub_group(), v1195_data, 9))));
              float v1933_data = ir8[8];
              ir8[8] = (v1933_data + (v1881_data * (sycl::group_broadcast(item.get_sub_group(), v1201_data, 9))));
              float v1939_data = ir8[9];
              ir8[9] = (v1939_data + (v1881_data * (sycl::group_broadcast(item.get_sub_group(), v1207_data, 9))));
              float v1945_data = ir8[10];
              ir8[10] = (v1945_data + (v1881_data * (sycl::group_broadcast(item.get_sub_group(), v1213_data, 9))));
              float v1951_data = ir8[11];
              ir8[11] = (v1951_data + (v1881_data * (sycl::group_broadcast(item.get_sub_group(), v1219_data, 9))));
              float v1957_data = ir8[12];
              ir8[12] = (v1957_data + (v1881_data * (sycl::group_broadcast(item.get_sub_group(), v1225_data, 9))));
              float v1962_data = r6[10];
              float v1966_data = ir8[0];
              ir8[0] = (v1966_data + (v1962_data * (sycl::group_broadcast(item.get_sub_group(), v1153_data, 10))));
              float v1972_data = ir8[1];
              ir8[1] = (v1972_data + (v1962_data * (sycl::group_broadcast(item.get_sub_group(), v1159_data, 10))));
              float v1978_data = ir8[2];
              ir8[2] = (v1978_data + (v1962_data * (sycl::group_broadcast(item.get_sub_group(), v1165_data, 10))));
              float v1984_data = ir8[3];
              ir8[3] = (v1984_data + (v1962_data * (sycl::group_broadcast(item.get_sub_group(), v1171_data, 10))));
              float v1990_data = ir8[4];
              ir8[4] = (v1990_data + (v1962_data * (sycl::group_broadcast(item.get_sub_group(), v1177_data, 10))));
              float v1996_data = ir8[5];
              ir8[5] = (v1996_data + (v1962_data * (sycl::group_broadcast(item.get_sub_group(), v1183_data, 10))));
              float v2002_data = ir8[6];
              ir8[6] = (v2002_data + (v1962_data * (sycl::group_broadcast(item.get_sub_group(), v1189_data, 10))));
              float v2008_data = ir8[7];
              ir8[7] = (v2008_data + (v1962_data * (sycl::group_broadcast(item.get_sub_group(), v1195_data, 10))));
              float v2014_data = ir8[8];
              ir8[8] = (v2014_data + (v1962_data * (sycl::group_broadcast(item.get_sub_group(), v1201_data, 10))));
              float v2020_data = ir8[9];
              ir8[9] = (v2020_data + (v1962_data * (sycl::group_broadcast(item.get_sub_group(), v1207_data, 10))));
              float v2026_data = ir8[10];
              ir8[10] = (v2026_data + (v1962_data * (sycl::group_broadcast(item.get_sub_group(), v1213_data, 10))));
              float v2032_data = ir8[11];
              ir8[11] = (v2032_data + (v1962_data * (sycl::group_broadcast(item.get_sub_group(), v1219_data, 10))));
              float v2038_data = ir8[12];
              ir8[12] = (v2038_data + (v1962_data * (sycl::group_broadcast(item.get_sub_group(), v1225_data, 10))));
              float v2043_data = r6[11];
              float v2047_data = ir8[0];
              ir8[0] = (v2047_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v1153_data, 11))));
              float v2053_data = ir8[1];
              ir8[1] = (v2053_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v1159_data, 11))));
              float v2059_data = ir8[2];
              ir8[2] = (v2059_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v1165_data, 11))));
              float v2065_data = ir8[3];
              ir8[3] = (v2065_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v1171_data, 11))));
              float v2071_data = ir8[4];
              ir8[4] = (v2071_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v1177_data, 11))));
              float v2077_data = ir8[5];
              ir8[5] = (v2077_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v1183_data, 11))));
              float v2083_data = ir8[6];
              ir8[6] = (v2083_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v1189_data, 11))));
              float v2089_data = ir8[7];
              ir8[7] = (v2089_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v1195_data, 11))));
              float v2095_data = ir8[8];
              ir8[8] = (v2095_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v1201_data, 11))));
              float v2101_data = ir8[9];
              ir8[9] = (v2101_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v1207_data, 11))));
              float v2107_data = ir8[10];
              ir8[10] = (v2107_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v1213_data, 11))));
              float v2113_data = ir8[11];
              ir8[11] = (v2113_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v1219_data, 11))));
              float v2119_data = ir8[12];
              ir8[12] = (v2119_data + (v2043_data * (sycl::group_broadcast(item.get_sub_group(), v1225_data, 11))));
              float v2124_data = r6[12];
              float v2128_data = ir8[0];
              ir8[0] = (v2128_data + (v2124_data * (sycl::group_broadcast(item.get_sub_group(), v1153_data, 12))));
              float v2134_data = ir8[1];
              ir8[1] = (v2134_data + (v2124_data * (sycl::group_broadcast(item.get_sub_group(), v1159_data, 12))));
              float v2140_data = ir8[2];
              ir8[2] = (v2140_data + (v2124_data * (sycl::group_broadcast(item.get_sub_group(), v1165_data, 12))));
              float v2146_data = ir8[3];
              ir8[3] = (v2146_data + (v2124_data * (sycl::group_broadcast(item.get_sub_group(), v1171_data, 12))));
              float v2152_data = ir8[4];
              ir8[4] = (v2152_data + (v2124_data * (sycl::group_broadcast(item.get_sub_group(), v1177_data, 12))));
              float v2158_data = ir8[5];
              ir8[5] = (v2158_data + (v2124_data * (sycl::group_broadcast(item.get_sub_group(), v1183_data, 12))));
              float v2164_data = ir8[6];
              ir8[6] = (v2164_data + (v2124_data * (sycl::group_broadcast(item.get_sub_group(), v1189_data, 12))));
              float v2170_data = ir8[7];
              ir8[7] = (v2170_data + (v2124_data * (sycl::group_broadcast(item.get_sub_group(), v1195_data, 12))));
              float v2176_data = ir8[8];
              ir8[8] = (v2176_data + (v2124_data * (sycl::group_broadcast(item.get_sub_group(), v1201_data, 12))));
              float v2182_data = ir8[9];
              ir8[9] = (v2182_data + (v2124_data * (sycl::group_broadcast(item.get_sub_group(), v1207_data, 12))));
              float v2188_data = ir8[10];
              ir8[10] = (v2188_data + (v2124_data * (sycl::group_broadcast(item.get_sub_group(), v1213_data, 12))));
              float v2194_data = ir8[11];
              ir8[11] = (v2194_data + (v2124_data * (sycl::group_broadcast(item.get_sub_group(), v1219_data, 12))));
              float v2200_data = ir8[12];
              ir8[12] = (v2200_data + (v2124_data * (sycl::group_broadcast(item.get_sub_group(), v1225_data, 12))));
              #pragma unroll
              for (int32_t v2205_n0 = 0; v2205_n0 < 1; ++v2205_n0) {
                #pragma unroll
                for (int32_t v2206_n1 = 0; v2206_n1 < 13; ++v2206_n1) {
                  int32_t v2207_a = v2205_n0 + v2206_n1;
                  float v2208_data = ir8[v2207_a];
                  r8[v2207_a] = v2208_data;
                }
              }
              // glb_m3 = store{r>g}(r8);
              #pragma unroll
              for (int32_t v2213_i0 = 0; v2213_i0 < 1; ++v2213_i0) {
                int32_t v2221_lead = v16_lead + (v2213_i0 * 32);
                #pragma unroll
                for (int32_t v2214_i1 = 0; v2214_i1 < 13; ++v2214_i1) {
                  float v2216_data = r8[(v2213_i0 + v2214_i1)];
                  glb_m3[(v2221_lead + (v2214_i1 * 32))] = v2216_data;
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

