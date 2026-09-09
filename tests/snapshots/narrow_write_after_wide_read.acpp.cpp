// === base name ===
kernel_e2404ad9096784b8

// === header ===
void launcher_kernel_e2404ad9096784b8(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_e2404ad9096784b8(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (32, 1, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_e2404ad9096784b8(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  m4,  m4_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_e2404ad9096784b8(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, float* m3, size_t m3_extraOffset, const float* m4, size_t m4_extraOffset, size_t numElements0, unsigned* flags0 ) {
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
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 416 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 384 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 156 + 0 + m2_extraOffset];
              float *const __restrict__ glb_m3 = &m3[batchId0 * 416 + 0 + m3_extraOffset];
              const float *const __restrict__ glb_m4 = &m4[batchId0 * 169 + 0 + m4_extraOffset];
              float r0[13]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v12_lead = item.get_local_id(0) % 32;
              #pragma unroll
              for (int32_t v13_i0 = 0; v13_i0 < 1; ++v13_i0) {
                int32_t v19_lead = v12_lead + (v13_i0 * 32);
                #pragma unroll
                for (int32_t v14_i1 = 0; v14_i1 < 13; ++v14_i1) {
                  float v22_data = glb_m0[(v19_lead + (v14_i1 * 32))];
                  r0[(v13_i0 + v14_i1)] = v22_data;
                }
              }
              float r2[12]{};
              // r2 = load{g>r}(glb_m1);
              #pragma unroll
              for (int32_t v28_i0 = 0; v28_i0 < 1; ++v28_i0) {
                int32_t v34_lead = v12_lead + (v28_i0 * 32);
                #pragma unroll
                for (int32_t v29_i1 = 0; v29_i1 < 12; ++v29_i1) {
                  float v37_data = glb_m1[(v34_lead + (v29_i1 * 32))];
                  r2[(v28_i0 + v29_i1)] = v37_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r1[13]{};
              // r1 = +(r0) + None
              // [(0, 32), (0, 13)] []
              float v43_data = r0[0];
              float v44_data = r1[0];
              r1[0] = (v44_data + v43_data);
              float v46_data = r0[1];
              float v47_data = r1[1];
              r1[1] = (v47_data + v46_data);
              float v49_data = r0[2];
              float v50_data = r1[2];
              r1[2] = (v50_data + v49_data);
              float v52_data = r0[3];
              float v53_data = r1[3];
              r1[3] = (v53_data + v52_data);
              float v55_data = r0[4];
              float v56_data = r1[4];
              r1[4] = (v56_data + v55_data);
              float v58_data = r0[5];
              float v59_data = r1[5];
              r1[5] = (v59_data + v58_data);
              float v61_data = r0[6];
              float v62_data = r1[6];
              r1[6] = (v62_data + v61_data);
              float v64_data = r0[7];
              float v65_data = r1[7];
              r1[7] = (v65_data + v64_data);
              float v67_data = r0[8];
              float v68_data = r1[8];
              r1[8] = (v68_data + v67_data);
              float v70_data = r0[9];
              float v71_data = r1[9];
              r1[9] = (v71_data + v70_data);
              float v73_data = r0[10];
              float v74_data = r1[10];
              r1[10] = (v74_data + v73_data);
              float v76_data = r0[11];
              float v77_data = r1[11];
              r1[11] = (v77_data + v76_data);
              float v79_data = r0[12];
              float v80_data = r1[12];
              r1[12] = (v80_data + v79_data);
              float r3[13]{};
              // r3 = load{g>r}(glb_m2);
              if (v12_lead < 12) {
                #pragma unroll
                for (int32_t v87_i1 = 0; v87_i1 < 13; ++v87_i1) {
                  float v95_data = glb_m2[(v12_lead + (v87_i1 * 12))];
                  r3[v87_i1] = v95_data;
                }
              }
              // wait(r2 = load{g>r}(glb_m1););
              // wait(r3 = load{g>r}(glb_m2););
              float r4[13]{};
              // r4 = +(r2 * r3) + name: r1, type: SymbolType.Register, lead: [0]
              // [(0, 32), (0, 13)] [(0, 12)]
              float ir4[13]{};
              float v102_data = r2[0];
              float v103_data = r3[0];
              float v106_data = ir4[0];
              ir4[0] = (v106_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 0))));
              float v109_data = r3[1];
              float v112_data = ir4[1];
              ir4[1] = (v112_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 0))));
              float v115_data = r3[2];
              float v118_data = ir4[2];
              ir4[2] = (v118_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v115_data, 0))));
              float v121_data = r3[3];
              float v124_data = ir4[3];
              ir4[3] = (v124_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v121_data, 0))));
              float v127_data = r3[4];
              float v130_data = ir4[4];
              ir4[4] = (v130_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 0))));
              float v133_data = r3[5];
              float v136_data = ir4[5];
              ir4[5] = (v136_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 0))));
              float v139_data = r3[6];
              float v142_data = ir4[6];
              ir4[6] = (v142_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 0))));
              float v145_data = r3[7];
              float v148_data = ir4[7];
              ir4[7] = (v148_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 0))));
              float v151_data = r3[8];
              float v154_data = ir4[8];
              ir4[8] = (v154_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 0))));
              float v157_data = r3[9];
              float v160_data = ir4[9];
              ir4[9] = (v160_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 0))));
              float v163_data = r3[10];
              float v166_data = ir4[10];
              ir4[10] = (v166_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 0))));
              float v169_data = r3[11];
              float v172_data = ir4[11];
              ir4[11] = (v172_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 0))));
              float v175_data = r3[12];
              float v178_data = ir4[12];
              ir4[12] = (v178_data + (v102_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 0))));
              float v183_data = r2[1];
              float v187_data = ir4[0];
              ir4[0] = (v187_data + (v183_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 1))));
              float v193_data = ir4[1];
              ir4[1] = (v193_data + (v183_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 1))));
              float v199_data = ir4[2];
              ir4[2] = (v199_data + (v183_data * (sycl::group_broadcast(item.get_sub_group(), v115_data, 1))));
              float v205_data = ir4[3];
              ir4[3] = (v205_data + (v183_data * (sycl::group_broadcast(item.get_sub_group(), v121_data, 1))));
              float v211_data = ir4[4];
              ir4[4] = (v211_data + (v183_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 1))));
              float v217_data = ir4[5];
              ir4[5] = (v217_data + (v183_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 1))));
              float v223_data = ir4[6];
              ir4[6] = (v223_data + (v183_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 1))));
              float v229_data = ir4[7];
              ir4[7] = (v229_data + (v183_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 1))));
              float v235_data = ir4[8];
              ir4[8] = (v235_data + (v183_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 1))));
              float v241_data = ir4[9];
              ir4[9] = (v241_data + (v183_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 1))));
              float v247_data = ir4[10];
              ir4[10] = (v247_data + (v183_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 1))));
              float v253_data = ir4[11];
              ir4[11] = (v253_data + (v183_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 1))));
              float v259_data = ir4[12];
              ir4[12] = (v259_data + (v183_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 1))));
              float v264_data = r2[2];
              float v268_data = ir4[0];
              ir4[0] = (v268_data + (v264_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 2))));
              float v274_data = ir4[1];
              ir4[1] = (v274_data + (v264_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 2))));
              float v280_data = ir4[2];
              ir4[2] = (v280_data + (v264_data * (sycl::group_broadcast(item.get_sub_group(), v115_data, 2))));
              float v286_data = ir4[3];
              ir4[3] = (v286_data + (v264_data * (sycl::group_broadcast(item.get_sub_group(), v121_data, 2))));
              float v292_data = ir4[4];
              ir4[4] = (v292_data + (v264_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 2))));
              float v298_data = ir4[5];
              ir4[5] = (v298_data + (v264_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 2))));
              float v304_data = ir4[6];
              ir4[6] = (v304_data + (v264_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 2))));
              float v310_data = ir4[7];
              ir4[7] = (v310_data + (v264_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 2))));
              float v316_data = ir4[8];
              ir4[8] = (v316_data + (v264_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 2))));
              float v322_data = ir4[9];
              ir4[9] = (v322_data + (v264_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 2))));
              float v328_data = ir4[10];
              ir4[10] = (v328_data + (v264_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 2))));
              float v334_data = ir4[11];
              ir4[11] = (v334_data + (v264_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 2))));
              float v340_data = ir4[12];
              ir4[12] = (v340_data + (v264_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 2))));
              float v345_data = r2[3];
              float v349_data = ir4[0];
              ir4[0] = (v349_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 3))));
              float v355_data = ir4[1];
              ir4[1] = (v355_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 3))));
              float v361_data = ir4[2];
              ir4[2] = (v361_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v115_data, 3))));
              float v367_data = ir4[3];
              ir4[3] = (v367_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v121_data, 3))));
              float v373_data = ir4[4];
              ir4[4] = (v373_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 3))));
              float v379_data = ir4[5];
              ir4[5] = (v379_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 3))));
              float v385_data = ir4[6];
              ir4[6] = (v385_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 3))));
              float v391_data = ir4[7];
              ir4[7] = (v391_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 3))));
              float v397_data = ir4[8];
              ir4[8] = (v397_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 3))));
              float v403_data = ir4[9];
              ir4[9] = (v403_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 3))));
              float v409_data = ir4[10];
              ir4[10] = (v409_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 3))));
              float v415_data = ir4[11];
              ir4[11] = (v415_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 3))));
              float v421_data = ir4[12];
              ir4[12] = (v421_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 3))));
              float v426_data = r2[4];
              float v430_data = ir4[0];
              ir4[0] = (v430_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 4))));
              float v436_data = ir4[1];
              ir4[1] = (v436_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 4))));
              float v442_data = ir4[2];
              ir4[2] = (v442_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v115_data, 4))));
              float v448_data = ir4[3];
              ir4[3] = (v448_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v121_data, 4))));
              float v454_data = ir4[4];
              ir4[4] = (v454_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 4))));
              float v460_data = ir4[5];
              ir4[5] = (v460_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 4))));
              float v466_data = ir4[6];
              ir4[6] = (v466_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 4))));
              float v472_data = ir4[7];
              ir4[7] = (v472_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 4))));
              float v478_data = ir4[8];
              ir4[8] = (v478_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 4))));
              float v484_data = ir4[9];
              ir4[9] = (v484_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 4))));
              float v490_data = ir4[10];
              ir4[10] = (v490_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 4))));
              float v496_data = ir4[11];
              ir4[11] = (v496_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 4))));
              float v502_data = ir4[12];
              ir4[12] = (v502_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 4))));
              float v507_data = r2[5];
              float v511_data = ir4[0];
              ir4[0] = (v511_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 5))));
              float v517_data = ir4[1];
              ir4[1] = (v517_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 5))));
              float v523_data = ir4[2];
              ir4[2] = (v523_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v115_data, 5))));
              float v529_data = ir4[3];
              ir4[3] = (v529_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v121_data, 5))));
              float v535_data = ir4[4];
              ir4[4] = (v535_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 5))));
              float v541_data = ir4[5];
              ir4[5] = (v541_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 5))));
              float v547_data = ir4[6];
              ir4[6] = (v547_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 5))));
              float v553_data = ir4[7];
              ir4[7] = (v553_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 5))));
              float v559_data = ir4[8];
              ir4[8] = (v559_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 5))));
              float v565_data = ir4[9];
              ir4[9] = (v565_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 5))));
              float v571_data = ir4[10];
              ir4[10] = (v571_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 5))));
              float v577_data = ir4[11];
              ir4[11] = (v577_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 5))));
              float v583_data = ir4[12];
              ir4[12] = (v583_data + (v507_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 5))));
              float v588_data = r2[6];
              float v592_data = ir4[0];
              ir4[0] = (v592_data + (v588_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 6))));
              float v598_data = ir4[1];
              ir4[1] = (v598_data + (v588_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 6))));
              float v604_data = ir4[2];
              ir4[2] = (v604_data + (v588_data * (sycl::group_broadcast(item.get_sub_group(), v115_data, 6))));
              float v610_data = ir4[3];
              ir4[3] = (v610_data + (v588_data * (sycl::group_broadcast(item.get_sub_group(), v121_data, 6))));
              float v616_data = ir4[4];
              ir4[4] = (v616_data + (v588_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 6))));
              float v622_data = ir4[5];
              ir4[5] = (v622_data + (v588_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 6))));
              float v628_data = ir4[6];
              ir4[6] = (v628_data + (v588_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 6))));
              float v634_data = ir4[7];
              ir4[7] = (v634_data + (v588_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 6))));
              float v640_data = ir4[8];
              ir4[8] = (v640_data + (v588_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 6))));
              float v646_data = ir4[9];
              ir4[9] = (v646_data + (v588_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 6))));
              float v652_data = ir4[10];
              ir4[10] = (v652_data + (v588_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 6))));
              float v658_data = ir4[11];
              ir4[11] = (v658_data + (v588_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 6))));
              float v664_data = ir4[12];
              ir4[12] = (v664_data + (v588_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 6))));
              float v669_data = r2[7];
              float v673_data = ir4[0];
              ir4[0] = (v673_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 7))));
              float v679_data = ir4[1];
              ir4[1] = (v679_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 7))));
              float v685_data = ir4[2];
              ir4[2] = (v685_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v115_data, 7))));
              float v691_data = ir4[3];
              ir4[3] = (v691_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v121_data, 7))));
              float v697_data = ir4[4];
              ir4[4] = (v697_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 7))));
              float v703_data = ir4[5];
              ir4[5] = (v703_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 7))));
              float v709_data = ir4[6];
              ir4[6] = (v709_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 7))));
              float v715_data = ir4[7];
              ir4[7] = (v715_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 7))));
              float v721_data = ir4[8];
              ir4[8] = (v721_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 7))));
              float v727_data = ir4[9];
              ir4[9] = (v727_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 7))));
              float v733_data = ir4[10];
              ir4[10] = (v733_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 7))));
              float v739_data = ir4[11];
              ir4[11] = (v739_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 7))));
              float v745_data = ir4[12];
              ir4[12] = (v745_data + (v669_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 7))));
              float v750_data = r2[8];
              float v754_data = ir4[0];
              ir4[0] = (v754_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 8))));
              float v760_data = ir4[1];
              ir4[1] = (v760_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 8))));
              float v766_data = ir4[2];
              ir4[2] = (v766_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v115_data, 8))));
              float v772_data = ir4[3];
              ir4[3] = (v772_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v121_data, 8))));
              float v778_data = ir4[4];
              ir4[4] = (v778_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 8))));
              float v784_data = ir4[5];
              ir4[5] = (v784_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 8))));
              float v790_data = ir4[6];
              ir4[6] = (v790_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 8))));
              float v796_data = ir4[7];
              ir4[7] = (v796_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 8))));
              float v802_data = ir4[8];
              ir4[8] = (v802_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 8))));
              float v808_data = ir4[9];
              ir4[9] = (v808_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 8))));
              float v814_data = ir4[10];
              ir4[10] = (v814_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 8))));
              float v820_data = ir4[11];
              ir4[11] = (v820_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 8))));
              float v826_data = ir4[12];
              ir4[12] = (v826_data + (v750_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 8))));
              float v831_data = r2[9];
              float v835_data = ir4[0];
              ir4[0] = (v835_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 9))));
              float v841_data = ir4[1];
              ir4[1] = (v841_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 9))));
              float v847_data = ir4[2];
              ir4[2] = (v847_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v115_data, 9))));
              float v853_data = ir4[3];
              ir4[3] = (v853_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v121_data, 9))));
              float v859_data = ir4[4];
              ir4[4] = (v859_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 9))));
              float v865_data = ir4[5];
              ir4[5] = (v865_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 9))));
              float v871_data = ir4[6];
              ir4[6] = (v871_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 9))));
              float v877_data = ir4[7];
              ir4[7] = (v877_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 9))));
              float v883_data = ir4[8];
              ir4[8] = (v883_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 9))));
              float v889_data = ir4[9];
              ir4[9] = (v889_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 9))));
              float v895_data = ir4[10];
              ir4[10] = (v895_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 9))));
              float v901_data = ir4[11];
              ir4[11] = (v901_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 9))));
              float v907_data = ir4[12];
              ir4[12] = (v907_data + (v831_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 9))));
              float v912_data = r2[10];
              float v916_data = ir4[0];
              ir4[0] = (v916_data + (v912_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 10))));
              float v922_data = ir4[1];
              ir4[1] = (v922_data + (v912_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 10))));
              float v928_data = ir4[2];
              ir4[2] = (v928_data + (v912_data * (sycl::group_broadcast(item.get_sub_group(), v115_data, 10))));
              float v934_data = ir4[3];
              ir4[3] = (v934_data + (v912_data * (sycl::group_broadcast(item.get_sub_group(), v121_data, 10))));
              float v940_data = ir4[4];
              ir4[4] = (v940_data + (v912_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 10))));
              float v946_data = ir4[5];
              ir4[5] = (v946_data + (v912_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 10))));
              float v952_data = ir4[6];
              ir4[6] = (v952_data + (v912_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 10))));
              float v958_data = ir4[7];
              ir4[7] = (v958_data + (v912_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 10))));
              float v964_data = ir4[8];
              ir4[8] = (v964_data + (v912_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 10))));
              float v970_data = ir4[9];
              ir4[9] = (v970_data + (v912_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 10))));
              float v976_data = ir4[10];
              ir4[10] = (v976_data + (v912_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 10))));
              float v982_data = ir4[11];
              ir4[11] = (v982_data + (v912_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 10))));
              float v988_data = ir4[12];
              ir4[12] = (v988_data + (v912_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 10))));
              float v993_data = r2[11];
              float v997_data = ir4[0];
              ir4[0] = (v997_data + (v993_data * (sycl::group_broadcast(item.get_sub_group(), v103_data, 11))));
              float v1003_data = ir4[1];
              ir4[1] = (v1003_data + (v993_data * (sycl::group_broadcast(item.get_sub_group(), v109_data, 11))));
              float v1009_data = ir4[2];
              ir4[2] = (v1009_data + (v993_data * (sycl::group_broadcast(item.get_sub_group(), v115_data, 11))));
              float v1015_data = ir4[3];
              ir4[3] = (v1015_data + (v993_data * (sycl::group_broadcast(item.get_sub_group(), v121_data, 11))));
              float v1021_data = ir4[4];
              ir4[4] = (v1021_data + (v993_data * (sycl::group_broadcast(item.get_sub_group(), v127_data, 11))));
              float v1027_data = ir4[5];
              ir4[5] = (v1027_data + (v993_data * (sycl::group_broadcast(item.get_sub_group(), v133_data, 11))));
              float v1033_data = ir4[6];
              ir4[6] = (v1033_data + (v993_data * (sycl::group_broadcast(item.get_sub_group(), v139_data, 11))));
              float v1039_data = ir4[7];
              ir4[7] = (v1039_data + (v993_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 11))));
              float v1045_data = ir4[8];
              ir4[8] = (v1045_data + (v993_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 11))));
              float v1051_data = ir4[9];
              ir4[9] = (v1051_data + (v993_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 11))));
              float v1057_data = ir4[10];
              ir4[10] = (v1057_data + (v993_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 11))));
              float v1063_data = ir4[11];
              ir4[11] = (v1063_data + (v993_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 11))));
              float v1069_data = ir4[12];
              ir4[12] = (v1069_data + (v993_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 11))));
              #pragma unroll
              for (int32_t v1074_n0 = 0; v1074_n0 < 1; ++v1074_n0) {
                #pragma unroll
                for (int32_t v1075_n1 = 0; v1075_n1 < 13; ++v1075_n1) {
                  int32_t v1076_a = v1074_n0 + v1075_n1;
                  float v1077_data = ir4[v1076_a];
                  float v1079_data = r1[v1076_a];
                  r4[v1076_a] = (v1079_data + v1077_data);
                }
              }
              float r5[1]{};
              // r5 = +(r4) + None
              // [(0, 32), (0, 1)] []
              float ir5[1]{};
              float v1087_data = r4[4];
              float v1088_data = ir5[0];
              ir5[0] = (v1088_data + v1087_data);
              #pragma unroll
              for (int32_t v1093_n0 = 0; v1093_n0 < 1; ++v1093_n0) {
                #pragma unroll
                for (int32_t v1094_n1 = 0; v1094_n1 < 1; ++v1094_n1) {
                  int32_t v1095_a = v1093_n0 + v1094_n1;
                  float v1096_data = ir5[v1095_a];
                  r5[v1095_a] = v1096_data;
                }
              }
              // glb_m0 = store{r>g}(r5);
              #pragma unroll
              for (int32_t v1101_i0 = 0; v1101_i0 < 1; ++v1101_i0) {
                int32_t v1109_lead = v12_lead + (v1101_i0 * 32);
                #pragma unroll
                for (int32_t v1102_i1 = 0; v1102_i1 < 1; ++v1102_i1) {
                  float v1104_data = r5[(v1101_i0 + v1102_i1)];
                  glb_m0[(v1109_lead + ((v1102_i1 + 4) * 32))] = v1104_data;
                }
              }
              float r6[13]{};
              // r6 = load{g>r}(glb_m0);
              #pragma unroll
              for (int32_t v1117_i0 = 0; v1117_i0 < 1; ++v1117_i0) {
                int32_t v1123_lead = v12_lead + (v1117_i0 * 32);
                #pragma unroll
                for (int32_t v1118_i1 = 0; v1118_i1 < 13; ++v1118_i1) {
                  float v1126_data = glb_m0[(v1123_lead + (v1118_i1 * 32))];
                  r6[(v1117_i0 + v1118_i1)] = v1126_data;
                }
              }
              float r7[13]{};
              // r7 = load{g>r}(glb_m4);
              if (v12_lead < 13) {
                #pragma unroll
                for (int32_t v1133_i1 = 0; v1133_i1 < 13; ++v1133_i1) {
                  float v1141_data = glb_m4[(v12_lead + (v1133_i1 * 13))];
                  r7[v1133_i1] = v1141_data;
                }
              }
              // wait(r6 = load{g>r}(glb_m0););
              // wait(r7 = load{g>r}(glb_m4););
              float r8[13]{};
              // r8 = +(r6 * r7) + None
              // [(0, 32), (0, 13)] [(0, 13)]
              float ir8[13]{};
              float v1148_data = r6[0];
              float v1149_data = r7[0];
              float v1152_data = ir8[0];
              ir8[0] = (v1152_data + (v1148_data * (sycl::group_broadcast(item.get_sub_group(), v1149_data, 0))));
              float v1155_data = r7[1];
              float v1158_data = ir8[1];
              ir8[1] = (v1158_data + (v1148_data * (sycl::group_broadcast(item.get_sub_group(), v1155_data, 0))));
              float v1161_data = r7[2];
              float v1164_data = ir8[2];
              ir8[2] = (v1164_data + (v1148_data * (sycl::group_broadcast(item.get_sub_group(), v1161_data, 0))));
              float v1167_data = r7[3];
              float v1170_data = ir8[3];
              ir8[3] = (v1170_data + (v1148_data * (sycl::group_broadcast(item.get_sub_group(), v1167_data, 0))));
              float v1173_data = r7[4];
              float v1176_data = ir8[4];
              ir8[4] = (v1176_data + (v1148_data * (sycl::group_broadcast(item.get_sub_group(), v1173_data, 0))));
              float v1179_data = r7[5];
              float v1182_data = ir8[5];
              ir8[5] = (v1182_data + (v1148_data * (sycl::group_broadcast(item.get_sub_group(), v1179_data, 0))));
              float v1185_data = r7[6];
              float v1188_data = ir8[6];
              ir8[6] = (v1188_data + (v1148_data * (sycl::group_broadcast(item.get_sub_group(), v1185_data, 0))));
              float v1191_data = r7[7];
              float v1194_data = ir8[7];
              ir8[7] = (v1194_data + (v1148_data * (sycl::group_broadcast(item.get_sub_group(), v1191_data, 0))));
              float v1197_data = r7[8];
              float v1200_data = ir8[8];
              ir8[8] = (v1200_data + (v1148_data * (sycl::group_broadcast(item.get_sub_group(), v1197_data, 0))));
              float v1203_data = r7[9];
              float v1206_data = ir8[9];
              ir8[9] = (v1206_data + (v1148_data * (sycl::group_broadcast(item.get_sub_group(), v1203_data, 0))));
              float v1209_data = r7[10];
              float v1212_data = ir8[10];
              ir8[10] = (v1212_data + (v1148_data * (sycl::group_broadcast(item.get_sub_group(), v1209_data, 0))));
              float v1215_data = r7[11];
              float v1218_data = ir8[11];
              ir8[11] = (v1218_data + (v1148_data * (sycl::group_broadcast(item.get_sub_group(), v1215_data, 0))));
              float v1221_data = r7[12];
              float v1224_data = ir8[12];
              ir8[12] = (v1224_data + (v1148_data * (sycl::group_broadcast(item.get_sub_group(), v1221_data, 0))));
              float v1229_data = r6[1];
              float v1233_data = ir8[0];
              ir8[0] = (v1233_data + (v1229_data * (sycl::group_broadcast(item.get_sub_group(), v1149_data, 1))));
              float v1239_data = ir8[1];
              ir8[1] = (v1239_data + (v1229_data * (sycl::group_broadcast(item.get_sub_group(), v1155_data, 1))));
              float v1245_data = ir8[2];
              ir8[2] = (v1245_data + (v1229_data * (sycl::group_broadcast(item.get_sub_group(), v1161_data, 1))));
              float v1251_data = ir8[3];
              ir8[3] = (v1251_data + (v1229_data * (sycl::group_broadcast(item.get_sub_group(), v1167_data, 1))));
              float v1257_data = ir8[4];
              ir8[4] = (v1257_data + (v1229_data * (sycl::group_broadcast(item.get_sub_group(), v1173_data, 1))));
              float v1263_data = ir8[5];
              ir8[5] = (v1263_data + (v1229_data * (sycl::group_broadcast(item.get_sub_group(), v1179_data, 1))));
              float v1269_data = ir8[6];
              ir8[6] = (v1269_data + (v1229_data * (sycl::group_broadcast(item.get_sub_group(), v1185_data, 1))));
              float v1275_data = ir8[7];
              ir8[7] = (v1275_data + (v1229_data * (sycl::group_broadcast(item.get_sub_group(), v1191_data, 1))));
              float v1281_data = ir8[8];
              ir8[8] = (v1281_data + (v1229_data * (sycl::group_broadcast(item.get_sub_group(), v1197_data, 1))));
              float v1287_data = ir8[9];
              ir8[9] = (v1287_data + (v1229_data * (sycl::group_broadcast(item.get_sub_group(), v1203_data, 1))));
              float v1293_data = ir8[10];
              ir8[10] = (v1293_data + (v1229_data * (sycl::group_broadcast(item.get_sub_group(), v1209_data, 1))));
              float v1299_data = ir8[11];
              ir8[11] = (v1299_data + (v1229_data * (sycl::group_broadcast(item.get_sub_group(), v1215_data, 1))));
              float v1305_data = ir8[12];
              ir8[12] = (v1305_data + (v1229_data * (sycl::group_broadcast(item.get_sub_group(), v1221_data, 1))));
              float v1310_data = r6[2];
              float v1314_data = ir8[0];
              ir8[0] = (v1314_data + (v1310_data * (sycl::group_broadcast(item.get_sub_group(), v1149_data, 2))));
              float v1320_data = ir8[1];
              ir8[1] = (v1320_data + (v1310_data * (sycl::group_broadcast(item.get_sub_group(), v1155_data, 2))));
              float v1326_data = ir8[2];
              ir8[2] = (v1326_data + (v1310_data * (sycl::group_broadcast(item.get_sub_group(), v1161_data, 2))));
              float v1332_data = ir8[3];
              ir8[3] = (v1332_data + (v1310_data * (sycl::group_broadcast(item.get_sub_group(), v1167_data, 2))));
              float v1338_data = ir8[4];
              ir8[4] = (v1338_data + (v1310_data * (sycl::group_broadcast(item.get_sub_group(), v1173_data, 2))));
              float v1344_data = ir8[5];
              ir8[5] = (v1344_data + (v1310_data * (sycl::group_broadcast(item.get_sub_group(), v1179_data, 2))));
              float v1350_data = ir8[6];
              ir8[6] = (v1350_data + (v1310_data * (sycl::group_broadcast(item.get_sub_group(), v1185_data, 2))));
              float v1356_data = ir8[7];
              ir8[7] = (v1356_data + (v1310_data * (sycl::group_broadcast(item.get_sub_group(), v1191_data, 2))));
              float v1362_data = ir8[8];
              ir8[8] = (v1362_data + (v1310_data * (sycl::group_broadcast(item.get_sub_group(), v1197_data, 2))));
              float v1368_data = ir8[9];
              ir8[9] = (v1368_data + (v1310_data * (sycl::group_broadcast(item.get_sub_group(), v1203_data, 2))));
              float v1374_data = ir8[10];
              ir8[10] = (v1374_data + (v1310_data * (sycl::group_broadcast(item.get_sub_group(), v1209_data, 2))));
              float v1380_data = ir8[11];
              ir8[11] = (v1380_data + (v1310_data * (sycl::group_broadcast(item.get_sub_group(), v1215_data, 2))));
              float v1386_data = ir8[12];
              ir8[12] = (v1386_data + (v1310_data * (sycl::group_broadcast(item.get_sub_group(), v1221_data, 2))));
              float v1391_data = r6[3];
              float v1395_data = ir8[0];
              ir8[0] = (v1395_data + (v1391_data * (sycl::group_broadcast(item.get_sub_group(), v1149_data, 3))));
              float v1401_data = ir8[1];
              ir8[1] = (v1401_data + (v1391_data * (sycl::group_broadcast(item.get_sub_group(), v1155_data, 3))));
              float v1407_data = ir8[2];
              ir8[2] = (v1407_data + (v1391_data * (sycl::group_broadcast(item.get_sub_group(), v1161_data, 3))));
              float v1413_data = ir8[3];
              ir8[3] = (v1413_data + (v1391_data * (sycl::group_broadcast(item.get_sub_group(), v1167_data, 3))));
              float v1419_data = ir8[4];
              ir8[4] = (v1419_data + (v1391_data * (sycl::group_broadcast(item.get_sub_group(), v1173_data, 3))));
              float v1425_data = ir8[5];
              ir8[5] = (v1425_data + (v1391_data * (sycl::group_broadcast(item.get_sub_group(), v1179_data, 3))));
              float v1431_data = ir8[6];
              ir8[6] = (v1431_data + (v1391_data * (sycl::group_broadcast(item.get_sub_group(), v1185_data, 3))));
              float v1437_data = ir8[7];
              ir8[7] = (v1437_data + (v1391_data * (sycl::group_broadcast(item.get_sub_group(), v1191_data, 3))));
              float v1443_data = ir8[8];
              ir8[8] = (v1443_data + (v1391_data * (sycl::group_broadcast(item.get_sub_group(), v1197_data, 3))));
              float v1449_data = ir8[9];
              ir8[9] = (v1449_data + (v1391_data * (sycl::group_broadcast(item.get_sub_group(), v1203_data, 3))));
              float v1455_data = ir8[10];
              ir8[10] = (v1455_data + (v1391_data * (sycl::group_broadcast(item.get_sub_group(), v1209_data, 3))));
              float v1461_data = ir8[11];
              ir8[11] = (v1461_data + (v1391_data * (sycl::group_broadcast(item.get_sub_group(), v1215_data, 3))));
              float v1467_data = ir8[12];
              ir8[12] = (v1467_data + (v1391_data * (sycl::group_broadcast(item.get_sub_group(), v1221_data, 3))));
              float v1472_data = r6[4];
              float v1476_data = ir8[0];
              ir8[0] = (v1476_data + (v1472_data * (sycl::group_broadcast(item.get_sub_group(), v1149_data, 4))));
              float v1482_data = ir8[1];
              ir8[1] = (v1482_data + (v1472_data * (sycl::group_broadcast(item.get_sub_group(), v1155_data, 4))));
              float v1488_data = ir8[2];
              ir8[2] = (v1488_data + (v1472_data * (sycl::group_broadcast(item.get_sub_group(), v1161_data, 4))));
              float v1494_data = ir8[3];
              ir8[3] = (v1494_data + (v1472_data * (sycl::group_broadcast(item.get_sub_group(), v1167_data, 4))));
              float v1500_data = ir8[4];
              ir8[4] = (v1500_data + (v1472_data * (sycl::group_broadcast(item.get_sub_group(), v1173_data, 4))));
              float v1506_data = ir8[5];
              ir8[5] = (v1506_data + (v1472_data * (sycl::group_broadcast(item.get_sub_group(), v1179_data, 4))));
              float v1512_data = ir8[6];
              ir8[6] = (v1512_data + (v1472_data * (sycl::group_broadcast(item.get_sub_group(), v1185_data, 4))));
              float v1518_data = ir8[7];
              ir8[7] = (v1518_data + (v1472_data * (sycl::group_broadcast(item.get_sub_group(), v1191_data, 4))));
              float v1524_data = ir8[8];
              ir8[8] = (v1524_data + (v1472_data * (sycl::group_broadcast(item.get_sub_group(), v1197_data, 4))));
              float v1530_data = ir8[9];
              ir8[9] = (v1530_data + (v1472_data * (sycl::group_broadcast(item.get_sub_group(), v1203_data, 4))));
              float v1536_data = ir8[10];
              ir8[10] = (v1536_data + (v1472_data * (sycl::group_broadcast(item.get_sub_group(), v1209_data, 4))));
              float v1542_data = ir8[11];
              ir8[11] = (v1542_data + (v1472_data * (sycl::group_broadcast(item.get_sub_group(), v1215_data, 4))));
              float v1548_data = ir8[12];
              ir8[12] = (v1548_data + (v1472_data * (sycl::group_broadcast(item.get_sub_group(), v1221_data, 4))));
              float v1553_data = r6[5];
              float v1557_data = ir8[0];
              ir8[0] = (v1557_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1149_data, 5))));
              float v1563_data = ir8[1];
              ir8[1] = (v1563_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1155_data, 5))));
              float v1569_data = ir8[2];
              ir8[2] = (v1569_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1161_data, 5))));
              float v1575_data = ir8[3];
              ir8[3] = (v1575_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1167_data, 5))));
              float v1581_data = ir8[4];
              ir8[4] = (v1581_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1173_data, 5))));
              float v1587_data = ir8[5];
              ir8[5] = (v1587_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1179_data, 5))));
              float v1593_data = ir8[6];
              ir8[6] = (v1593_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1185_data, 5))));
              float v1599_data = ir8[7];
              ir8[7] = (v1599_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1191_data, 5))));
              float v1605_data = ir8[8];
              ir8[8] = (v1605_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1197_data, 5))));
              float v1611_data = ir8[9];
              ir8[9] = (v1611_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1203_data, 5))));
              float v1617_data = ir8[10];
              ir8[10] = (v1617_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1209_data, 5))));
              float v1623_data = ir8[11];
              ir8[11] = (v1623_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1215_data, 5))));
              float v1629_data = ir8[12];
              ir8[12] = (v1629_data + (v1553_data * (sycl::group_broadcast(item.get_sub_group(), v1221_data, 5))));
              float v1634_data = r6[6];
              float v1638_data = ir8[0];
              ir8[0] = (v1638_data + (v1634_data * (sycl::group_broadcast(item.get_sub_group(), v1149_data, 6))));
              float v1644_data = ir8[1];
              ir8[1] = (v1644_data + (v1634_data * (sycl::group_broadcast(item.get_sub_group(), v1155_data, 6))));
              float v1650_data = ir8[2];
              ir8[2] = (v1650_data + (v1634_data * (sycl::group_broadcast(item.get_sub_group(), v1161_data, 6))));
              float v1656_data = ir8[3];
              ir8[3] = (v1656_data + (v1634_data * (sycl::group_broadcast(item.get_sub_group(), v1167_data, 6))));
              float v1662_data = ir8[4];
              ir8[4] = (v1662_data + (v1634_data * (sycl::group_broadcast(item.get_sub_group(), v1173_data, 6))));
              float v1668_data = ir8[5];
              ir8[5] = (v1668_data + (v1634_data * (sycl::group_broadcast(item.get_sub_group(), v1179_data, 6))));
              float v1674_data = ir8[6];
              ir8[6] = (v1674_data + (v1634_data * (sycl::group_broadcast(item.get_sub_group(), v1185_data, 6))));
              float v1680_data = ir8[7];
              ir8[7] = (v1680_data + (v1634_data * (sycl::group_broadcast(item.get_sub_group(), v1191_data, 6))));
              float v1686_data = ir8[8];
              ir8[8] = (v1686_data + (v1634_data * (sycl::group_broadcast(item.get_sub_group(), v1197_data, 6))));
              float v1692_data = ir8[9];
              ir8[9] = (v1692_data + (v1634_data * (sycl::group_broadcast(item.get_sub_group(), v1203_data, 6))));
              float v1698_data = ir8[10];
              ir8[10] = (v1698_data + (v1634_data * (sycl::group_broadcast(item.get_sub_group(), v1209_data, 6))));
              float v1704_data = ir8[11];
              ir8[11] = (v1704_data + (v1634_data * (sycl::group_broadcast(item.get_sub_group(), v1215_data, 6))));
              float v1710_data = ir8[12];
              ir8[12] = (v1710_data + (v1634_data * (sycl::group_broadcast(item.get_sub_group(), v1221_data, 6))));
              float v1715_data = r6[7];
              float v1719_data = ir8[0];
              ir8[0] = (v1719_data + (v1715_data * (sycl::group_broadcast(item.get_sub_group(), v1149_data, 7))));
              float v1725_data = ir8[1];
              ir8[1] = (v1725_data + (v1715_data * (sycl::group_broadcast(item.get_sub_group(), v1155_data, 7))));
              float v1731_data = ir8[2];
              ir8[2] = (v1731_data + (v1715_data * (sycl::group_broadcast(item.get_sub_group(), v1161_data, 7))));
              float v1737_data = ir8[3];
              ir8[3] = (v1737_data + (v1715_data * (sycl::group_broadcast(item.get_sub_group(), v1167_data, 7))));
              float v1743_data = ir8[4];
              ir8[4] = (v1743_data + (v1715_data * (sycl::group_broadcast(item.get_sub_group(), v1173_data, 7))));
              float v1749_data = ir8[5];
              ir8[5] = (v1749_data + (v1715_data * (sycl::group_broadcast(item.get_sub_group(), v1179_data, 7))));
              float v1755_data = ir8[6];
              ir8[6] = (v1755_data + (v1715_data * (sycl::group_broadcast(item.get_sub_group(), v1185_data, 7))));
              float v1761_data = ir8[7];
              ir8[7] = (v1761_data + (v1715_data * (sycl::group_broadcast(item.get_sub_group(), v1191_data, 7))));
              float v1767_data = ir8[8];
              ir8[8] = (v1767_data + (v1715_data * (sycl::group_broadcast(item.get_sub_group(), v1197_data, 7))));
              float v1773_data = ir8[9];
              ir8[9] = (v1773_data + (v1715_data * (sycl::group_broadcast(item.get_sub_group(), v1203_data, 7))));
              float v1779_data = ir8[10];
              ir8[10] = (v1779_data + (v1715_data * (sycl::group_broadcast(item.get_sub_group(), v1209_data, 7))));
              float v1785_data = ir8[11];
              ir8[11] = (v1785_data + (v1715_data * (sycl::group_broadcast(item.get_sub_group(), v1215_data, 7))));
              float v1791_data = ir8[12];
              ir8[12] = (v1791_data + (v1715_data * (sycl::group_broadcast(item.get_sub_group(), v1221_data, 7))));
              float v1796_data = r6[8];
              float v1800_data = ir8[0];
              ir8[0] = (v1800_data + (v1796_data * (sycl::group_broadcast(item.get_sub_group(), v1149_data, 8))));
              float v1806_data = ir8[1];
              ir8[1] = (v1806_data + (v1796_data * (sycl::group_broadcast(item.get_sub_group(), v1155_data, 8))));
              float v1812_data = ir8[2];
              ir8[2] = (v1812_data + (v1796_data * (sycl::group_broadcast(item.get_sub_group(), v1161_data, 8))));
              float v1818_data = ir8[3];
              ir8[3] = (v1818_data + (v1796_data * (sycl::group_broadcast(item.get_sub_group(), v1167_data, 8))));
              float v1824_data = ir8[4];
              ir8[4] = (v1824_data + (v1796_data * (sycl::group_broadcast(item.get_sub_group(), v1173_data, 8))));
              float v1830_data = ir8[5];
              ir8[5] = (v1830_data + (v1796_data * (sycl::group_broadcast(item.get_sub_group(), v1179_data, 8))));
              float v1836_data = ir8[6];
              ir8[6] = (v1836_data + (v1796_data * (sycl::group_broadcast(item.get_sub_group(), v1185_data, 8))));
              float v1842_data = ir8[7];
              ir8[7] = (v1842_data + (v1796_data * (sycl::group_broadcast(item.get_sub_group(), v1191_data, 8))));
              float v1848_data = ir8[8];
              ir8[8] = (v1848_data + (v1796_data * (sycl::group_broadcast(item.get_sub_group(), v1197_data, 8))));
              float v1854_data = ir8[9];
              ir8[9] = (v1854_data + (v1796_data * (sycl::group_broadcast(item.get_sub_group(), v1203_data, 8))));
              float v1860_data = ir8[10];
              ir8[10] = (v1860_data + (v1796_data * (sycl::group_broadcast(item.get_sub_group(), v1209_data, 8))));
              float v1866_data = ir8[11];
              ir8[11] = (v1866_data + (v1796_data * (sycl::group_broadcast(item.get_sub_group(), v1215_data, 8))));
              float v1872_data = ir8[12];
              ir8[12] = (v1872_data + (v1796_data * (sycl::group_broadcast(item.get_sub_group(), v1221_data, 8))));
              float v1877_data = r6[9];
              float v1881_data = ir8[0];
              ir8[0] = (v1881_data + (v1877_data * (sycl::group_broadcast(item.get_sub_group(), v1149_data, 9))));
              float v1887_data = ir8[1];
              ir8[1] = (v1887_data + (v1877_data * (sycl::group_broadcast(item.get_sub_group(), v1155_data, 9))));
              float v1893_data = ir8[2];
              ir8[2] = (v1893_data + (v1877_data * (sycl::group_broadcast(item.get_sub_group(), v1161_data, 9))));
              float v1899_data = ir8[3];
              ir8[3] = (v1899_data + (v1877_data * (sycl::group_broadcast(item.get_sub_group(), v1167_data, 9))));
              float v1905_data = ir8[4];
              ir8[4] = (v1905_data + (v1877_data * (sycl::group_broadcast(item.get_sub_group(), v1173_data, 9))));
              float v1911_data = ir8[5];
              ir8[5] = (v1911_data + (v1877_data * (sycl::group_broadcast(item.get_sub_group(), v1179_data, 9))));
              float v1917_data = ir8[6];
              ir8[6] = (v1917_data + (v1877_data * (sycl::group_broadcast(item.get_sub_group(), v1185_data, 9))));
              float v1923_data = ir8[7];
              ir8[7] = (v1923_data + (v1877_data * (sycl::group_broadcast(item.get_sub_group(), v1191_data, 9))));
              float v1929_data = ir8[8];
              ir8[8] = (v1929_data + (v1877_data * (sycl::group_broadcast(item.get_sub_group(), v1197_data, 9))));
              float v1935_data = ir8[9];
              ir8[9] = (v1935_data + (v1877_data * (sycl::group_broadcast(item.get_sub_group(), v1203_data, 9))));
              float v1941_data = ir8[10];
              ir8[10] = (v1941_data + (v1877_data * (sycl::group_broadcast(item.get_sub_group(), v1209_data, 9))));
              float v1947_data = ir8[11];
              ir8[11] = (v1947_data + (v1877_data * (sycl::group_broadcast(item.get_sub_group(), v1215_data, 9))));
              float v1953_data = ir8[12];
              ir8[12] = (v1953_data + (v1877_data * (sycl::group_broadcast(item.get_sub_group(), v1221_data, 9))));
              float v1958_data = r6[10];
              float v1962_data = ir8[0];
              ir8[0] = (v1962_data + (v1958_data * (sycl::group_broadcast(item.get_sub_group(), v1149_data, 10))));
              float v1968_data = ir8[1];
              ir8[1] = (v1968_data + (v1958_data * (sycl::group_broadcast(item.get_sub_group(), v1155_data, 10))));
              float v1974_data = ir8[2];
              ir8[2] = (v1974_data + (v1958_data * (sycl::group_broadcast(item.get_sub_group(), v1161_data, 10))));
              float v1980_data = ir8[3];
              ir8[3] = (v1980_data + (v1958_data * (sycl::group_broadcast(item.get_sub_group(), v1167_data, 10))));
              float v1986_data = ir8[4];
              ir8[4] = (v1986_data + (v1958_data * (sycl::group_broadcast(item.get_sub_group(), v1173_data, 10))));
              float v1992_data = ir8[5];
              ir8[5] = (v1992_data + (v1958_data * (sycl::group_broadcast(item.get_sub_group(), v1179_data, 10))));
              float v1998_data = ir8[6];
              ir8[6] = (v1998_data + (v1958_data * (sycl::group_broadcast(item.get_sub_group(), v1185_data, 10))));
              float v2004_data = ir8[7];
              ir8[7] = (v2004_data + (v1958_data * (sycl::group_broadcast(item.get_sub_group(), v1191_data, 10))));
              float v2010_data = ir8[8];
              ir8[8] = (v2010_data + (v1958_data * (sycl::group_broadcast(item.get_sub_group(), v1197_data, 10))));
              float v2016_data = ir8[9];
              ir8[9] = (v2016_data + (v1958_data * (sycl::group_broadcast(item.get_sub_group(), v1203_data, 10))));
              float v2022_data = ir8[10];
              ir8[10] = (v2022_data + (v1958_data * (sycl::group_broadcast(item.get_sub_group(), v1209_data, 10))));
              float v2028_data = ir8[11];
              ir8[11] = (v2028_data + (v1958_data * (sycl::group_broadcast(item.get_sub_group(), v1215_data, 10))));
              float v2034_data = ir8[12];
              ir8[12] = (v2034_data + (v1958_data * (sycl::group_broadcast(item.get_sub_group(), v1221_data, 10))));
              float v2039_data = r6[11];
              float v2043_data = ir8[0];
              ir8[0] = (v2043_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v1149_data, 11))));
              float v2049_data = ir8[1];
              ir8[1] = (v2049_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v1155_data, 11))));
              float v2055_data = ir8[2];
              ir8[2] = (v2055_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v1161_data, 11))));
              float v2061_data = ir8[3];
              ir8[3] = (v2061_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v1167_data, 11))));
              float v2067_data = ir8[4];
              ir8[4] = (v2067_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v1173_data, 11))));
              float v2073_data = ir8[5];
              ir8[5] = (v2073_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v1179_data, 11))));
              float v2079_data = ir8[6];
              ir8[6] = (v2079_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v1185_data, 11))));
              float v2085_data = ir8[7];
              ir8[7] = (v2085_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v1191_data, 11))));
              float v2091_data = ir8[8];
              ir8[8] = (v2091_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v1197_data, 11))));
              float v2097_data = ir8[9];
              ir8[9] = (v2097_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v1203_data, 11))));
              float v2103_data = ir8[10];
              ir8[10] = (v2103_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v1209_data, 11))));
              float v2109_data = ir8[11];
              ir8[11] = (v2109_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v1215_data, 11))));
              float v2115_data = ir8[12];
              ir8[12] = (v2115_data + (v2039_data * (sycl::group_broadcast(item.get_sub_group(), v1221_data, 11))));
              float v2120_data = r6[12];
              float v2124_data = ir8[0];
              ir8[0] = (v2124_data + (v2120_data * (sycl::group_broadcast(item.get_sub_group(), v1149_data, 12))));
              float v2130_data = ir8[1];
              ir8[1] = (v2130_data + (v2120_data * (sycl::group_broadcast(item.get_sub_group(), v1155_data, 12))));
              float v2136_data = ir8[2];
              ir8[2] = (v2136_data + (v2120_data * (sycl::group_broadcast(item.get_sub_group(), v1161_data, 12))));
              float v2142_data = ir8[3];
              ir8[3] = (v2142_data + (v2120_data * (sycl::group_broadcast(item.get_sub_group(), v1167_data, 12))));
              float v2148_data = ir8[4];
              ir8[4] = (v2148_data + (v2120_data * (sycl::group_broadcast(item.get_sub_group(), v1173_data, 12))));
              float v2154_data = ir8[5];
              ir8[5] = (v2154_data + (v2120_data * (sycl::group_broadcast(item.get_sub_group(), v1179_data, 12))));
              float v2160_data = ir8[6];
              ir8[6] = (v2160_data + (v2120_data * (sycl::group_broadcast(item.get_sub_group(), v1185_data, 12))));
              float v2166_data = ir8[7];
              ir8[7] = (v2166_data + (v2120_data * (sycl::group_broadcast(item.get_sub_group(), v1191_data, 12))));
              float v2172_data = ir8[8];
              ir8[8] = (v2172_data + (v2120_data * (sycl::group_broadcast(item.get_sub_group(), v1197_data, 12))));
              float v2178_data = ir8[9];
              ir8[9] = (v2178_data + (v2120_data * (sycl::group_broadcast(item.get_sub_group(), v1203_data, 12))));
              float v2184_data = ir8[10];
              ir8[10] = (v2184_data + (v2120_data * (sycl::group_broadcast(item.get_sub_group(), v1209_data, 12))));
              float v2190_data = ir8[11];
              ir8[11] = (v2190_data + (v2120_data * (sycl::group_broadcast(item.get_sub_group(), v1215_data, 12))));
              float v2196_data = ir8[12];
              ir8[12] = (v2196_data + (v2120_data * (sycl::group_broadcast(item.get_sub_group(), v1221_data, 12))));
              #pragma unroll
              for (int32_t v2201_n0 = 0; v2201_n0 < 1; ++v2201_n0) {
                #pragma unroll
                for (int32_t v2202_n1 = 0; v2202_n1 < 13; ++v2202_n1) {
                  int32_t v2203_a = v2201_n0 + v2202_n1;
                  float v2204_data = ir8[v2203_a];
                  r8[v2203_a] = v2204_data;
                }
              }
              // glb_m3 = store{r>g}(r8);
              #pragma unroll
              for (int32_t v2209_i0 = 0; v2209_i0 < 1; ++v2209_i0) {
                int32_t v2217_lead = v12_lead + (v2209_i0 * 32);
                #pragma unroll
                for (int32_t v2210_i1 = 0; v2210_i1 < 13; ++v2210_i1) {
                  float v2212_data = r8[(v2209_i0 + v2210_i1)];
                  glb_m3[(v2217_lead + (v2210_i1 * 32))] = v2212_data;
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

