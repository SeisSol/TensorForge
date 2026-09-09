// === base name ===
kernel_08033ddfd0ce1813

// === header ===
void launcher_kernel_08033ddfd0ce1813(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_08033ddfd0ce1813(float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_08033ddfd0ce1813(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_08033ddfd0ce1813(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, const float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 12×16(12×16) {0..12}×{0..16} strided
        // m1 12×20(12×20) {0..12}×{0..20} strided
        // m2 16×20(16×20) {0..16}×{0..20} strided
        // m0 12×16(12×16) {0..12}×{0..16} strided({0..12}×{0..16})[0, 1] = m1 12×20(12×20) {0..12}×{0..20} strided({0..12}×{0..20})[0, -1]×m2 16×20(16×20) {0..16}×{0..20} strided({0..16}×{0..20})[1, -1]
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[batchId0 * 192 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 240 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[batchId0 * 320 + 0 + m2_extraOffset];
              float r0[20]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v12_lead = item.get_local_id(0) % 16;
              if (v12_lead < 12) {
                #pragma unroll
                for (int32_t v14_i1 = 0; v14_i1 < 20; ++v14_i1) {
                  float v22_data = glb_m1[(v12_lead + (v14_i1 * 12))];
                  r0[v14_i1] = v22_data;
                }
              }
              float r1[20]{};
              // r1 = load{g>r}(glb_m2);
              #pragma unroll
              for (int32_t v28_i0 = 0; v28_i0 < 1; ++v28_i0) {
                int32_t v34_lead = v12_lead + (v28_i0 * 16);
                #pragma unroll
                for (int32_t v29_i1 = 0; v29_i1 < 20; ++v29_i1) {
                  float v37_data = glb_m2[(v34_lead + (v29_i1 * 16))];
                  r1[(v28_i0 + v29_i1)] = v37_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[16]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 16)] [(0, 20)]
              float ir2[16]{};
              if (v12_lead < 12) {
                float v45_data = r0[0];
                float v46_data = r1[0];
                float v49_data = ir2[0];
                ir2[0] = (v49_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 0))));
                float v55_data = ir2[1];
                ir2[1] = (v55_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 1))));
                float v61_data = ir2[2];
                ir2[2] = (v61_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 2))));
                float v67_data = ir2[3];
                ir2[3] = (v67_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 3))));
                float v73_data = ir2[4];
                ir2[4] = (v73_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 4))));
                float v79_data = ir2[5];
                ir2[5] = (v79_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 5))));
                float v85_data = ir2[6];
                ir2[6] = (v85_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 6))));
                float v91_data = ir2[7];
                ir2[7] = (v91_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 7))));
                float v97_data = ir2[8];
                ir2[8] = (v97_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 8))));
                float v103_data = ir2[9];
                ir2[9] = (v103_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 9))));
                float v109_data = ir2[10];
                ir2[10] = (v109_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 10))));
                float v115_data = ir2[11];
                ir2[11] = (v115_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 11))));
                float v121_data = ir2[12];
                ir2[12] = (v121_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 12))));
                float v127_data = ir2[13];
                ir2[13] = (v127_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 13))));
                float v133_data = ir2[14];
                ir2[14] = (v133_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 14))));
                float v139_data = ir2[15];
                ir2[15] = (v139_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 15))));
              }
              if (v12_lead < 12) {
                float v145_data = r0[1];
                float v146_data = r1[1];
                float v149_data = ir2[0];
                ir2[0] = (v149_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 0))));
                float v155_data = ir2[1];
                ir2[1] = (v155_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 1))));
                float v161_data = ir2[2];
                ir2[2] = (v161_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 2))));
                float v167_data = ir2[3];
                ir2[3] = (v167_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 3))));
                float v173_data = ir2[4];
                ir2[4] = (v173_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 4))));
                float v179_data = ir2[5];
                ir2[5] = (v179_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 5))));
                float v185_data = ir2[6];
                ir2[6] = (v185_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 6))));
                float v191_data = ir2[7];
                ir2[7] = (v191_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 7))));
                float v197_data = ir2[8];
                ir2[8] = (v197_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 8))));
                float v203_data = ir2[9];
                ir2[9] = (v203_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 9))));
                float v209_data = ir2[10];
                ir2[10] = (v209_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 10))));
                float v215_data = ir2[11];
                ir2[11] = (v215_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 11))));
                float v221_data = ir2[12];
                ir2[12] = (v221_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 12))));
                float v227_data = ir2[13];
                ir2[13] = (v227_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 13))));
                float v233_data = ir2[14];
                ir2[14] = (v233_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 14))));
                float v239_data = ir2[15];
                ir2[15] = (v239_data + (v145_data * (sycl::group_broadcast(item.get_sub_group(), v146_data, 15))));
              }
              if (v12_lead < 12) {
                float v245_data = r0[2];
                float v246_data = r1[2];
                float v249_data = ir2[0];
                ir2[0] = (v249_data + (v245_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 0))));
                float v255_data = ir2[1];
                ir2[1] = (v255_data + (v245_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 1))));
                float v261_data = ir2[2];
                ir2[2] = (v261_data + (v245_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 2))));
                float v267_data = ir2[3];
                ir2[3] = (v267_data + (v245_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 3))));
                float v273_data = ir2[4];
                ir2[4] = (v273_data + (v245_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 4))));
                float v279_data = ir2[5];
                ir2[5] = (v279_data + (v245_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 5))));
                float v285_data = ir2[6];
                ir2[6] = (v285_data + (v245_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 6))));
                float v291_data = ir2[7];
                ir2[7] = (v291_data + (v245_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 7))));
                float v297_data = ir2[8];
                ir2[8] = (v297_data + (v245_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 8))));
                float v303_data = ir2[9];
                ir2[9] = (v303_data + (v245_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 9))));
                float v309_data = ir2[10];
                ir2[10] = (v309_data + (v245_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 10))));
                float v315_data = ir2[11];
                ir2[11] = (v315_data + (v245_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 11))));
                float v321_data = ir2[12];
                ir2[12] = (v321_data + (v245_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 12))));
                float v327_data = ir2[13];
                ir2[13] = (v327_data + (v245_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 13))));
                float v333_data = ir2[14];
                ir2[14] = (v333_data + (v245_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 14))));
                float v339_data = ir2[15];
                ir2[15] = (v339_data + (v245_data * (sycl::group_broadcast(item.get_sub_group(), v246_data, 15))));
              }
              if (v12_lead < 12) {
                float v345_data = r0[3];
                float v346_data = r1[3];
                float v349_data = ir2[0];
                ir2[0] = (v349_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 0))));
                float v355_data = ir2[1];
                ir2[1] = (v355_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 1))));
                float v361_data = ir2[2];
                ir2[2] = (v361_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 2))));
                float v367_data = ir2[3];
                ir2[3] = (v367_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 3))));
                float v373_data = ir2[4];
                ir2[4] = (v373_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 4))));
                float v379_data = ir2[5];
                ir2[5] = (v379_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 5))));
                float v385_data = ir2[6];
                ir2[6] = (v385_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 6))));
                float v391_data = ir2[7];
                ir2[7] = (v391_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 7))));
                float v397_data = ir2[8];
                ir2[8] = (v397_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 8))));
                float v403_data = ir2[9];
                ir2[9] = (v403_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 9))));
                float v409_data = ir2[10];
                ir2[10] = (v409_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 10))));
                float v415_data = ir2[11];
                ir2[11] = (v415_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 11))));
                float v421_data = ir2[12];
                ir2[12] = (v421_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 12))));
                float v427_data = ir2[13];
                ir2[13] = (v427_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 13))));
                float v433_data = ir2[14];
                ir2[14] = (v433_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 14))));
                float v439_data = ir2[15];
                ir2[15] = (v439_data + (v345_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 15))));
              }
              if (v12_lead < 12) {
                float v445_data = r0[4];
                float v446_data = r1[4];
                float v449_data = ir2[0];
                ir2[0] = (v449_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 0))));
                float v455_data = ir2[1];
                ir2[1] = (v455_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 1))));
                float v461_data = ir2[2];
                ir2[2] = (v461_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 2))));
                float v467_data = ir2[3];
                ir2[3] = (v467_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 3))));
                float v473_data = ir2[4];
                ir2[4] = (v473_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 4))));
                float v479_data = ir2[5];
                ir2[5] = (v479_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 5))));
                float v485_data = ir2[6];
                ir2[6] = (v485_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 6))));
                float v491_data = ir2[7];
                ir2[7] = (v491_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 7))));
                float v497_data = ir2[8];
                ir2[8] = (v497_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 8))));
                float v503_data = ir2[9];
                ir2[9] = (v503_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 9))));
                float v509_data = ir2[10];
                ir2[10] = (v509_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 10))));
                float v515_data = ir2[11];
                ir2[11] = (v515_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 11))));
                float v521_data = ir2[12];
                ir2[12] = (v521_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 12))));
                float v527_data = ir2[13];
                ir2[13] = (v527_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 13))));
                float v533_data = ir2[14];
                ir2[14] = (v533_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 14))));
                float v539_data = ir2[15];
                ir2[15] = (v539_data + (v445_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 15))));
              }
              if (v12_lead < 12) {
                float v545_data = r0[5];
                float v546_data = r1[5];
                float v549_data = ir2[0];
                ir2[0] = (v549_data + (v545_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 0))));
                float v555_data = ir2[1];
                ir2[1] = (v555_data + (v545_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 1))));
                float v561_data = ir2[2];
                ir2[2] = (v561_data + (v545_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 2))));
                float v567_data = ir2[3];
                ir2[3] = (v567_data + (v545_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 3))));
                float v573_data = ir2[4];
                ir2[4] = (v573_data + (v545_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 4))));
                float v579_data = ir2[5];
                ir2[5] = (v579_data + (v545_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 5))));
                float v585_data = ir2[6];
                ir2[6] = (v585_data + (v545_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 6))));
                float v591_data = ir2[7];
                ir2[7] = (v591_data + (v545_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 7))));
                float v597_data = ir2[8];
                ir2[8] = (v597_data + (v545_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 8))));
                float v603_data = ir2[9];
                ir2[9] = (v603_data + (v545_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 9))));
                float v609_data = ir2[10];
                ir2[10] = (v609_data + (v545_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 10))));
                float v615_data = ir2[11];
                ir2[11] = (v615_data + (v545_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 11))));
                float v621_data = ir2[12];
                ir2[12] = (v621_data + (v545_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 12))));
                float v627_data = ir2[13];
                ir2[13] = (v627_data + (v545_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 13))));
                float v633_data = ir2[14];
                ir2[14] = (v633_data + (v545_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 14))));
                float v639_data = ir2[15];
                ir2[15] = (v639_data + (v545_data * (sycl::group_broadcast(item.get_sub_group(), v546_data, 15))));
              }
              if (v12_lead < 12) {
                float v645_data = r0[6];
                float v646_data = r1[6];
                float v649_data = ir2[0];
                ir2[0] = (v649_data + (v645_data * (sycl::group_broadcast(item.get_sub_group(), v646_data, 0))));
                float v655_data = ir2[1];
                ir2[1] = (v655_data + (v645_data * (sycl::group_broadcast(item.get_sub_group(), v646_data, 1))));
                float v661_data = ir2[2];
                ir2[2] = (v661_data + (v645_data * (sycl::group_broadcast(item.get_sub_group(), v646_data, 2))));
                float v667_data = ir2[3];
                ir2[3] = (v667_data + (v645_data * (sycl::group_broadcast(item.get_sub_group(), v646_data, 3))));
                float v673_data = ir2[4];
                ir2[4] = (v673_data + (v645_data * (sycl::group_broadcast(item.get_sub_group(), v646_data, 4))));
                float v679_data = ir2[5];
                ir2[5] = (v679_data + (v645_data * (sycl::group_broadcast(item.get_sub_group(), v646_data, 5))));
                float v685_data = ir2[6];
                ir2[6] = (v685_data + (v645_data * (sycl::group_broadcast(item.get_sub_group(), v646_data, 6))));
                float v691_data = ir2[7];
                ir2[7] = (v691_data + (v645_data * (sycl::group_broadcast(item.get_sub_group(), v646_data, 7))));
                float v697_data = ir2[8];
                ir2[8] = (v697_data + (v645_data * (sycl::group_broadcast(item.get_sub_group(), v646_data, 8))));
                float v703_data = ir2[9];
                ir2[9] = (v703_data + (v645_data * (sycl::group_broadcast(item.get_sub_group(), v646_data, 9))));
                float v709_data = ir2[10];
                ir2[10] = (v709_data + (v645_data * (sycl::group_broadcast(item.get_sub_group(), v646_data, 10))));
                float v715_data = ir2[11];
                ir2[11] = (v715_data + (v645_data * (sycl::group_broadcast(item.get_sub_group(), v646_data, 11))));
                float v721_data = ir2[12];
                ir2[12] = (v721_data + (v645_data * (sycl::group_broadcast(item.get_sub_group(), v646_data, 12))));
                float v727_data = ir2[13];
                ir2[13] = (v727_data + (v645_data * (sycl::group_broadcast(item.get_sub_group(), v646_data, 13))));
                float v733_data = ir2[14];
                ir2[14] = (v733_data + (v645_data * (sycl::group_broadcast(item.get_sub_group(), v646_data, 14))));
                float v739_data = ir2[15];
                ir2[15] = (v739_data + (v645_data * (sycl::group_broadcast(item.get_sub_group(), v646_data, 15))));
              }
              if (v12_lead < 12) {
                float v745_data = r0[7];
                float v746_data = r1[7];
                float v749_data = ir2[0];
                ir2[0] = (v749_data + (v745_data * (sycl::group_broadcast(item.get_sub_group(), v746_data, 0))));
                float v755_data = ir2[1];
                ir2[1] = (v755_data + (v745_data * (sycl::group_broadcast(item.get_sub_group(), v746_data, 1))));
                float v761_data = ir2[2];
                ir2[2] = (v761_data + (v745_data * (sycl::group_broadcast(item.get_sub_group(), v746_data, 2))));
                float v767_data = ir2[3];
                ir2[3] = (v767_data + (v745_data * (sycl::group_broadcast(item.get_sub_group(), v746_data, 3))));
                float v773_data = ir2[4];
                ir2[4] = (v773_data + (v745_data * (sycl::group_broadcast(item.get_sub_group(), v746_data, 4))));
                float v779_data = ir2[5];
                ir2[5] = (v779_data + (v745_data * (sycl::group_broadcast(item.get_sub_group(), v746_data, 5))));
                float v785_data = ir2[6];
                ir2[6] = (v785_data + (v745_data * (sycl::group_broadcast(item.get_sub_group(), v746_data, 6))));
                float v791_data = ir2[7];
                ir2[7] = (v791_data + (v745_data * (sycl::group_broadcast(item.get_sub_group(), v746_data, 7))));
                float v797_data = ir2[8];
                ir2[8] = (v797_data + (v745_data * (sycl::group_broadcast(item.get_sub_group(), v746_data, 8))));
                float v803_data = ir2[9];
                ir2[9] = (v803_data + (v745_data * (sycl::group_broadcast(item.get_sub_group(), v746_data, 9))));
                float v809_data = ir2[10];
                ir2[10] = (v809_data + (v745_data * (sycl::group_broadcast(item.get_sub_group(), v746_data, 10))));
                float v815_data = ir2[11];
                ir2[11] = (v815_data + (v745_data * (sycl::group_broadcast(item.get_sub_group(), v746_data, 11))));
                float v821_data = ir2[12];
                ir2[12] = (v821_data + (v745_data * (sycl::group_broadcast(item.get_sub_group(), v746_data, 12))));
                float v827_data = ir2[13];
                ir2[13] = (v827_data + (v745_data * (sycl::group_broadcast(item.get_sub_group(), v746_data, 13))));
                float v833_data = ir2[14];
                ir2[14] = (v833_data + (v745_data * (sycl::group_broadcast(item.get_sub_group(), v746_data, 14))));
                float v839_data = ir2[15];
                ir2[15] = (v839_data + (v745_data * (sycl::group_broadcast(item.get_sub_group(), v746_data, 15))));
              }
              if (v12_lead < 12) {
                float v845_data = r0[8];
                float v846_data = r1[8];
                float v849_data = ir2[0];
                ir2[0] = (v849_data + (v845_data * (sycl::group_broadcast(item.get_sub_group(), v846_data, 0))));
                float v855_data = ir2[1];
                ir2[1] = (v855_data + (v845_data * (sycl::group_broadcast(item.get_sub_group(), v846_data, 1))));
                float v861_data = ir2[2];
                ir2[2] = (v861_data + (v845_data * (sycl::group_broadcast(item.get_sub_group(), v846_data, 2))));
                float v867_data = ir2[3];
                ir2[3] = (v867_data + (v845_data * (sycl::group_broadcast(item.get_sub_group(), v846_data, 3))));
                float v873_data = ir2[4];
                ir2[4] = (v873_data + (v845_data * (sycl::group_broadcast(item.get_sub_group(), v846_data, 4))));
                float v879_data = ir2[5];
                ir2[5] = (v879_data + (v845_data * (sycl::group_broadcast(item.get_sub_group(), v846_data, 5))));
                float v885_data = ir2[6];
                ir2[6] = (v885_data + (v845_data * (sycl::group_broadcast(item.get_sub_group(), v846_data, 6))));
                float v891_data = ir2[7];
                ir2[7] = (v891_data + (v845_data * (sycl::group_broadcast(item.get_sub_group(), v846_data, 7))));
                float v897_data = ir2[8];
                ir2[8] = (v897_data + (v845_data * (sycl::group_broadcast(item.get_sub_group(), v846_data, 8))));
                float v903_data = ir2[9];
                ir2[9] = (v903_data + (v845_data * (sycl::group_broadcast(item.get_sub_group(), v846_data, 9))));
                float v909_data = ir2[10];
                ir2[10] = (v909_data + (v845_data * (sycl::group_broadcast(item.get_sub_group(), v846_data, 10))));
                float v915_data = ir2[11];
                ir2[11] = (v915_data + (v845_data * (sycl::group_broadcast(item.get_sub_group(), v846_data, 11))));
                float v921_data = ir2[12];
                ir2[12] = (v921_data + (v845_data * (sycl::group_broadcast(item.get_sub_group(), v846_data, 12))));
                float v927_data = ir2[13];
                ir2[13] = (v927_data + (v845_data * (sycl::group_broadcast(item.get_sub_group(), v846_data, 13))));
                float v933_data = ir2[14];
                ir2[14] = (v933_data + (v845_data * (sycl::group_broadcast(item.get_sub_group(), v846_data, 14))));
                float v939_data = ir2[15];
                ir2[15] = (v939_data + (v845_data * (sycl::group_broadcast(item.get_sub_group(), v846_data, 15))));
              }
              if (v12_lead < 12) {
                float v945_data = r0[9];
                float v946_data = r1[9];
                float v949_data = ir2[0];
                ir2[0] = (v949_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v946_data, 0))));
                float v955_data = ir2[1];
                ir2[1] = (v955_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v946_data, 1))));
                float v961_data = ir2[2];
                ir2[2] = (v961_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v946_data, 2))));
                float v967_data = ir2[3];
                ir2[3] = (v967_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v946_data, 3))));
                float v973_data = ir2[4];
                ir2[4] = (v973_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v946_data, 4))));
                float v979_data = ir2[5];
                ir2[5] = (v979_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v946_data, 5))));
                float v985_data = ir2[6];
                ir2[6] = (v985_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v946_data, 6))));
                float v991_data = ir2[7];
                ir2[7] = (v991_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v946_data, 7))));
                float v997_data = ir2[8];
                ir2[8] = (v997_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v946_data, 8))));
                float v1003_data = ir2[9];
                ir2[9] = (v1003_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v946_data, 9))));
                float v1009_data = ir2[10];
                ir2[10] = (v1009_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v946_data, 10))));
                float v1015_data = ir2[11];
                ir2[11] = (v1015_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v946_data, 11))));
                float v1021_data = ir2[12];
                ir2[12] = (v1021_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v946_data, 12))));
                float v1027_data = ir2[13];
                ir2[13] = (v1027_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v946_data, 13))));
                float v1033_data = ir2[14];
                ir2[14] = (v1033_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v946_data, 14))));
                float v1039_data = ir2[15];
                ir2[15] = (v1039_data + (v945_data * (sycl::group_broadcast(item.get_sub_group(), v946_data, 15))));
              }
              if (v12_lead < 12) {
                float v1045_data = r0[10];
                float v1046_data = r1[10];
                float v1049_data = ir2[0];
                ir2[0] = (v1049_data + (v1045_data * (sycl::group_broadcast(item.get_sub_group(), v1046_data, 0))));
                float v1055_data = ir2[1];
                ir2[1] = (v1055_data + (v1045_data * (sycl::group_broadcast(item.get_sub_group(), v1046_data, 1))));
                float v1061_data = ir2[2];
                ir2[2] = (v1061_data + (v1045_data * (sycl::group_broadcast(item.get_sub_group(), v1046_data, 2))));
                float v1067_data = ir2[3];
                ir2[3] = (v1067_data + (v1045_data * (sycl::group_broadcast(item.get_sub_group(), v1046_data, 3))));
                float v1073_data = ir2[4];
                ir2[4] = (v1073_data + (v1045_data * (sycl::group_broadcast(item.get_sub_group(), v1046_data, 4))));
                float v1079_data = ir2[5];
                ir2[5] = (v1079_data + (v1045_data * (sycl::group_broadcast(item.get_sub_group(), v1046_data, 5))));
                float v1085_data = ir2[6];
                ir2[6] = (v1085_data + (v1045_data * (sycl::group_broadcast(item.get_sub_group(), v1046_data, 6))));
                float v1091_data = ir2[7];
                ir2[7] = (v1091_data + (v1045_data * (sycl::group_broadcast(item.get_sub_group(), v1046_data, 7))));
                float v1097_data = ir2[8];
                ir2[8] = (v1097_data + (v1045_data * (sycl::group_broadcast(item.get_sub_group(), v1046_data, 8))));
                float v1103_data = ir2[9];
                ir2[9] = (v1103_data + (v1045_data * (sycl::group_broadcast(item.get_sub_group(), v1046_data, 9))));
                float v1109_data = ir2[10];
                ir2[10] = (v1109_data + (v1045_data * (sycl::group_broadcast(item.get_sub_group(), v1046_data, 10))));
                float v1115_data = ir2[11];
                ir2[11] = (v1115_data + (v1045_data * (sycl::group_broadcast(item.get_sub_group(), v1046_data, 11))));
                float v1121_data = ir2[12];
                ir2[12] = (v1121_data + (v1045_data * (sycl::group_broadcast(item.get_sub_group(), v1046_data, 12))));
                float v1127_data = ir2[13];
                ir2[13] = (v1127_data + (v1045_data * (sycl::group_broadcast(item.get_sub_group(), v1046_data, 13))));
                float v1133_data = ir2[14];
                ir2[14] = (v1133_data + (v1045_data * (sycl::group_broadcast(item.get_sub_group(), v1046_data, 14))));
                float v1139_data = ir2[15];
                ir2[15] = (v1139_data + (v1045_data * (sycl::group_broadcast(item.get_sub_group(), v1046_data, 15))));
              }
              if (v12_lead < 12) {
                float v1145_data = r0[11];
                float v1146_data = r1[11];
                float v1149_data = ir2[0];
                ir2[0] = (v1149_data + (v1145_data * (sycl::group_broadcast(item.get_sub_group(), v1146_data, 0))));
                float v1155_data = ir2[1];
                ir2[1] = (v1155_data + (v1145_data * (sycl::group_broadcast(item.get_sub_group(), v1146_data, 1))));
                float v1161_data = ir2[2];
                ir2[2] = (v1161_data + (v1145_data * (sycl::group_broadcast(item.get_sub_group(), v1146_data, 2))));
                float v1167_data = ir2[3];
                ir2[3] = (v1167_data + (v1145_data * (sycl::group_broadcast(item.get_sub_group(), v1146_data, 3))));
                float v1173_data = ir2[4];
                ir2[4] = (v1173_data + (v1145_data * (sycl::group_broadcast(item.get_sub_group(), v1146_data, 4))));
                float v1179_data = ir2[5];
                ir2[5] = (v1179_data + (v1145_data * (sycl::group_broadcast(item.get_sub_group(), v1146_data, 5))));
                float v1185_data = ir2[6];
                ir2[6] = (v1185_data + (v1145_data * (sycl::group_broadcast(item.get_sub_group(), v1146_data, 6))));
                float v1191_data = ir2[7];
                ir2[7] = (v1191_data + (v1145_data * (sycl::group_broadcast(item.get_sub_group(), v1146_data, 7))));
                float v1197_data = ir2[8];
                ir2[8] = (v1197_data + (v1145_data * (sycl::group_broadcast(item.get_sub_group(), v1146_data, 8))));
                float v1203_data = ir2[9];
                ir2[9] = (v1203_data + (v1145_data * (sycl::group_broadcast(item.get_sub_group(), v1146_data, 9))));
                float v1209_data = ir2[10];
                ir2[10] = (v1209_data + (v1145_data * (sycl::group_broadcast(item.get_sub_group(), v1146_data, 10))));
                float v1215_data = ir2[11];
                ir2[11] = (v1215_data + (v1145_data * (sycl::group_broadcast(item.get_sub_group(), v1146_data, 11))));
                float v1221_data = ir2[12];
                ir2[12] = (v1221_data + (v1145_data * (sycl::group_broadcast(item.get_sub_group(), v1146_data, 12))));
                float v1227_data = ir2[13];
                ir2[13] = (v1227_data + (v1145_data * (sycl::group_broadcast(item.get_sub_group(), v1146_data, 13))));
                float v1233_data = ir2[14];
                ir2[14] = (v1233_data + (v1145_data * (sycl::group_broadcast(item.get_sub_group(), v1146_data, 14))));
                float v1239_data = ir2[15];
                ir2[15] = (v1239_data + (v1145_data * (sycl::group_broadcast(item.get_sub_group(), v1146_data, 15))));
              }
              if (v12_lead < 12) {
                float v1245_data = r0[12];
                float v1246_data = r1[12];
                float v1249_data = ir2[0];
                ir2[0] = (v1249_data + (v1245_data * (sycl::group_broadcast(item.get_sub_group(), v1246_data, 0))));
                float v1255_data = ir2[1];
                ir2[1] = (v1255_data + (v1245_data * (sycl::group_broadcast(item.get_sub_group(), v1246_data, 1))));
                float v1261_data = ir2[2];
                ir2[2] = (v1261_data + (v1245_data * (sycl::group_broadcast(item.get_sub_group(), v1246_data, 2))));
                float v1267_data = ir2[3];
                ir2[3] = (v1267_data + (v1245_data * (sycl::group_broadcast(item.get_sub_group(), v1246_data, 3))));
                float v1273_data = ir2[4];
                ir2[4] = (v1273_data + (v1245_data * (sycl::group_broadcast(item.get_sub_group(), v1246_data, 4))));
                float v1279_data = ir2[5];
                ir2[5] = (v1279_data + (v1245_data * (sycl::group_broadcast(item.get_sub_group(), v1246_data, 5))));
                float v1285_data = ir2[6];
                ir2[6] = (v1285_data + (v1245_data * (sycl::group_broadcast(item.get_sub_group(), v1246_data, 6))));
                float v1291_data = ir2[7];
                ir2[7] = (v1291_data + (v1245_data * (sycl::group_broadcast(item.get_sub_group(), v1246_data, 7))));
                float v1297_data = ir2[8];
                ir2[8] = (v1297_data + (v1245_data * (sycl::group_broadcast(item.get_sub_group(), v1246_data, 8))));
                float v1303_data = ir2[9];
                ir2[9] = (v1303_data + (v1245_data * (sycl::group_broadcast(item.get_sub_group(), v1246_data, 9))));
                float v1309_data = ir2[10];
                ir2[10] = (v1309_data + (v1245_data * (sycl::group_broadcast(item.get_sub_group(), v1246_data, 10))));
                float v1315_data = ir2[11];
                ir2[11] = (v1315_data + (v1245_data * (sycl::group_broadcast(item.get_sub_group(), v1246_data, 11))));
                float v1321_data = ir2[12];
                ir2[12] = (v1321_data + (v1245_data * (sycl::group_broadcast(item.get_sub_group(), v1246_data, 12))));
                float v1327_data = ir2[13];
                ir2[13] = (v1327_data + (v1245_data * (sycl::group_broadcast(item.get_sub_group(), v1246_data, 13))));
                float v1333_data = ir2[14];
                ir2[14] = (v1333_data + (v1245_data * (sycl::group_broadcast(item.get_sub_group(), v1246_data, 14))));
                float v1339_data = ir2[15];
                ir2[15] = (v1339_data + (v1245_data * (sycl::group_broadcast(item.get_sub_group(), v1246_data, 15))));
              }
              if (v12_lead < 12) {
                float v1345_data = r0[13];
                float v1346_data = r1[13];
                float v1349_data = ir2[0];
                ir2[0] = (v1349_data + (v1345_data * (sycl::group_broadcast(item.get_sub_group(), v1346_data, 0))));
                float v1355_data = ir2[1];
                ir2[1] = (v1355_data + (v1345_data * (sycl::group_broadcast(item.get_sub_group(), v1346_data, 1))));
                float v1361_data = ir2[2];
                ir2[2] = (v1361_data + (v1345_data * (sycl::group_broadcast(item.get_sub_group(), v1346_data, 2))));
                float v1367_data = ir2[3];
                ir2[3] = (v1367_data + (v1345_data * (sycl::group_broadcast(item.get_sub_group(), v1346_data, 3))));
                float v1373_data = ir2[4];
                ir2[4] = (v1373_data + (v1345_data * (sycl::group_broadcast(item.get_sub_group(), v1346_data, 4))));
                float v1379_data = ir2[5];
                ir2[5] = (v1379_data + (v1345_data * (sycl::group_broadcast(item.get_sub_group(), v1346_data, 5))));
                float v1385_data = ir2[6];
                ir2[6] = (v1385_data + (v1345_data * (sycl::group_broadcast(item.get_sub_group(), v1346_data, 6))));
                float v1391_data = ir2[7];
                ir2[7] = (v1391_data + (v1345_data * (sycl::group_broadcast(item.get_sub_group(), v1346_data, 7))));
                float v1397_data = ir2[8];
                ir2[8] = (v1397_data + (v1345_data * (sycl::group_broadcast(item.get_sub_group(), v1346_data, 8))));
                float v1403_data = ir2[9];
                ir2[9] = (v1403_data + (v1345_data * (sycl::group_broadcast(item.get_sub_group(), v1346_data, 9))));
                float v1409_data = ir2[10];
                ir2[10] = (v1409_data + (v1345_data * (sycl::group_broadcast(item.get_sub_group(), v1346_data, 10))));
                float v1415_data = ir2[11];
                ir2[11] = (v1415_data + (v1345_data * (sycl::group_broadcast(item.get_sub_group(), v1346_data, 11))));
                float v1421_data = ir2[12];
                ir2[12] = (v1421_data + (v1345_data * (sycl::group_broadcast(item.get_sub_group(), v1346_data, 12))));
                float v1427_data = ir2[13];
                ir2[13] = (v1427_data + (v1345_data * (sycl::group_broadcast(item.get_sub_group(), v1346_data, 13))));
                float v1433_data = ir2[14];
                ir2[14] = (v1433_data + (v1345_data * (sycl::group_broadcast(item.get_sub_group(), v1346_data, 14))));
                float v1439_data = ir2[15];
                ir2[15] = (v1439_data + (v1345_data * (sycl::group_broadcast(item.get_sub_group(), v1346_data, 15))));
              }
              if (v12_lead < 12) {
                float v1445_data = r0[14];
                float v1446_data = r1[14];
                float v1449_data = ir2[0];
                ir2[0] = (v1449_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1446_data, 0))));
                float v1455_data = ir2[1];
                ir2[1] = (v1455_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1446_data, 1))));
                float v1461_data = ir2[2];
                ir2[2] = (v1461_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1446_data, 2))));
                float v1467_data = ir2[3];
                ir2[3] = (v1467_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1446_data, 3))));
                float v1473_data = ir2[4];
                ir2[4] = (v1473_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1446_data, 4))));
                float v1479_data = ir2[5];
                ir2[5] = (v1479_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1446_data, 5))));
                float v1485_data = ir2[6];
                ir2[6] = (v1485_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1446_data, 6))));
                float v1491_data = ir2[7];
                ir2[7] = (v1491_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1446_data, 7))));
                float v1497_data = ir2[8];
                ir2[8] = (v1497_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1446_data, 8))));
                float v1503_data = ir2[9];
                ir2[9] = (v1503_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1446_data, 9))));
                float v1509_data = ir2[10];
                ir2[10] = (v1509_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1446_data, 10))));
                float v1515_data = ir2[11];
                ir2[11] = (v1515_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1446_data, 11))));
                float v1521_data = ir2[12];
                ir2[12] = (v1521_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1446_data, 12))));
                float v1527_data = ir2[13];
                ir2[13] = (v1527_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1446_data, 13))));
                float v1533_data = ir2[14];
                ir2[14] = (v1533_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1446_data, 14))));
                float v1539_data = ir2[15];
                ir2[15] = (v1539_data + (v1445_data * (sycl::group_broadcast(item.get_sub_group(), v1446_data, 15))));
              }
              if (v12_lead < 12) {
                float v1545_data = r0[15];
                float v1546_data = r1[15];
                float v1549_data = ir2[0];
                ir2[0] = (v1549_data + (v1545_data * (sycl::group_broadcast(item.get_sub_group(), v1546_data, 0))));
                float v1555_data = ir2[1];
                ir2[1] = (v1555_data + (v1545_data * (sycl::group_broadcast(item.get_sub_group(), v1546_data, 1))));
                float v1561_data = ir2[2];
                ir2[2] = (v1561_data + (v1545_data * (sycl::group_broadcast(item.get_sub_group(), v1546_data, 2))));
                float v1567_data = ir2[3];
                ir2[3] = (v1567_data + (v1545_data * (sycl::group_broadcast(item.get_sub_group(), v1546_data, 3))));
                float v1573_data = ir2[4];
                ir2[4] = (v1573_data + (v1545_data * (sycl::group_broadcast(item.get_sub_group(), v1546_data, 4))));
                float v1579_data = ir2[5];
                ir2[5] = (v1579_data + (v1545_data * (sycl::group_broadcast(item.get_sub_group(), v1546_data, 5))));
                float v1585_data = ir2[6];
                ir2[6] = (v1585_data + (v1545_data * (sycl::group_broadcast(item.get_sub_group(), v1546_data, 6))));
                float v1591_data = ir2[7];
                ir2[7] = (v1591_data + (v1545_data * (sycl::group_broadcast(item.get_sub_group(), v1546_data, 7))));
                float v1597_data = ir2[8];
                ir2[8] = (v1597_data + (v1545_data * (sycl::group_broadcast(item.get_sub_group(), v1546_data, 8))));
                float v1603_data = ir2[9];
                ir2[9] = (v1603_data + (v1545_data * (sycl::group_broadcast(item.get_sub_group(), v1546_data, 9))));
                float v1609_data = ir2[10];
                ir2[10] = (v1609_data + (v1545_data * (sycl::group_broadcast(item.get_sub_group(), v1546_data, 10))));
                float v1615_data = ir2[11];
                ir2[11] = (v1615_data + (v1545_data * (sycl::group_broadcast(item.get_sub_group(), v1546_data, 11))));
                float v1621_data = ir2[12];
                ir2[12] = (v1621_data + (v1545_data * (sycl::group_broadcast(item.get_sub_group(), v1546_data, 12))));
                float v1627_data = ir2[13];
                ir2[13] = (v1627_data + (v1545_data * (sycl::group_broadcast(item.get_sub_group(), v1546_data, 13))));
                float v1633_data = ir2[14];
                ir2[14] = (v1633_data + (v1545_data * (sycl::group_broadcast(item.get_sub_group(), v1546_data, 14))));
                float v1639_data = ir2[15];
                ir2[15] = (v1639_data + (v1545_data * (sycl::group_broadcast(item.get_sub_group(), v1546_data, 15))));
              }
              if (v12_lead < 12) {
                float v1645_data = r0[16];
                float v1646_data = r1[16];
                float v1649_data = ir2[0];
                ir2[0] = (v1649_data + (v1645_data * (sycl::group_broadcast(item.get_sub_group(), v1646_data, 0))));
                float v1655_data = ir2[1];
                ir2[1] = (v1655_data + (v1645_data * (sycl::group_broadcast(item.get_sub_group(), v1646_data, 1))));
                float v1661_data = ir2[2];
                ir2[2] = (v1661_data + (v1645_data * (sycl::group_broadcast(item.get_sub_group(), v1646_data, 2))));
                float v1667_data = ir2[3];
                ir2[3] = (v1667_data + (v1645_data * (sycl::group_broadcast(item.get_sub_group(), v1646_data, 3))));
                float v1673_data = ir2[4];
                ir2[4] = (v1673_data + (v1645_data * (sycl::group_broadcast(item.get_sub_group(), v1646_data, 4))));
                float v1679_data = ir2[5];
                ir2[5] = (v1679_data + (v1645_data * (sycl::group_broadcast(item.get_sub_group(), v1646_data, 5))));
                float v1685_data = ir2[6];
                ir2[6] = (v1685_data + (v1645_data * (sycl::group_broadcast(item.get_sub_group(), v1646_data, 6))));
                float v1691_data = ir2[7];
                ir2[7] = (v1691_data + (v1645_data * (sycl::group_broadcast(item.get_sub_group(), v1646_data, 7))));
                float v1697_data = ir2[8];
                ir2[8] = (v1697_data + (v1645_data * (sycl::group_broadcast(item.get_sub_group(), v1646_data, 8))));
                float v1703_data = ir2[9];
                ir2[9] = (v1703_data + (v1645_data * (sycl::group_broadcast(item.get_sub_group(), v1646_data, 9))));
                float v1709_data = ir2[10];
                ir2[10] = (v1709_data + (v1645_data * (sycl::group_broadcast(item.get_sub_group(), v1646_data, 10))));
                float v1715_data = ir2[11];
                ir2[11] = (v1715_data + (v1645_data * (sycl::group_broadcast(item.get_sub_group(), v1646_data, 11))));
                float v1721_data = ir2[12];
                ir2[12] = (v1721_data + (v1645_data * (sycl::group_broadcast(item.get_sub_group(), v1646_data, 12))));
                float v1727_data = ir2[13];
                ir2[13] = (v1727_data + (v1645_data * (sycl::group_broadcast(item.get_sub_group(), v1646_data, 13))));
                float v1733_data = ir2[14];
                ir2[14] = (v1733_data + (v1645_data * (sycl::group_broadcast(item.get_sub_group(), v1646_data, 14))));
                float v1739_data = ir2[15];
                ir2[15] = (v1739_data + (v1645_data * (sycl::group_broadcast(item.get_sub_group(), v1646_data, 15))));
              }
              if (v12_lead < 12) {
                float v1745_data = r0[17];
                float v1746_data = r1[17];
                float v1749_data = ir2[0];
                ir2[0] = (v1749_data + (v1745_data * (sycl::group_broadcast(item.get_sub_group(), v1746_data, 0))));
                float v1755_data = ir2[1];
                ir2[1] = (v1755_data + (v1745_data * (sycl::group_broadcast(item.get_sub_group(), v1746_data, 1))));
                float v1761_data = ir2[2];
                ir2[2] = (v1761_data + (v1745_data * (sycl::group_broadcast(item.get_sub_group(), v1746_data, 2))));
                float v1767_data = ir2[3];
                ir2[3] = (v1767_data + (v1745_data * (sycl::group_broadcast(item.get_sub_group(), v1746_data, 3))));
                float v1773_data = ir2[4];
                ir2[4] = (v1773_data + (v1745_data * (sycl::group_broadcast(item.get_sub_group(), v1746_data, 4))));
                float v1779_data = ir2[5];
                ir2[5] = (v1779_data + (v1745_data * (sycl::group_broadcast(item.get_sub_group(), v1746_data, 5))));
                float v1785_data = ir2[6];
                ir2[6] = (v1785_data + (v1745_data * (sycl::group_broadcast(item.get_sub_group(), v1746_data, 6))));
                float v1791_data = ir2[7];
                ir2[7] = (v1791_data + (v1745_data * (sycl::group_broadcast(item.get_sub_group(), v1746_data, 7))));
                float v1797_data = ir2[8];
                ir2[8] = (v1797_data + (v1745_data * (sycl::group_broadcast(item.get_sub_group(), v1746_data, 8))));
                float v1803_data = ir2[9];
                ir2[9] = (v1803_data + (v1745_data * (sycl::group_broadcast(item.get_sub_group(), v1746_data, 9))));
                float v1809_data = ir2[10];
                ir2[10] = (v1809_data + (v1745_data * (sycl::group_broadcast(item.get_sub_group(), v1746_data, 10))));
                float v1815_data = ir2[11];
                ir2[11] = (v1815_data + (v1745_data * (sycl::group_broadcast(item.get_sub_group(), v1746_data, 11))));
                float v1821_data = ir2[12];
                ir2[12] = (v1821_data + (v1745_data * (sycl::group_broadcast(item.get_sub_group(), v1746_data, 12))));
                float v1827_data = ir2[13];
                ir2[13] = (v1827_data + (v1745_data * (sycl::group_broadcast(item.get_sub_group(), v1746_data, 13))));
                float v1833_data = ir2[14];
                ir2[14] = (v1833_data + (v1745_data * (sycl::group_broadcast(item.get_sub_group(), v1746_data, 14))));
                float v1839_data = ir2[15];
                ir2[15] = (v1839_data + (v1745_data * (sycl::group_broadcast(item.get_sub_group(), v1746_data, 15))));
              }
              if (v12_lead < 12) {
                float v1845_data = r0[18];
                float v1846_data = r1[18];
                float v1849_data = ir2[0];
                ir2[0] = (v1849_data + (v1845_data * (sycl::group_broadcast(item.get_sub_group(), v1846_data, 0))));
                float v1855_data = ir2[1];
                ir2[1] = (v1855_data + (v1845_data * (sycl::group_broadcast(item.get_sub_group(), v1846_data, 1))));
                float v1861_data = ir2[2];
                ir2[2] = (v1861_data + (v1845_data * (sycl::group_broadcast(item.get_sub_group(), v1846_data, 2))));
                float v1867_data = ir2[3];
                ir2[3] = (v1867_data + (v1845_data * (sycl::group_broadcast(item.get_sub_group(), v1846_data, 3))));
                float v1873_data = ir2[4];
                ir2[4] = (v1873_data + (v1845_data * (sycl::group_broadcast(item.get_sub_group(), v1846_data, 4))));
                float v1879_data = ir2[5];
                ir2[5] = (v1879_data + (v1845_data * (sycl::group_broadcast(item.get_sub_group(), v1846_data, 5))));
                float v1885_data = ir2[6];
                ir2[6] = (v1885_data + (v1845_data * (sycl::group_broadcast(item.get_sub_group(), v1846_data, 6))));
                float v1891_data = ir2[7];
                ir2[7] = (v1891_data + (v1845_data * (sycl::group_broadcast(item.get_sub_group(), v1846_data, 7))));
                float v1897_data = ir2[8];
                ir2[8] = (v1897_data + (v1845_data * (sycl::group_broadcast(item.get_sub_group(), v1846_data, 8))));
                float v1903_data = ir2[9];
                ir2[9] = (v1903_data + (v1845_data * (sycl::group_broadcast(item.get_sub_group(), v1846_data, 9))));
                float v1909_data = ir2[10];
                ir2[10] = (v1909_data + (v1845_data * (sycl::group_broadcast(item.get_sub_group(), v1846_data, 10))));
                float v1915_data = ir2[11];
                ir2[11] = (v1915_data + (v1845_data * (sycl::group_broadcast(item.get_sub_group(), v1846_data, 11))));
                float v1921_data = ir2[12];
                ir2[12] = (v1921_data + (v1845_data * (sycl::group_broadcast(item.get_sub_group(), v1846_data, 12))));
                float v1927_data = ir2[13];
                ir2[13] = (v1927_data + (v1845_data * (sycl::group_broadcast(item.get_sub_group(), v1846_data, 13))));
                float v1933_data = ir2[14];
                ir2[14] = (v1933_data + (v1845_data * (sycl::group_broadcast(item.get_sub_group(), v1846_data, 14))));
                float v1939_data = ir2[15];
                ir2[15] = (v1939_data + (v1845_data * (sycl::group_broadcast(item.get_sub_group(), v1846_data, 15))));
              }
              if (v12_lead < 12) {
                float v1945_data = r0[19];
                float v1946_data = r1[19];
                float v1949_data = ir2[0];
                ir2[0] = (v1949_data + (v1945_data * (sycl::group_broadcast(item.get_sub_group(), v1946_data, 0))));
                float v1955_data = ir2[1];
                ir2[1] = (v1955_data + (v1945_data * (sycl::group_broadcast(item.get_sub_group(), v1946_data, 1))));
                float v1961_data = ir2[2];
                ir2[2] = (v1961_data + (v1945_data * (sycl::group_broadcast(item.get_sub_group(), v1946_data, 2))));
                float v1967_data = ir2[3];
                ir2[3] = (v1967_data + (v1945_data * (sycl::group_broadcast(item.get_sub_group(), v1946_data, 3))));
                float v1973_data = ir2[4];
                ir2[4] = (v1973_data + (v1945_data * (sycl::group_broadcast(item.get_sub_group(), v1946_data, 4))));
                float v1979_data = ir2[5];
                ir2[5] = (v1979_data + (v1945_data * (sycl::group_broadcast(item.get_sub_group(), v1946_data, 5))));
                float v1985_data = ir2[6];
                ir2[6] = (v1985_data + (v1945_data * (sycl::group_broadcast(item.get_sub_group(), v1946_data, 6))));
                float v1991_data = ir2[7];
                ir2[7] = (v1991_data + (v1945_data * (sycl::group_broadcast(item.get_sub_group(), v1946_data, 7))));
                float v1997_data = ir2[8];
                ir2[8] = (v1997_data + (v1945_data * (sycl::group_broadcast(item.get_sub_group(), v1946_data, 8))));
                float v2003_data = ir2[9];
                ir2[9] = (v2003_data + (v1945_data * (sycl::group_broadcast(item.get_sub_group(), v1946_data, 9))));
                float v2009_data = ir2[10];
                ir2[10] = (v2009_data + (v1945_data * (sycl::group_broadcast(item.get_sub_group(), v1946_data, 10))));
                float v2015_data = ir2[11];
                ir2[11] = (v2015_data + (v1945_data * (sycl::group_broadcast(item.get_sub_group(), v1946_data, 11))));
                float v2021_data = ir2[12];
                ir2[12] = (v2021_data + (v1945_data * (sycl::group_broadcast(item.get_sub_group(), v1946_data, 12))));
                float v2027_data = ir2[13];
                ir2[13] = (v2027_data + (v1945_data * (sycl::group_broadcast(item.get_sub_group(), v1946_data, 13))));
                float v2033_data = ir2[14];
                ir2[14] = (v2033_data + (v1945_data * (sycl::group_broadcast(item.get_sub_group(), v1946_data, 14))));
                float v2039_data = ir2[15];
                ir2[15] = (v2039_data + (v1945_data * (sycl::group_broadcast(item.get_sub_group(), v1946_data, 15))));
              }
              if (v12_lead < 12) {
                #pragma unroll
                for (int32_t v2045_n1 = 0; v2045_n1 < 16; ++v2045_n1) {
                  float v2047_data = ir2[v2045_n1];
                  r2[v2045_n1] = v2047_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              if (v12_lead < 12) {
                #pragma unroll
                for (int32_t v2053_i1 = 0; v2053_i1 < 16; ++v2053_i1) {
                  float v2055_data = r2[v2053_i1];
                  glb_m0[(v12_lead + (v2053_i1 * 12))] = v2055_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

