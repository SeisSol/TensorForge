// === base name ===
kernel_5ce24442b80ccb8a

// === header ===
void launcher_kernel_5ce24442b80ccb8a(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_5ce24442b80ccb8a(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
  sycl::range<3> block (16, 16, 1);
  sycl::range<3> grid (std::min(gridsize, numElements0), 1, 1);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_5ce24442b80ccb8a(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_5ce24442b80ccb8a(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8×8(8×8) {0..8}×{0..8} strided
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // C = abs(TMP)
        {
          const auto batchId_start = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[80 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          float* __restrict__ s0 = &localShrMem0[0];
          for (size_t batchId0 = item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)); batchId0 < numElements0; batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            const auto batchId1 = batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId0;
            const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[batchId0 * 64 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[batchId0 * 64 + 0 + m2_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v13_lead = item.get_local_id(0) % 16;
              if (v13_lead < 8) {
                #pragma unroll
                for (int32_t v15_i1 = 0; v15_i1 < 8; ++v15_i1) {
                  float v23_data = glb_m0[(v13_lead + (v15_i1 * 8))];
                  r0[v15_i1] = v23_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m1);
              if (v13_lead < 8) {
                #pragma unroll
                for (int32_t v30_i1 = 0; v30_i1 < 8; ++v30_i1) {
                  float v38_data = glb_m1[(v13_lead + (v30_i1 * 8))];
                  r1[v30_i1] = v38_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              // wait(r1 = load{g>r}(glb_m1););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              if (v13_lead < 8) {
                float v45_data = r0[0];
                float v46_data = r1[0];
                float v49_data = r2[0];
                r2[0] = (v49_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v46_data, 0))));
                float v52_data = r1[1];
                float v55_data = r2[1];
                r2[1] = (v55_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v52_data, 0))));
                float v58_data = r1[2];
                float v61_data = r2[2];
                r2[2] = (v61_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v58_data, 0))));
                float v64_data = r1[3];
                float v67_data = r2[3];
                r2[3] = (v67_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v64_data, 0))));
                float v70_data = r1[4];
                float v73_data = r2[4];
                r2[4] = (v73_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v70_data, 0))));
                float v76_data = r1[5];
                float v79_data = r2[5];
                r2[5] = (v79_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v76_data, 0))));
                float v82_data = r1[6];
                float v85_data = r2[6];
                r2[6] = (v85_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v82_data, 0))));
                float v88_data = r1[7];
                float v91_data = r2[7];
                r2[7] = (v91_data + (v45_data * (sycl::group_broadcast(item.get_sub_group(), v88_data, 0))));
              }
              if (v13_lead < 8) {
                float v97_data = r0[1];
                float v98_data = r1[0];
                float v101_data = r2[0];
                r2[0] = (v101_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v98_data, 1))));
                float v104_data = r1[1];
                float v107_data = r2[1];
                r2[1] = (v107_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v104_data, 1))));
                float v110_data = r1[2];
                float v113_data = r2[2];
                r2[2] = (v113_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v110_data, 1))));
                float v116_data = r1[3];
                float v119_data = r2[3];
                r2[3] = (v119_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v116_data, 1))));
                float v122_data = r1[4];
                float v125_data = r2[4];
                r2[4] = (v125_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v122_data, 1))));
                float v128_data = r1[5];
                float v131_data = r2[5];
                r2[5] = (v131_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v128_data, 1))));
                float v134_data = r1[6];
                float v137_data = r2[6];
                r2[6] = (v137_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v134_data, 1))));
                float v140_data = r1[7];
                float v143_data = r2[7];
                r2[7] = (v143_data + (v97_data * (sycl::group_broadcast(item.get_sub_group(), v140_data, 1))));
              }
              if (v13_lead < 8) {
                float v149_data = r0[2];
                float v150_data = r1[0];
                float v153_data = r2[0];
                r2[0] = (v153_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v150_data, 2))));
                float v156_data = r1[1];
                float v159_data = r2[1];
                r2[1] = (v159_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v156_data, 2))));
                float v162_data = r1[2];
                float v165_data = r2[2];
                r2[2] = (v165_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v162_data, 2))));
                float v168_data = r1[3];
                float v171_data = r2[3];
                r2[3] = (v171_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v168_data, 2))));
                float v174_data = r1[4];
                float v177_data = r2[4];
                r2[4] = (v177_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v174_data, 2))));
                float v180_data = r1[5];
                float v183_data = r2[5];
                r2[5] = (v183_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v180_data, 2))));
                float v186_data = r1[6];
                float v189_data = r2[6];
                r2[6] = (v189_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v186_data, 2))));
                float v192_data = r1[7];
                float v195_data = r2[7];
                r2[7] = (v195_data + (v149_data * (sycl::group_broadcast(item.get_sub_group(), v192_data, 2))));
              }
              if (v13_lead < 8) {
                float v201_data = r0[3];
                float v202_data = r1[0];
                float v205_data = r2[0];
                r2[0] = (v205_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v202_data, 3))));
                float v208_data = r1[1];
                float v211_data = r2[1];
                r2[1] = (v211_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v208_data, 3))));
                float v214_data = r1[2];
                float v217_data = r2[2];
                r2[2] = (v217_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v214_data, 3))));
                float v220_data = r1[3];
                float v223_data = r2[3];
                r2[3] = (v223_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v220_data, 3))));
                float v226_data = r1[4];
                float v229_data = r2[4];
                r2[4] = (v229_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v226_data, 3))));
                float v232_data = r1[5];
                float v235_data = r2[5];
                r2[5] = (v235_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v232_data, 3))));
                float v238_data = r1[6];
                float v241_data = r2[6];
                r2[6] = (v241_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v238_data, 3))));
                float v244_data = r1[7];
                float v247_data = r2[7];
                r2[7] = (v247_data + (v201_data * (sycl::group_broadcast(item.get_sub_group(), v244_data, 3))));
              }
              if (v13_lead < 8) {
                float v253_data = r0[4];
                float v254_data = r1[0];
                float v257_data = r2[0];
                r2[0] = (v257_data + (v253_data * (sycl::group_broadcast(item.get_sub_group(), v254_data, 4))));
                float v260_data = r1[1];
                float v263_data = r2[1];
                r2[1] = (v263_data + (v253_data * (sycl::group_broadcast(item.get_sub_group(), v260_data, 4))));
                float v266_data = r1[2];
                float v269_data = r2[2];
                r2[2] = (v269_data + (v253_data * (sycl::group_broadcast(item.get_sub_group(), v266_data, 4))));
                float v272_data = r1[3];
                float v275_data = r2[3];
                r2[3] = (v275_data + (v253_data * (sycl::group_broadcast(item.get_sub_group(), v272_data, 4))));
                float v278_data = r1[4];
                float v281_data = r2[4];
                r2[4] = (v281_data + (v253_data * (sycl::group_broadcast(item.get_sub_group(), v278_data, 4))));
                float v284_data = r1[5];
                float v287_data = r2[5];
                r2[5] = (v287_data + (v253_data * (sycl::group_broadcast(item.get_sub_group(), v284_data, 4))));
                float v290_data = r1[6];
                float v293_data = r2[6];
                r2[6] = (v293_data + (v253_data * (sycl::group_broadcast(item.get_sub_group(), v290_data, 4))));
                float v296_data = r1[7];
                float v299_data = r2[7];
                r2[7] = (v299_data + (v253_data * (sycl::group_broadcast(item.get_sub_group(), v296_data, 4))));
              }
              if (v13_lead < 8) {
                float v305_data = r0[5];
                float v306_data = r1[0];
                float v309_data = r2[0];
                r2[0] = (v309_data + (v305_data * (sycl::group_broadcast(item.get_sub_group(), v306_data, 5))));
                float v312_data = r1[1];
                float v315_data = r2[1];
                r2[1] = (v315_data + (v305_data * (sycl::group_broadcast(item.get_sub_group(), v312_data, 5))));
                float v318_data = r1[2];
                float v321_data = r2[2];
                r2[2] = (v321_data + (v305_data * (sycl::group_broadcast(item.get_sub_group(), v318_data, 5))));
                float v324_data = r1[3];
                float v327_data = r2[3];
                r2[3] = (v327_data + (v305_data * (sycl::group_broadcast(item.get_sub_group(), v324_data, 5))));
                float v330_data = r1[4];
                float v333_data = r2[4];
                r2[4] = (v333_data + (v305_data * (sycl::group_broadcast(item.get_sub_group(), v330_data, 5))));
                float v336_data = r1[5];
                float v339_data = r2[5];
                r2[5] = (v339_data + (v305_data * (sycl::group_broadcast(item.get_sub_group(), v336_data, 5))));
                float v342_data = r1[6];
                float v345_data = r2[6];
                r2[6] = (v345_data + (v305_data * (sycl::group_broadcast(item.get_sub_group(), v342_data, 5))));
                float v348_data = r1[7];
                float v351_data = r2[7];
                r2[7] = (v351_data + (v305_data * (sycl::group_broadcast(item.get_sub_group(), v348_data, 5))));
              }
              if (v13_lead < 8) {
                float v357_data = r0[6];
                float v358_data = r1[0];
                float v361_data = r2[0];
                r2[0] = (v361_data + (v357_data * (sycl::group_broadcast(item.get_sub_group(), v358_data, 6))));
                float v364_data = r1[1];
                float v367_data = r2[1];
                r2[1] = (v367_data + (v357_data * (sycl::group_broadcast(item.get_sub_group(), v364_data, 6))));
                float v370_data = r1[2];
                float v373_data = r2[2];
                r2[2] = (v373_data + (v357_data * (sycl::group_broadcast(item.get_sub_group(), v370_data, 6))));
                float v376_data = r1[3];
                float v379_data = r2[3];
                r2[3] = (v379_data + (v357_data * (sycl::group_broadcast(item.get_sub_group(), v376_data, 6))));
                float v382_data = r1[4];
                float v385_data = r2[4];
                r2[4] = (v385_data + (v357_data * (sycl::group_broadcast(item.get_sub_group(), v382_data, 6))));
                float v388_data = r1[5];
                float v391_data = r2[5];
                r2[5] = (v391_data + (v357_data * (sycl::group_broadcast(item.get_sub_group(), v388_data, 6))));
                float v394_data = r1[6];
                float v397_data = r2[6];
                r2[6] = (v397_data + (v357_data * (sycl::group_broadcast(item.get_sub_group(), v394_data, 6))));
                float v400_data = r1[7];
                float v403_data = r2[7];
                r2[7] = (v403_data + (v357_data * (sycl::group_broadcast(item.get_sub_group(), v400_data, 6))));
              }
              if (v13_lead < 8) {
                float v409_data = r0[7];
                float v410_data = r1[0];
                float v413_data = r2[0];
                r2[0] = (v413_data + (v409_data * (sycl::group_broadcast(item.get_sub_group(), v410_data, 7))));
                float v416_data = r1[1];
                float v419_data = r2[1];
                r2[1] = (v419_data + (v409_data * (sycl::group_broadcast(item.get_sub_group(), v416_data, 7))));
                float v422_data = r1[2];
                float v425_data = r2[2];
                r2[2] = (v425_data + (v409_data * (sycl::group_broadcast(item.get_sub_group(), v422_data, 7))));
                float v428_data = r1[3];
                float v431_data = r2[3];
                r2[3] = (v431_data + (v409_data * (sycl::group_broadcast(item.get_sub_group(), v428_data, 7))));
                float v434_data = r1[4];
                float v437_data = r2[4];
                r2[4] = (v437_data + (v409_data * (sycl::group_broadcast(item.get_sub_group(), v434_data, 7))));
                float v440_data = r1[5];
                float v443_data = r2[5];
                r2[5] = (v443_data + (v409_data * (sycl::group_broadcast(item.get_sub_group(), v440_data, 7))));
                float v446_data = r1[6];
                float v449_data = r2[6];
                r2[6] = (v449_data + (v409_data * (sycl::group_broadcast(item.get_sub_group(), v446_data, 7))));
                float v452_data = r1[7];
                float v455_data = r2[7];
                r2[7] = (v455_data + (v409_data * (sycl::group_broadcast(item.get_sub_group(), v452_data, 7))));
              }
              // s0 = store{r>s}(localShrMem0, r2);
              if (v13_lead < 8) {
                #pragma unroll
                for (int32_t v461_i1 = 0; v461_i1 < 8; ++v461_i1) {
                  float v463_data = r2[v461_i1];
                  int32_t v470_a = v13_lead + (v461_i1 * 8);
                  s0[(v470_a ^ ((v470_a >> 5) & 31))] = v463_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // glb_m2 = abs(s0)
              if (v13_lead < 8) {
                #pragma unroll
                for (int32_t v478_k1 = 0; v478_k1 < 8; ++v478_k1) {
                  int32_t v484_a = v478_k1 * 8;
                  int32_t v485_a = v13_lead + v484_a;
                  float v489_data = s0[(v485_a ^ ((v485_a >> 5) & 31))];
                  glb_m2[(v13_lead + v484_a)] = (sycl::fabs(v489_data));
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

