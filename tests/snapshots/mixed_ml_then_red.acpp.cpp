// === base name ===
kernel_55f5edc917f5f6fe

// === header ===
void launcher_kernel_55f5edc917f5f6fe(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_55f5edc917f5f6fe(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_55f5edc917f5f6fe(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_55f5edc917f5f6fe(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (1280, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 8×8(8×8) {0..8}×{0..8} strided
        // m1 8×8(8×8) {0..8}×{0..8} strided
        // m2 8(8) {0..8} strided
        // t0 8×8(8×8) {0..8}×{0..8} pointer_based({0..8}×{0..8})[0, 1] = m0 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[0, -1]×m1 8×8(8×8) {0..8}×{0..8} strided({0..8}×{0..8})[-1, 1]
        // OUT = +(TMP, dims=[1])
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[80 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[64];
          float * __restrict__ s0 = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 64 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 64 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 8 + 0 + m2_extraOffset];
              float r0[8]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v17_lead = item.get_local_id(0) % 16;
              if (v17_lead < 8) {
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 8; ++v19_i1) {
                  float v27_data = glb_m0[(v17_lead + (v19_i1 * 8))];
                  r0[v19_i1] = v27_data;
                }
              }
              float r1[8]{};
              // r1 = load{g>r}(glb_m1);
              if (v17_lead < 8) {
                #pragma unroll
                for (int32_t v34_i1 = 0; v34_i1 < 8; ++v34_i1) {
                  float v42_data = glb_m1[(v17_lead + (v34_i1 * 8))];
                  r1[v34_i1] = v42_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              // wait(r1 = load{g>r}(glb_m1););
              float r2[8]{};
              // r2 = +(r0 * r1) + None
              // [(0, 8), (0, 8)] [(0, 8)]
              if (v17_lead < 8) {
                float v49_data = r0[0];
                float v50_data = r1[0];
                float v53_data = r2[0];
                r2[0] = (v53_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v50_data, 0))));
                float v56_data = r1[1];
                float v59_data = r2[1];
                r2[1] = (v59_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v56_data, 0))));
                float v62_data = r1[2];
                float v65_data = r2[2];
                r2[2] = (v65_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v62_data, 0))));
                float v68_data = r1[3];
                float v71_data = r2[3];
                r2[3] = (v71_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v68_data, 0))));
                float v74_data = r1[4];
                float v77_data = r2[4];
                r2[4] = (v77_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v74_data, 0))));
                float v80_data = r1[5];
                float v83_data = r2[5];
                r2[5] = (v83_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v80_data, 0))));
                float v86_data = r1[6];
                float v89_data = r2[6];
                r2[6] = (v89_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v86_data, 0))));
                float v92_data = r1[7];
                float v95_data = r2[7];
                r2[7] = (v95_data + (v49_data * (sycl::group_broadcast(item.get_sub_group(), v92_data, 0))));
              }
              if (v17_lead < 8) {
                float v101_data = r0[1];
                float v102_data = r1[0];
                float v105_data = r2[0];
                r2[0] = (v105_data + (v101_data * (sycl::group_broadcast(item.get_sub_group(), v102_data, 1))));
                float v108_data = r1[1];
                float v111_data = r2[1];
                r2[1] = (v111_data + (v101_data * (sycl::group_broadcast(item.get_sub_group(), v108_data, 1))));
                float v114_data = r1[2];
                float v117_data = r2[2];
                r2[2] = (v117_data + (v101_data * (sycl::group_broadcast(item.get_sub_group(), v114_data, 1))));
                float v120_data = r1[3];
                float v123_data = r2[3];
                r2[3] = (v123_data + (v101_data * (sycl::group_broadcast(item.get_sub_group(), v120_data, 1))));
                float v126_data = r1[4];
                float v129_data = r2[4];
                r2[4] = (v129_data + (v101_data * (sycl::group_broadcast(item.get_sub_group(), v126_data, 1))));
                float v132_data = r1[5];
                float v135_data = r2[5];
                r2[5] = (v135_data + (v101_data * (sycl::group_broadcast(item.get_sub_group(), v132_data, 1))));
                float v138_data = r1[6];
                float v141_data = r2[6];
                r2[6] = (v141_data + (v101_data * (sycl::group_broadcast(item.get_sub_group(), v138_data, 1))));
                float v144_data = r1[7];
                float v147_data = r2[7];
                r2[7] = (v147_data + (v101_data * (sycl::group_broadcast(item.get_sub_group(), v144_data, 1))));
              }
              if (v17_lead < 8) {
                float v153_data = r0[2];
                float v154_data = r1[0];
                float v157_data = r2[0];
                r2[0] = (v157_data + (v153_data * (sycl::group_broadcast(item.get_sub_group(), v154_data, 2))));
                float v160_data = r1[1];
                float v163_data = r2[1];
                r2[1] = (v163_data + (v153_data * (sycl::group_broadcast(item.get_sub_group(), v160_data, 2))));
                float v166_data = r1[2];
                float v169_data = r2[2];
                r2[2] = (v169_data + (v153_data * (sycl::group_broadcast(item.get_sub_group(), v166_data, 2))));
                float v172_data = r1[3];
                float v175_data = r2[3];
                r2[3] = (v175_data + (v153_data * (sycl::group_broadcast(item.get_sub_group(), v172_data, 2))));
                float v178_data = r1[4];
                float v181_data = r2[4];
                r2[4] = (v181_data + (v153_data * (sycl::group_broadcast(item.get_sub_group(), v178_data, 2))));
                float v184_data = r1[5];
                float v187_data = r2[5];
                r2[5] = (v187_data + (v153_data * (sycl::group_broadcast(item.get_sub_group(), v184_data, 2))));
                float v190_data = r1[6];
                float v193_data = r2[6];
                r2[6] = (v193_data + (v153_data * (sycl::group_broadcast(item.get_sub_group(), v190_data, 2))));
                float v196_data = r1[7];
                float v199_data = r2[7];
                r2[7] = (v199_data + (v153_data * (sycl::group_broadcast(item.get_sub_group(), v196_data, 2))));
              }
              if (v17_lead < 8) {
                float v205_data = r0[3];
                float v206_data = r1[0];
                float v209_data = r2[0];
                r2[0] = (v209_data + (v205_data * (sycl::group_broadcast(item.get_sub_group(), v206_data, 3))));
                float v212_data = r1[1];
                float v215_data = r2[1];
                r2[1] = (v215_data + (v205_data * (sycl::group_broadcast(item.get_sub_group(), v212_data, 3))));
                float v218_data = r1[2];
                float v221_data = r2[2];
                r2[2] = (v221_data + (v205_data * (sycl::group_broadcast(item.get_sub_group(), v218_data, 3))));
                float v224_data = r1[3];
                float v227_data = r2[3];
                r2[3] = (v227_data + (v205_data * (sycl::group_broadcast(item.get_sub_group(), v224_data, 3))));
                float v230_data = r1[4];
                float v233_data = r2[4];
                r2[4] = (v233_data + (v205_data * (sycl::group_broadcast(item.get_sub_group(), v230_data, 3))));
                float v236_data = r1[5];
                float v239_data = r2[5];
                r2[5] = (v239_data + (v205_data * (sycl::group_broadcast(item.get_sub_group(), v236_data, 3))));
                float v242_data = r1[6];
                float v245_data = r2[6];
                r2[6] = (v245_data + (v205_data * (sycl::group_broadcast(item.get_sub_group(), v242_data, 3))));
                float v248_data = r1[7];
                float v251_data = r2[7];
                r2[7] = (v251_data + (v205_data * (sycl::group_broadcast(item.get_sub_group(), v248_data, 3))));
              }
              if (v17_lead < 8) {
                float v257_data = r0[4];
                float v258_data = r1[0];
                float v261_data = r2[0];
                r2[0] = (v261_data + (v257_data * (sycl::group_broadcast(item.get_sub_group(), v258_data, 4))));
                float v264_data = r1[1];
                float v267_data = r2[1];
                r2[1] = (v267_data + (v257_data * (sycl::group_broadcast(item.get_sub_group(), v264_data, 4))));
                float v270_data = r1[2];
                float v273_data = r2[2];
                r2[2] = (v273_data + (v257_data * (sycl::group_broadcast(item.get_sub_group(), v270_data, 4))));
                float v276_data = r1[3];
                float v279_data = r2[3];
                r2[3] = (v279_data + (v257_data * (sycl::group_broadcast(item.get_sub_group(), v276_data, 4))));
                float v282_data = r1[4];
                float v285_data = r2[4];
                r2[4] = (v285_data + (v257_data * (sycl::group_broadcast(item.get_sub_group(), v282_data, 4))));
                float v288_data = r1[5];
                float v291_data = r2[5];
                r2[5] = (v291_data + (v257_data * (sycl::group_broadcast(item.get_sub_group(), v288_data, 4))));
                float v294_data = r1[6];
                float v297_data = r2[6];
                r2[6] = (v297_data + (v257_data * (sycl::group_broadcast(item.get_sub_group(), v294_data, 4))));
                float v300_data = r1[7];
                float v303_data = r2[7];
                r2[7] = (v303_data + (v257_data * (sycl::group_broadcast(item.get_sub_group(), v300_data, 4))));
              }
              if (v17_lead < 8) {
                float v309_data = r0[5];
                float v310_data = r1[0];
                float v313_data = r2[0];
                r2[0] = (v313_data + (v309_data * (sycl::group_broadcast(item.get_sub_group(), v310_data, 5))));
                float v316_data = r1[1];
                float v319_data = r2[1];
                r2[1] = (v319_data + (v309_data * (sycl::group_broadcast(item.get_sub_group(), v316_data, 5))));
                float v322_data = r1[2];
                float v325_data = r2[2];
                r2[2] = (v325_data + (v309_data * (sycl::group_broadcast(item.get_sub_group(), v322_data, 5))));
                float v328_data = r1[3];
                float v331_data = r2[3];
                r2[3] = (v331_data + (v309_data * (sycl::group_broadcast(item.get_sub_group(), v328_data, 5))));
                float v334_data = r1[4];
                float v337_data = r2[4];
                r2[4] = (v337_data + (v309_data * (sycl::group_broadcast(item.get_sub_group(), v334_data, 5))));
                float v340_data = r1[5];
                float v343_data = r2[5];
                r2[5] = (v343_data + (v309_data * (sycl::group_broadcast(item.get_sub_group(), v340_data, 5))));
                float v346_data = r1[6];
                float v349_data = r2[6];
                r2[6] = (v349_data + (v309_data * (sycl::group_broadcast(item.get_sub_group(), v346_data, 5))));
                float v352_data = r1[7];
                float v355_data = r2[7];
                r2[7] = (v355_data + (v309_data * (sycl::group_broadcast(item.get_sub_group(), v352_data, 5))));
              }
              if (v17_lead < 8) {
                float v361_data = r0[6];
                float v362_data = r1[0];
                float v365_data = r2[0];
                r2[0] = (v365_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v362_data, 6))));
                float v368_data = r1[1];
                float v371_data = r2[1];
                r2[1] = (v371_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v368_data, 6))));
                float v374_data = r1[2];
                float v377_data = r2[2];
                r2[2] = (v377_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v374_data, 6))));
                float v380_data = r1[3];
                float v383_data = r2[3];
                r2[3] = (v383_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v380_data, 6))));
                float v386_data = r1[4];
                float v389_data = r2[4];
                r2[4] = (v389_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v386_data, 6))));
                float v392_data = r1[5];
                float v395_data = r2[5];
                r2[5] = (v395_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v392_data, 6))));
                float v398_data = r1[6];
                float v401_data = r2[6];
                r2[6] = (v401_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v398_data, 6))));
                float v404_data = r1[7];
                float v407_data = r2[7];
                r2[7] = (v407_data + (v361_data * (sycl::group_broadcast(item.get_sub_group(), v404_data, 6))));
              }
              if (v17_lead < 8) {
                float v413_data = r0[7];
                float v414_data = r1[0];
                float v417_data = r2[0];
                r2[0] = (v417_data + (v413_data * (sycl::group_broadcast(item.get_sub_group(), v414_data, 7))));
                float v420_data = r1[1];
                float v423_data = r2[1];
                r2[1] = (v423_data + (v413_data * (sycl::group_broadcast(item.get_sub_group(), v420_data, 7))));
                float v426_data = r1[2];
                float v429_data = r2[2];
                r2[2] = (v429_data + (v413_data * (sycl::group_broadcast(item.get_sub_group(), v426_data, 7))));
                float v432_data = r1[3];
                float v435_data = r2[3];
                r2[3] = (v435_data + (v413_data * (sycl::group_broadcast(item.get_sub_group(), v432_data, 7))));
                float v438_data = r1[4];
                float v441_data = r2[4];
                r2[4] = (v441_data + (v413_data * (sycl::group_broadcast(item.get_sub_group(), v438_data, 7))));
                float v444_data = r1[5];
                float v447_data = r2[5];
                r2[5] = (v447_data + (v413_data * (sycl::group_broadcast(item.get_sub_group(), v444_data, 7))));
                float v450_data = r1[6];
                float v453_data = r2[6];
                r2[6] = (v453_data + (v413_data * (sycl::group_broadcast(item.get_sub_group(), v450_data, 7))));
                float v456_data = r1[7];
                float v459_data = r2[7];
                r2[7] = (v459_data + (v413_data * (sycl::group_broadcast(item.get_sub_group(), v456_data, 7))));
              }
              // s0 = store{r>s}(localShrMem0, r2);
              if (v17_lead < 8) {
                #pragma unroll
                for (int32_t v465_i1 = 0; v465_i1 < 8; ++v465_i1) {
                  float v467_data = r2[v465_i1];
                  int32_t v474_a = v17_lead + (v465_i1 * 8);
                  s0[(v474_a ^ ((v474_a >> 5) & 31))] = v467_data;
                }
              }
              sycl::group_barrier(item.get_sub_group());
              // glb_m2 = +(s0, dims=[1])
              if (v17_lead < 8) {
                float v483_acc0 = 0.0f;
                #pragma unroll
                for (int32_t v482_r1 = 0; v482_r1 < 8; ++v482_r1) {
                  int32_t v490_a = v17_lead + (v482_r1 * 8);
                  float v494_data = s0[(v490_a ^ ((v490_a >> 5) & 31))];
                  v483_acc0 = (v483_acc0 + v494_data);
                }
                glb_m2[v17_lead] = v483_acc0;
              }
              sycl::group_barrier(item.get_sub_group());
            }
          }
        }
      });
    }
  });
}

