// === base name ===
kernel_769af9a4745aad6f

// === header ===
void launcher_kernel_769af9a4745aad6f(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_769af9a4745aad6f(const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_769af9a4745aad6f(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  m3,  m3_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_769af9a4745aad6f(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, const float* m0, size_t m0_extraOffset, const float* m1, size_t m1_extraOffset, float* m2, size_t m2_extraOffset, const float* m3, size_t m3_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
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
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v2_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v2_batchId0 < numElements0; v2_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v3_ahead1 = v2_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v5_batchId1 = (v3_ahead1 < numElements0) ? v3_ahead1 : v2_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v2_batchId0]);
            if (allowed) {
              const float *const __restrict__ glb_m0 = &m0[v2_batchId0 * 72 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v2_batchId0 * 36 + 0 + m1_extraOffset];
              float *const __restrict__ glb_m2 = &m2[v2_batchId0 * 72 + 0 + m2_extraOffset];
              const float *const __restrict__ glb_m3 = &m3[v2_batchId0 * 144 + 0 + m3_extraOffset];
              float r0[6]{};
              // r0 = load{g>r}(glb_m0);
              int32_t v17_lead = item.get_local_id(0) % 16;
              if (v17_lead < 12) {
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 6; ++v19_i1) {
                  float v27_data = glb_m0[(v17_lead + (v19_i1 * 12))];
                  r0[v19_i1] = v27_data;
                }
              }
              float r1[6]{};
              // r1 = load{g>r}(glb_m1);
              if (v17_lead < 6) {
                #pragma unroll
                for (int32_t v34_i1 = 0; v34_i1 < 6; ++v34_i1) {
                  float v42_data = glb_m1[(v17_lead + (v34_i1 * 6))];
                  r1[v34_i1] = v42_data;
                }
              }
              // wait(r0 = load{g>r}(glb_m0););
              float r3[12]{};
              // r3 = load{g>r}(glb_m3);
              if (v17_lead < 12) {
                #pragma unroll
                for (int32_t v49_i1 = 0; v49_i1 < 12; ++v49_i1) {
                  float v57_data = glb_m3[(v17_lead + (v49_i1 * 12))];
                  r3[v49_i1] = v57_data;
                }
              }
              // wait(r1 = load{g>r}(glb_m1););
              float r2[6]{};
              // r2 = +(r0 * r1) + None
              // [(0, 12), (0, 6)] [(0, 6)]
              if (v17_lead < 12) {
                float v64_data = r0[0];
                float v65_data = r1[0];
                float v68_data = r2[0];
                r2[0] = (v68_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v65_data, 0))));
                float v71_data = r1[1];
                float v74_data = r2[1];
                r2[1] = (v74_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v71_data, 0))));
                float v77_data = r1[2];
                float v80_data = r2[2];
                r2[2] = (v80_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v77_data, 0))));
                float v83_data = r1[3];
                float v86_data = r2[3];
                r2[3] = (v86_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v83_data, 0))));
                float v89_data = r1[4];
                float v92_data = r2[4];
                r2[4] = (v92_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v89_data, 0))));
                float v95_data = r1[5];
                float v98_data = r2[5];
                r2[5] = (v98_data + (v64_data * (sycl::group_broadcast(item.get_sub_group(), v95_data, 0))));
              }
              if (v17_lead < 12) {
                float v104_data = r0[1];
                float v105_data = r1[0];
                float v108_data = r2[0];
                r2[0] = (v108_data + (v104_data * (sycl::group_broadcast(item.get_sub_group(), v105_data, 1))));
                float v111_data = r1[1];
                float v114_data = r2[1];
                r2[1] = (v114_data + (v104_data * (sycl::group_broadcast(item.get_sub_group(), v111_data, 1))));
                float v117_data = r1[2];
                float v120_data = r2[2];
                r2[2] = (v120_data + (v104_data * (sycl::group_broadcast(item.get_sub_group(), v117_data, 1))));
                float v123_data = r1[3];
                float v126_data = r2[3];
                r2[3] = (v126_data + (v104_data * (sycl::group_broadcast(item.get_sub_group(), v123_data, 1))));
                float v129_data = r1[4];
                float v132_data = r2[4];
                r2[4] = (v132_data + (v104_data * (sycl::group_broadcast(item.get_sub_group(), v129_data, 1))));
                float v135_data = r1[5];
                float v138_data = r2[5];
                r2[5] = (v138_data + (v104_data * (sycl::group_broadcast(item.get_sub_group(), v135_data, 1))));
              }
              if (v17_lead < 12) {
                float v144_data = r0[2];
                float v145_data = r1[0];
                float v148_data = r2[0];
                r2[0] = (v148_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v145_data, 2))));
                float v151_data = r1[1];
                float v154_data = r2[1];
                r2[1] = (v154_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v151_data, 2))));
                float v157_data = r1[2];
                float v160_data = r2[2];
                r2[2] = (v160_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v157_data, 2))));
                float v163_data = r1[3];
                float v166_data = r2[3];
                r2[3] = (v166_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v163_data, 2))));
                float v169_data = r1[4];
                float v172_data = r2[4];
                r2[4] = (v172_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v169_data, 2))));
                float v175_data = r1[5];
                float v178_data = r2[5];
                r2[5] = (v178_data + (v144_data * (sycl::group_broadcast(item.get_sub_group(), v175_data, 2))));
              }
              if (v17_lead < 12) {
                float v184_data = r0[3];
                float v185_data = r1[0];
                float v188_data = r2[0];
                r2[0] = (v188_data + (v184_data * (sycl::group_broadcast(item.get_sub_group(), v185_data, 3))));
                float v191_data = r1[1];
                float v194_data = r2[1];
                r2[1] = (v194_data + (v184_data * (sycl::group_broadcast(item.get_sub_group(), v191_data, 3))));
                float v197_data = r1[2];
                float v200_data = r2[2];
                r2[2] = (v200_data + (v184_data * (sycl::group_broadcast(item.get_sub_group(), v197_data, 3))));
                float v203_data = r1[3];
                float v206_data = r2[3];
                r2[3] = (v206_data + (v184_data * (sycl::group_broadcast(item.get_sub_group(), v203_data, 3))));
                float v209_data = r1[4];
                float v212_data = r2[4];
                r2[4] = (v212_data + (v184_data * (sycl::group_broadcast(item.get_sub_group(), v209_data, 3))));
                float v215_data = r1[5];
                float v218_data = r2[5];
                r2[5] = (v218_data + (v184_data * (sycl::group_broadcast(item.get_sub_group(), v215_data, 3))));
              }
              if (v17_lead < 12) {
                float v224_data = r0[4];
                float v225_data = r1[0];
                float v228_data = r2[0];
                r2[0] = (v228_data + (v224_data * (sycl::group_broadcast(item.get_sub_group(), v225_data, 4))));
                float v231_data = r1[1];
                float v234_data = r2[1];
                r2[1] = (v234_data + (v224_data * (sycl::group_broadcast(item.get_sub_group(), v231_data, 4))));
                float v237_data = r1[2];
                float v240_data = r2[2];
                r2[2] = (v240_data + (v224_data * (sycl::group_broadcast(item.get_sub_group(), v237_data, 4))));
                float v243_data = r1[3];
                float v246_data = r2[3];
                r2[3] = (v246_data + (v224_data * (sycl::group_broadcast(item.get_sub_group(), v243_data, 4))));
                float v249_data = r1[4];
                float v252_data = r2[4];
                r2[4] = (v252_data + (v224_data * (sycl::group_broadcast(item.get_sub_group(), v249_data, 4))));
                float v255_data = r1[5];
                float v258_data = r2[5];
                r2[5] = (v258_data + (v224_data * (sycl::group_broadcast(item.get_sub_group(), v255_data, 4))));
              }
              if (v17_lead < 12) {
                float v264_data = r0[5];
                float v265_data = r1[0];
                float v268_data = r2[0];
                r2[0] = (v268_data + (v264_data * (sycl::group_broadcast(item.get_sub_group(), v265_data, 5))));
                float v271_data = r1[1];
                float v274_data = r2[1];
                r2[1] = (v274_data + (v264_data * (sycl::group_broadcast(item.get_sub_group(), v271_data, 5))));
                float v277_data = r1[2];
                float v280_data = r2[2];
                r2[2] = (v280_data + (v264_data * (sycl::group_broadcast(item.get_sub_group(), v277_data, 5))));
                float v283_data = r1[3];
                float v286_data = r2[3];
                r2[3] = (v286_data + (v264_data * (sycl::group_broadcast(item.get_sub_group(), v283_data, 5))));
                float v289_data = r1[4];
                float v292_data = r2[4];
                r2[4] = (v292_data + (v264_data * (sycl::group_broadcast(item.get_sub_group(), v289_data, 5))));
                float v295_data = r1[5];
                float v298_data = r2[5];
                r2[5] = (v298_data + (v264_data * (sycl::group_broadcast(item.get_sub_group(), v295_data, 5))));
              }
              // wait(r3 = load{g>r}(glb_m3););
              float r4[6]{};
              // r4 = +(r3 * r2) + None
              // [(0, 12), (0, 6)] [(0, 12)]
              float ir4[6]{};
              if (v17_lead < 12) {
                float v306_data = r3[0];
                float v307_data = r2[0];
                float v310_data = ir4[0];
                ir4[0] = (v310_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v307_data, 0))));
                float v313_data = r2[1];
                float v316_data = ir4[1];
                ir4[1] = (v316_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v313_data, 0))));
                float v319_data = r2[2];
                float v322_data = ir4[2];
                ir4[2] = (v322_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v319_data, 0))));
                float v325_data = r2[3];
                float v328_data = ir4[3];
                ir4[3] = (v328_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v325_data, 0))));
                float v331_data = r2[4];
                float v334_data = ir4[4];
                ir4[4] = (v334_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v331_data, 0))));
                float v337_data = r2[5];
                float v340_data = ir4[5];
                ir4[5] = (v340_data + (v306_data * (sycl::group_broadcast(item.get_sub_group(), v337_data, 0))));
              }
              if (v17_lead < 12) {
                float v346_data = r3[1];
                float v347_data = r2[0];
                float v350_data = ir4[0];
                ir4[0] = (v350_data + (v346_data * (sycl::group_broadcast(item.get_sub_group(), v347_data, 1))));
                float v353_data = r2[1];
                float v356_data = ir4[1];
                ir4[1] = (v356_data + (v346_data * (sycl::group_broadcast(item.get_sub_group(), v353_data, 1))));
                float v359_data = r2[2];
                float v362_data = ir4[2];
                ir4[2] = (v362_data + (v346_data * (sycl::group_broadcast(item.get_sub_group(), v359_data, 1))));
                float v365_data = r2[3];
                float v368_data = ir4[3];
                ir4[3] = (v368_data + (v346_data * (sycl::group_broadcast(item.get_sub_group(), v365_data, 1))));
                float v371_data = r2[4];
                float v374_data = ir4[4];
                ir4[4] = (v374_data + (v346_data * (sycl::group_broadcast(item.get_sub_group(), v371_data, 1))));
                float v377_data = r2[5];
                float v380_data = ir4[5];
                ir4[5] = (v380_data + (v346_data * (sycl::group_broadcast(item.get_sub_group(), v377_data, 1))));
              }
              if (v17_lead < 12) {
                float v386_data = r3[2];
                float v387_data = r2[0];
                float v390_data = ir4[0];
                ir4[0] = (v390_data + (v386_data * (sycl::group_broadcast(item.get_sub_group(), v387_data, 2))));
                float v393_data = r2[1];
                float v396_data = ir4[1];
                ir4[1] = (v396_data + (v386_data * (sycl::group_broadcast(item.get_sub_group(), v393_data, 2))));
                float v399_data = r2[2];
                float v402_data = ir4[2];
                ir4[2] = (v402_data + (v386_data * (sycl::group_broadcast(item.get_sub_group(), v399_data, 2))));
                float v405_data = r2[3];
                float v408_data = ir4[3];
                ir4[3] = (v408_data + (v386_data * (sycl::group_broadcast(item.get_sub_group(), v405_data, 2))));
                float v411_data = r2[4];
                float v414_data = ir4[4];
                ir4[4] = (v414_data + (v386_data * (sycl::group_broadcast(item.get_sub_group(), v411_data, 2))));
                float v417_data = r2[5];
                float v420_data = ir4[5];
                ir4[5] = (v420_data + (v386_data * (sycl::group_broadcast(item.get_sub_group(), v417_data, 2))));
              }
              if (v17_lead < 12) {
                float v426_data = r3[3];
                float v427_data = r2[0];
                float v430_data = ir4[0];
                ir4[0] = (v430_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v427_data, 3))));
                float v433_data = r2[1];
                float v436_data = ir4[1];
                ir4[1] = (v436_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v433_data, 3))));
                float v439_data = r2[2];
                float v442_data = ir4[2];
                ir4[2] = (v442_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v439_data, 3))));
                float v445_data = r2[3];
                float v448_data = ir4[3];
                ir4[3] = (v448_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v445_data, 3))));
                float v451_data = r2[4];
                float v454_data = ir4[4];
                ir4[4] = (v454_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v451_data, 3))));
                float v457_data = r2[5];
                float v460_data = ir4[5];
                ir4[5] = (v460_data + (v426_data * (sycl::group_broadcast(item.get_sub_group(), v457_data, 3))));
              }
              if (v17_lead < 12) {
                float v466_data = r3[4];
                float v467_data = r2[0];
                float v470_data = ir4[0];
                ir4[0] = (v470_data + (v466_data * (sycl::group_broadcast(item.get_sub_group(), v467_data, 4))));
                float v473_data = r2[1];
                float v476_data = ir4[1];
                ir4[1] = (v476_data + (v466_data * (sycl::group_broadcast(item.get_sub_group(), v473_data, 4))));
                float v479_data = r2[2];
                float v482_data = ir4[2];
                ir4[2] = (v482_data + (v466_data * (sycl::group_broadcast(item.get_sub_group(), v479_data, 4))));
                float v485_data = r2[3];
                float v488_data = ir4[3];
                ir4[3] = (v488_data + (v466_data * (sycl::group_broadcast(item.get_sub_group(), v485_data, 4))));
                float v491_data = r2[4];
                float v494_data = ir4[4];
                ir4[4] = (v494_data + (v466_data * (sycl::group_broadcast(item.get_sub_group(), v491_data, 4))));
                float v497_data = r2[5];
                float v500_data = ir4[5];
                ir4[5] = (v500_data + (v466_data * (sycl::group_broadcast(item.get_sub_group(), v497_data, 4))));
              }
              if (v17_lead < 12) {
                float v506_data = r3[5];
                float v507_data = r2[0];
                float v510_data = ir4[0];
                ir4[0] = (v510_data + (v506_data * (sycl::group_broadcast(item.get_sub_group(), v507_data, 5))));
                float v513_data = r2[1];
                float v516_data = ir4[1];
                ir4[1] = (v516_data + (v506_data * (sycl::group_broadcast(item.get_sub_group(), v513_data, 5))));
                float v519_data = r2[2];
                float v522_data = ir4[2];
                ir4[2] = (v522_data + (v506_data * (sycl::group_broadcast(item.get_sub_group(), v519_data, 5))));
                float v525_data = r2[3];
                float v528_data = ir4[3];
                ir4[3] = (v528_data + (v506_data * (sycl::group_broadcast(item.get_sub_group(), v525_data, 5))));
                float v531_data = r2[4];
                float v534_data = ir4[4];
                ir4[4] = (v534_data + (v506_data * (sycl::group_broadcast(item.get_sub_group(), v531_data, 5))));
                float v537_data = r2[5];
                float v540_data = ir4[5];
                ir4[5] = (v540_data + (v506_data * (sycl::group_broadcast(item.get_sub_group(), v537_data, 5))));
              }
              if (v17_lead < 12) {
                float v546_data = r3[6];
                float v547_data = r2[0];
                float v550_data = ir4[0];
                ir4[0] = (v550_data + (v546_data * (sycl::group_broadcast(item.get_sub_group(), v547_data, 6))));
                float v553_data = r2[1];
                float v556_data = ir4[1];
                ir4[1] = (v556_data + (v546_data * (sycl::group_broadcast(item.get_sub_group(), v553_data, 6))));
                float v559_data = r2[2];
                float v562_data = ir4[2];
                ir4[2] = (v562_data + (v546_data * (sycl::group_broadcast(item.get_sub_group(), v559_data, 6))));
                float v565_data = r2[3];
                float v568_data = ir4[3];
                ir4[3] = (v568_data + (v546_data * (sycl::group_broadcast(item.get_sub_group(), v565_data, 6))));
                float v571_data = r2[4];
                float v574_data = ir4[4];
                ir4[4] = (v574_data + (v546_data * (sycl::group_broadcast(item.get_sub_group(), v571_data, 6))));
                float v577_data = r2[5];
                float v580_data = ir4[5];
                ir4[5] = (v580_data + (v546_data * (sycl::group_broadcast(item.get_sub_group(), v577_data, 6))));
              }
              if (v17_lead < 12) {
                float v586_data = r3[7];
                float v587_data = r2[0];
                float v590_data = ir4[0];
                ir4[0] = (v590_data + (v586_data * (sycl::group_broadcast(item.get_sub_group(), v587_data, 7))));
                float v593_data = r2[1];
                float v596_data = ir4[1];
                ir4[1] = (v596_data + (v586_data * (sycl::group_broadcast(item.get_sub_group(), v593_data, 7))));
                float v599_data = r2[2];
                float v602_data = ir4[2];
                ir4[2] = (v602_data + (v586_data * (sycl::group_broadcast(item.get_sub_group(), v599_data, 7))));
                float v605_data = r2[3];
                float v608_data = ir4[3];
                ir4[3] = (v608_data + (v586_data * (sycl::group_broadcast(item.get_sub_group(), v605_data, 7))));
                float v611_data = r2[4];
                float v614_data = ir4[4];
                ir4[4] = (v614_data + (v586_data * (sycl::group_broadcast(item.get_sub_group(), v611_data, 7))));
                float v617_data = r2[5];
                float v620_data = ir4[5];
                ir4[5] = (v620_data + (v586_data * (sycl::group_broadcast(item.get_sub_group(), v617_data, 7))));
              }
              if (v17_lead < 12) {
                float v626_data = r3[8];
                float v627_data = r2[0];
                float v630_data = ir4[0];
                ir4[0] = (v630_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v627_data, 8))));
                float v633_data = r2[1];
                float v636_data = ir4[1];
                ir4[1] = (v636_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v633_data, 8))));
                float v639_data = r2[2];
                float v642_data = ir4[2];
                ir4[2] = (v642_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v639_data, 8))));
                float v645_data = r2[3];
                float v648_data = ir4[3];
                ir4[3] = (v648_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v645_data, 8))));
                float v651_data = r2[4];
                float v654_data = ir4[4];
                ir4[4] = (v654_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v651_data, 8))));
                float v657_data = r2[5];
                float v660_data = ir4[5];
                ir4[5] = (v660_data + (v626_data * (sycl::group_broadcast(item.get_sub_group(), v657_data, 8))));
              }
              if (v17_lead < 12) {
                float v666_data = r3[9];
                float v667_data = r2[0];
                float v670_data = ir4[0];
                ir4[0] = (v670_data + (v666_data * (sycl::group_broadcast(item.get_sub_group(), v667_data, 9))));
                float v673_data = r2[1];
                float v676_data = ir4[1];
                ir4[1] = (v676_data + (v666_data * (sycl::group_broadcast(item.get_sub_group(), v673_data, 9))));
                float v679_data = r2[2];
                float v682_data = ir4[2];
                ir4[2] = (v682_data + (v666_data * (sycl::group_broadcast(item.get_sub_group(), v679_data, 9))));
                float v685_data = r2[3];
                float v688_data = ir4[3];
                ir4[3] = (v688_data + (v666_data * (sycl::group_broadcast(item.get_sub_group(), v685_data, 9))));
                float v691_data = r2[4];
                float v694_data = ir4[4];
                ir4[4] = (v694_data + (v666_data * (sycl::group_broadcast(item.get_sub_group(), v691_data, 9))));
                float v697_data = r2[5];
                float v700_data = ir4[5];
                ir4[5] = (v700_data + (v666_data * (sycl::group_broadcast(item.get_sub_group(), v697_data, 9))));
              }
              if (v17_lead < 12) {
                float v706_data = r3[10];
                float v707_data = r2[0];
                float v710_data = ir4[0];
                ir4[0] = (v710_data + (v706_data * (sycl::group_broadcast(item.get_sub_group(), v707_data, 10))));
                float v713_data = r2[1];
                float v716_data = ir4[1];
                ir4[1] = (v716_data + (v706_data * (sycl::group_broadcast(item.get_sub_group(), v713_data, 10))));
                float v719_data = r2[2];
                float v722_data = ir4[2];
                ir4[2] = (v722_data + (v706_data * (sycl::group_broadcast(item.get_sub_group(), v719_data, 10))));
                float v725_data = r2[3];
                float v728_data = ir4[3];
                ir4[3] = (v728_data + (v706_data * (sycl::group_broadcast(item.get_sub_group(), v725_data, 10))));
                float v731_data = r2[4];
                float v734_data = ir4[4];
                ir4[4] = (v734_data + (v706_data * (sycl::group_broadcast(item.get_sub_group(), v731_data, 10))));
                float v737_data = r2[5];
                float v740_data = ir4[5];
                ir4[5] = (v740_data + (v706_data * (sycl::group_broadcast(item.get_sub_group(), v737_data, 10))));
              }
              if (v17_lead < 12) {
                float v746_data = r3[11];
                float v747_data = r2[0];
                float v750_data = ir4[0];
                ir4[0] = (v750_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v747_data, 11))));
                float v753_data = r2[1];
                float v756_data = ir4[1];
                ir4[1] = (v756_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v753_data, 11))));
                float v759_data = r2[2];
                float v762_data = ir4[2];
                ir4[2] = (v762_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v759_data, 11))));
                float v765_data = r2[3];
                float v768_data = ir4[3];
                ir4[3] = (v768_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v765_data, 11))));
                float v771_data = r2[4];
                float v774_data = ir4[4];
                ir4[4] = (v774_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v771_data, 11))));
                float v777_data = r2[5];
                float v780_data = ir4[5];
                ir4[5] = (v780_data + (v746_data * (sycl::group_broadcast(item.get_sub_group(), v777_data, 11))));
              }
              if (v17_lead < 12) {
                #pragma unroll
                for (int32_t v786_n1 = 0; v786_n1 < 6; ++v786_n1) {
                  float v788_data = ir4[v786_n1];
                  r4[v786_n1] = v788_data;
                }
              }
              // glb_m2 = store{r>g}(r4);
              if (v17_lead < 12) {
                #pragma unroll
                for (int32_t v794_i1 = 0; v794_i1 < 6; ++v794_i1) {
                  float v796_data = r4[v794_i1];
                  glb_m2[(v17_lead + (v794_i1 * 12))] = v796_data;
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

