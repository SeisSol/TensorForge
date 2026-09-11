// === base name ===
kernel_66c79787b3721682

// === header ===
void launcher_kernel_66c79787b3721682(double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
void launcher_kernel_66c79787b3721682(double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 , void* streamPtr) {
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
  kernel_kernel_66c79787b3721682(stream, grid, block,  m0,  m0_extraOffset,  m1,  m1_extraOffset,  m2,  m2_extraOffset,  numElements0,  flags0 );
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_66c79787b3721682(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, double* m0, size_t m0_extraOffset, const double* m1, size_t m1_extraOffset, const double* m2, size_t m2_extraOffset, size_t numElements0, unsigned* flags0 ) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<double, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_size.get(0), group_size.get(1), group_count.get(0) * group_size.get(2)}, group_size}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // meta data:
        // m0 16×16(16×16) {0..16}×{0..16} strided
        // m1 16×16(16×16) {0..16}×{0..16} strided
        // m2 16×16(16×16) {0..16}×{0..16} strided
        // m0 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, 1] = m1 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[0, -1]×m2 16×16(16×16) {0..16}×{0..16} strided({0..16}×{0..16})[-1, 1]
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_global_range(0) * item.get_group().get_local_range(1)) : batchId1;
          double* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          double* tempShrMem = &localShrMem0[0];
          for (size_t v2_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(0))); v2_batchId0 < numElements0; v2_batchId0 += (item.get_global_range(0) * item.get_group().get_local_range(1))) {
            size_t v3_ahead1 = v2_batchId0 + (item.get_global_range(0) * item.get_group().get_local_range(1));
            size_t v5_batchId1 = (v3_ahead1 < numElements0) ? v3_ahead1 : v2_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v2_batchId0]);
            if (allowed) {
              double *const __restrict__ glb_m0 = &m0[v2_batchId0 * 256 + 0 + m0_extraOffset];
              const double *const __restrict__ glb_m1 = &m1[v2_batchId0 * 256 + 0 + m1_extraOffset];
              const double *const __restrict__ glb_m2 = &m2[v2_batchId0 * 46 + 0 + m2_extraOffset];
              double r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v16_lead = item.get_local_id(0) % 16;
              #pragma unroll
              for (int32_t v17_i0 = 0; v17_i0 < 1; ++v17_i0) {
                int32_t v23_lead = v16_lead + (v17_i0 * 16);
                #pragma unroll
                for (int32_t v18_i1 = 0; v18_i1 < 16; ++v18_i1) {
                  double v26_data = glb_m1[(v23_lead + (v18_i1 * 16))];
                  r0[(v17_i0 + v18_i1)] = v26_data;
                }
              }
              double r1[16]{};
              // r1 = load{g>r}(glb_m2);
              double v29_lin = glb_m2[0 + item.get_local_id(0) * 1];
              r1[0] = v29_lin;
              double v30_lin = glb_m2[16 + item.get_local_id(0) * 1];
              r1[1] = v30_lin;
              double v31_lin = glb_m2[32 + item.get_local_id(0) * 1];
              r1[2] = v31_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              double r2[16]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              double ir2[16]{};
              double v37_data = r0[0];
              double v38_data = r1[0];
              double v40_data = ir2[0];
              ir2[0] = (v40_data + (v37_data * v38_data));
              double v43_data = r1[2];
              double v45_data = ir2[1];
              ir2[1] = (v45_data + (v37_data * v43_data));
              double v64_data = r0[1];
              double v65_data = r1[1];
              double v67_data = ir2[0];
              ir2[0] = (v67_data + (v64_data * v65_data));
              double v70_data = r1[3];
              double v72_data = ir2[1];
              ir2[1] = (v72_data + (v64_data * v70_data));
              double v75_data = r1[5];
              double v77_data = ir2[2];
              ir2[2] = (v77_data + (v64_data * v75_data));
              double v95_data = r0[2];
              double v97_data = r1[4];
              double v99_data = ir2[1];
              ir2[1] = (v99_data + (v95_data * v97_data));
              double v102_data = r1[6];
              double v104_data = ir2[2];
              ir2[2] = (v104_data + (v95_data * v102_data));
              double v107_data = r1[8];
              double v109_data = ir2[3];
              ir2[3] = (v109_data + (v95_data * v107_data));
              double v126_data = r0[3];
              double v129_data = r1[7];
              double v131_data = ir2[2];
              ir2[2] = (v131_data + (v126_data * v129_data));
              double v134_data = r1[9];
              double v136_data = ir2[3];
              ir2[3] = (v136_data + (v126_data * v134_data));
              double v139_data = r1[11];
              double v141_data = ir2[4];
              ir2[4] = (v141_data + (v126_data * v139_data));
              double v157_data = r0[4];
              double v161_data = r1[10];
              double v163_data = ir2[3];
              ir2[3] = (v163_data + (v157_data * v161_data));
              double v166_data = r1[12];
              double v168_data = ir2[4];
              ir2[4] = (v168_data + (v157_data * v166_data));
              double v171_data = r1[14];
              double v173_data = ir2[5];
              ir2[5] = (v173_data + (v157_data * v171_data));
              double v188_data = r0[5];
              double v193_data = r1[13];
              double v195_data = ir2[4];
              ir2[4] = (v195_data + (v188_data * v193_data));
              double v198_data = r1[15];
              double v200_data = ir2[5];
              ir2[5] = (v200_data + (v188_data * v198_data));
              double v203_data = r1[17];
              double v205_data = ir2[6];
              ir2[6] = (v205_data + (v188_data * v203_data));
              double v219_data = r0[6];
              double v225_data = r1[16];
              double v227_data = ir2[5];
              ir2[5] = (v227_data + (v219_data * v225_data));
              double v230_data = r1[18];
              double v232_data = ir2[6];
              ir2[6] = (v232_data + (v219_data * v230_data));
              double v235_data = r1[20];
              double v237_data = ir2[7];
              ir2[7] = (v237_data + (v219_data * v235_data));
              double v250_data = r0[7];
              double v257_data = r1[19];
              double v259_data = ir2[6];
              ir2[6] = (v259_data + (v250_data * v257_data));
              double v262_data = r1[21];
              double v264_data = ir2[7];
              ir2[7] = (v264_data + (v250_data * v262_data));
              double v267_data = r1[23];
              double v269_data = ir2[8];
              ir2[8] = (v269_data + (v250_data * v267_data));
              double v281_data = r0[8];
              double v289_data = r1[22];
              double v291_data = ir2[7];
              ir2[7] = (v291_data + (v281_data * v289_data));
              double v294_data = r1[24];
              double v296_data = ir2[8];
              ir2[8] = (v296_data + (v281_data * v294_data));
              double v299_data = r1[26];
              double v301_data = ir2[9];
              ir2[9] = (v301_data + (v281_data * v299_data));
              double v312_data = r0[9];
              double v321_data = r1[25];
              double v323_data = ir2[8];
              ir2[8] = (v323_data + (v312_data * v321_data));
              double v326_data = r1[27];
              double v328_data = ir2[9];
              ir2[9] = (v328_data + (v312_data * v326_data));
              double v331_data = r1[29];
              double v333_data = ir2[10];
              ir2[10] = (v333_data + (v312_data * v331_data));
              double v343_data = r0[10];
              double v353_data = r1[28];
              double v355_data = ir2[9];
              ir2[9] = (v355_data + (v343_data * v353_data));
              double v358_data = r1[30];
              double v360_data = ir2[10];
              ir2[10] = (v360_data + (v343_data * v358_data));
              double v363_data = r1[32];
              double v365_data = ir2[11];
              ir2[11] = (v365_data + (v343_data * v363_data));
              double v374_data = r0[11];
              double v385_data = r1[31];
              double v387_data = ir2[10];
              ir2[10] = (v387_data + (v374_data * v385_data));
              double v390_data = r1[33];
              double v392_data = ir2[11];
              ir2[11] = (v392_data + (v374_data * v390_data));
              double v395_data = r1[35];
              double v397_data = ir2[12];
              ir2[12] = (v397_data + (v374_data * v395_data));
              double v405_data = r0[12];
              double v417_data = r1[34];
              double v419_data = ir2[11];
              ir2[11] = (v419_data + (v405_data * v417_data));
              double v422_data = r1[36];
              double v424_data = ir2[12];
              ir2[12] = (v424_data + (v405_data * v422_data));
              double v427_data = r1[38];
              double v429_data = ir2[13];
              ir2[13] = (v429_data + (v405_data * v427_data));
              double v436_data = r0[13];
              double v449_data = r1[37];
              double v451_data = ir2[12];
              ir2[12] = (v451_data + (v436_data * v449_data));
              double v454_data = r1[39];
              double v456_data = ir2[13];
              ir2[13] = (v456_data + (v436_data * v454_data));
              double v459_data = r1[41];
              double v461_data = ir2[14];
              ir2[14] = (v461_data + (v436_data * v459_data));
              double v467_data = r0[14];
              double v481_data = r1[40];
              double v483_data = ir2[13];
              ir2[13] = (v483_data + (v467_data * v481_data));
              double v486_data = r1[42];
              double v488_data = ir2[14];
              ir2[14] = (v488_data + (v467_data * v486_data));
              double v491_data = r1[44];
              double v493_data = ir2[15];
              ir2[15] = (v493_data + (v467_data * v491_data));
              double v498_data = r0[15];
              double v513_data = r1[43];
              double v515_data = ir2[14];
              ir2[14] = (v515_data + (v498_data * v513_data));
              double v518_data = r1[45];
              double v520_data = ir2[15];
              ir2[15] = (v520_data + (v498_data * v518_data));
              #pragma unroll
              for (int32_t v525_n0 = 0; v525_n0 < 1; ++v525_n0) {
                #pragma unroll
                for (int32_t v526_n1 = 0; v526_n1 < 16; ++v526_n1) {
                  int32_t v527_a = v525_n0 + v526_n1;
                  double v528_data = ir2[v527_a];
                  r2[v527_a] = v528_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v533_i0 = 0; v533_i0 < 1; ++v533_i0) {
                int32_t v541_lead = v16_lead + (v533_i0 * 16);
                #pragma unroll
                for (int32_t v534_i1 = 0; v534_i1 < 16; ++v534_i1) {
                  double v536_data = r2[(v533_i0 + v534_i1)];
                  glb_m0[(v541_lead + (v534_i1 * 16))] = v536_data;
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

