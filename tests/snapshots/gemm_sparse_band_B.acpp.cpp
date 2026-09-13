// === base name ===
kernel_4a5b754e461dff15

// === header ===
#ifndef TENSORFORGE_LAUNCH_TYPES
#define TENSORFORGE_LAUNCH_TYPES
#include <cstddef>
namespace tensorforge {
// Fixed when the kernel is generated: `launch_info_<kernel>`.
struct LaunchInfo {
  unsigned block[3];
  unsigned threadsPerMult;
  unsigned activeThreads;
  unsigned leadWidth;
  unsigned multsPerBlock;
  std::size_t sharedMemBytes;
  bool cooperative;
  bool persistent;
  unsigned sections;
};
// What one launch uses, the grid included: `launch_config_<kernel>`.
struct LaunchConfig {
  std::size_t grid[3];
  std::size_t block[3];
  std::size_t sharedMemBytes;
  bool cooperative;
};
} // namespace tensorforge
#endif
inline constexpr tensorforge::LaunchInfo launch_info_kernel_4a5b754e461dff15 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_4a5b754e461dff15(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_4a5b754e461dff15(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


// === launcher ===
#ifndef TENSORFORGE_LAUNCH_TYPES
#define TENSORFORGE_LAUNCH_TYPES
#include <cstddef>
namespace tensorforge {
// Fixed when the kernel is generated: `launch_info_<kernel>`.
struct LaunchInfo {
  unsigned block[3];
  unsigned threadsPerMult;
  unsigned activeThreads;
  unsigned leadWidth;
  unsigned multsPerBlock;
  std::size_t sharedMemBytes;
  bool cooperative;
  bool persistent;
  unsigned sections;
};
// What one launch uses, the grid included: `launch_config_<kernel>`.
struct LaunchConfig {
  std::size_t grid[3];
  std::size_t block[3];
  std::size_t sharedMemBytes;
  bool cooperative;
};
} // namespace tensorforge
#endif
tensorforge::LaunchConfig launch_config_kernel_4a5b754e461dff15(size_t numElements0, void* streamPtr) {
  (void)numElements0;
  (void)streamPtr;
  sycl::range<3> block (16, 16, 1);
  static std::size_t gridsize = 0;
  if (gridsize == 0 && streamPtr != nullptr) {
    gridsize = static_cast<sycl::queue *>(streamPtr)->get_device().get_info<sycl::info::device::max_compute_units>();
  }
  tensorforge::LaunchConfig config{};
  config.grid[0] = std::min(gridsize, numElements0);
  config.grid[1] = 1;
  config.grid[2] = 1;
  config.block[0] = 16;
  config.block[1] = 16;
  config.block[2] = 1;
  config.sharedMemBytes = 256 * sizeof(float);
  config.cooperative = false;
  return config;
}
void launcher_kernel_4a5b754e461dff15(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_4a5b754e461dff15(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_4a5b754e461dff15(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_4a5b754e461dff15(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
  stream->submit([&](sycl::handler &cgh) {
    sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> totalShrMem (256, cgh); {
      cgh.parallel_for(sycl::nd_range<3>{{group_count.get(2) * group_size.get(2), group_count.get(1) * group_size.get(1), group_count.get(0) * group_size.get(0)}, {group_size.get(2), group_size.get(1), group_size.get(0)}}, [=](sycl::nd_item<3> item)  {
        // generated with TensorForge. Version: 0.0.1
        // options: default
        // launch: 16 lanes x 16 per block = block 16x16x1, 1024 B shared, occupancy grid
        // operands:
        //   m0 16×16(16×16) {0..16}×{0..16} strided
        //   m1 16×16(16×16) {0..16}×{0..16} strided
        //   m2 16×16(16×16) {0..16}×{0..16} strided
        // operations:
        //   m0[i,j] = m1[i,k] × m2[k,j]
        // tensorforge-meta: {"fp":"float","launch":{"active_threads":16,"block":[16,16,1],"cooperative":false,"lead_width":1,"mults_per_block":16,"persistent":true,"sections":[{"barrier":false,"mults_per_block":16,"shared_elements":256}],"shared_bytes":1024,"shared_elements":256,"threads_per_mult":16},"loops":[],"operands":[{"addressing":"strided","alias":"C","bbox":[[0,0],[16,16]],"name":"m0","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"A","bbox":[[0,0],[16,16]],"name":"m1","ordered":false,"parts":1,"shape":[16,16],"variant":false},{"addressing":"strided","alias":"B","bbox":[[0,0],[16,16]],"name":"m2","ordered":false,"parts":1,"shape":[16,16],"variant":false}],"operations":[{"add":false,"dest":{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m0","offset":[0,0],"shape":[16,16]},"kind":"multilinear","ops":[{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m1","offset":[0,0],"shape":[16,16]},{"addressing":"strided","bbox":[[0,0],[16,16]],"is_tmp":false,"name":"m2","offset":[0,0],"shape":[16,16]}],"permute":[[0,1],[0,1]],"target":[[0,-1],[-1,1]]}],"version":"0.0.1\n"}
        {
          const auto batchId_start = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2)));
          const auto batchId1 = batchId_start < numElements0 ? batchId_start : 0;
          const auto batchId2 = batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) < numElements0 ? batchId1 + (item.get_group_range(2) * item.get_group().get_local_range(1)) : batchId1;
          float* localShrMem0 = &totalShrMem[16 * item.get_local_id(1) + 0];
          float* tempShrMem = &localShrMem0[0];
          for (size_t v3_batchId0 = (item.get_local_id(1) + item.get_group().get_local_range(1) * (item.get_group().get_group_id(2))); v3_batchId0 < numElements0; v3_batchId0 += (item.get_group_range(2) * item.get_group().get_local_range(1))) {
            size_t v4_ahead1 = v3_batchId0 + (item.get_group_range(2) * item.get_group().get_local_range(1));
            size_t v6_batchId1 = (v4_ahead1 < numElements0) ? v4_ahead1 : v3_batchId0;
            const bool allowed = flags0 == nullptr ? true : static_cast<bool>(flags0[v3_batchId0]);
            if (allowed) {
              float *const __restrict__ glb_m0 = &m0[v3_batchId0 * 256 + 0 + m0_extraOffset];
              const float *const __restrict__ glb_m1 = &m1[v3_batchId0 * 256 + 0 + m1_extraOffset];
              const float *const __restrict__ glb_m2 = &m2[v3_batchId0 * 46 + 0 + m2_extraOffset];
              float r0[16]{};
              // r0 = load{g>r}(glb_m1);
              int32_t v17_lead = item.get_local_id(2) % 16;
              #pragma unroll
              for (int32_t v18_i0 = 0; v18_i0 < 1; ++v18_i0) {
                int32_t v21_lead = v17_lead + (v18_i0 * 16);
                #pragma unroll
                for (int32_t v19_i1 = 0; v19_i1 < 16; ++v19_i1) {
                  float v24_data = glb_m1[(v21_lead + (v19_i1 * 16))];
                  r0[(v18_i0 + v19_i1)] = v24_data;
                }
              }
              float r1[16]{};
              // r1 = load{g>r}(glb_m2);
              float v27_lin = glb_m2[0 + item.get_local_id(2) * 1];
              r1[0] = v27_lin;
              float v28_lin = glb_m2[16 + item.get_local_id(2) * 1];
              r1[1] = v28_lin;
              float v29_lin = glb_m2[32 + item.get_local_id(2) * 1];
              r1[2] = v29_lin;
              // wait(r0 = load{g>r}(glb_m1););
              // wait(r1 = load{g>r}(glb_m2););
              float r2[16]{};
              // r2 = +(r0 * r1) + None
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir2[16]{};
              float v32_data = r0[0];
              float v33_data = r1[0];
              float v35_data = ir2[0];
              ir2[0] = (v35_data + (v32_data * v33_data));
              float v38_data = r1[2];
              float v40_data = ir2[1];
              ir2[1] = (v40_data + (v32_data * v38_data));
              float v56_data = r0[1];
              float v57_data = r1[1];
              float v59_data = ir2[0];
              ir2[0] = (v59_data + (v56_data * v57_data));
              float v62_data = r1[3];
              float v64_data = ir2[1];
              ir2[1] = (v64_data + (v56_data * v62_data));
              float v67_data = r1[5];
              float v69_data = ir2[2];
              ir2[2] = (v69_data + (v56_data * v67_data));
              float v84_data = r0[2];
              float v86_data = r1[4];
              float v88_data = ir2[1];
              ir2[1] = (v88_data + (v84_data * v86_data));
              float v91_data = r1[6];
              float v93_data = ir2[2];
              ir2[2] = (v93_data + (v84_data * v91_data));
              float v96_data = r1[8];
              float v98_data = ir2[3];
              ir2[3] = (v98_data + (v84_data * v96_data));
              float v112_data = r0[3];
              float v115_data = r1[7];
              float v117_data = ir2[2];
              ir2[2] = (v117_data + (v112_data * v115_data));
              float v120_data = r1[9];
              float v122_data = ir2[3];
              ir2[3] = (v122_data + (v112_data * v120_data));
              float v125_data = r1[11];
              float v127_data = ir2[4];
              ir2[4] = (v127_data + (v112_data * v125_data));
              float v140_data = r0[4];
              float v144_data = r1[10];
              float v146_data = ir2[3];
              ir2[3] = (v146_data + (v140_data * v144_data));
              float v149_data = r1[12];
              float v151_data = ir2[4];
              ir2[4] = (v151_data + (v140_data * v149_data));
              float v154_data = r1[14];
              float v156_data = ir2[5];
              ir2[5] = (v156_data + (v140_data * v154_data));
              float v168_data = r0[5];
              float v173_data = r1[13];
              float v175_data = ir2[4];
              ir2[4] = (v175_data + (v168_data * v173_data));
              float v178_data = r1[15];
              float v180_data = ir2[5];
              ir2[5] = (v180_data + (v168_data * v178_data));
              float v183_data = r1[17];
              float v185_data = ir2[6];
              ir2[6] = (v185_data + (v168_data * v183_data));
              float v196_data = r0[6];
              float v202_data = r1[16];
              float v204_data = ir2[5];
              ir2[5] = (v204_data + (v196_data * v202_data));
              float v207_data = r1[18];
              float v209_data = ir2[6];
              ir2[6] = (v209_data + (v196_data * v207_data));
              float v212_data = r1[20];
              float v214_data = ir2[7];
              ir2[7] = (v214_data + (v196_data * v212_data));
              float v224_data = r0[7];
              float v231_data = r1[19];
              float v233_data = ir2[6];
              ir2[6] = (v233_data + (v224_data * v231_data));
              float v236_data = r1[21];
              float v238_data = ir2[7];
              ir2[7] = (v238_data + (v224_data * v236_data));
              float v241_data = r1[23];
              float v243_data = ir2[8];
              ir2[8] = (v243_data + (v224_data * v241_data));
              float v252_data = r0[8];
              float v260_data = r1[22];
              float v262_data = ir2[7];
              ir2[7] = (v262_data + (v252_data * v260_data));
              float v265_data = r1[24];
              float v267_data = ir2[8];
              ir2[8] = (v267_data + (v252_data * v265_data));
              float v270_data = r1[26];
              float v272_data = ir2[9];
              ir2[9] = (v272_data + (v252_data * v270_data));
              float v280_data = r0[9];
              float v289_data = r1[25];
              float v291_data = ir2[8];
              ir2[8] = (v291_data + (v280_data * v289_data));
              float v294_data = r1[27];
              float v296_data = ir2[9];
              ir2[9] = (v296_data + (v280_data * v294_data));
              float v299_data = r1[29];
              float v301_data = ir2[10];
              ir2[10] = (v301_data + (v280_data * v299_data));
              float v308_data = r0[10];
              float v318_data = r1[28];
              float v320_data = ir2[9];
              ir2[9] = (v320_data + (v308_data * v318_data));
              float v323_data = r1[30];
              float v325_data = ir2[10];
              ir2[10] = (v325_data + (v308_data * v323_data));
              float v328_data = r1[32];
              float v330_data = ir2[11];
              ir2[11] = (v330_data + (v308_data * v328_data));
              float v336_data = r0[11];
              float v347_data = r1[31];
              float v349_data = ir2[10];
              ir2[10] = (v349_data + (v336_data * v347_data));
              float v352_data = r1[33];
              float v354_data = ir2[11];
              ir2[11] = (v354_data + (v336_data * v352_data));
              float v357_data = r1[35];
              float v359_data = ir2[12];
              ir2[12] = (v359_data + (v336_data * v357_data));
              float v364_data = r0[12];
              float v376_data = r1[34];
              float v378_data = ir2[11];
              ir2[11] = (v378_data + (v364_data * v376_data));
              float v381_data = r1[36];
              float v383_data = ir2[12];
              ir2[12] = (v383_data + (v364_data * v381_data));
              float v386_data = r1[38];
              float v388_data = ir2[13];
              ir2[13] = (v388_data + (v364_data * v386_data));
              float v392_data = r0[13];
              float v405_data = r1[37];
              float v407_data = ir2[12];
              ir2[12] = (v407_data + (v392_data * v405_data));
              float v410_data = r1[39];
              float v412_data = ir2[13];
              ir2[13] = (v412_data + (v392_data * v410_data));
              float v415_data = r1[41];
              float v417_data = ir2[14];
              ir2[14] = (v417_data + (v392_data * v415_data));
              float v420_data = r0[14];
              float v434_data = r1[40];
              float v436_data = ir2[13];
              ir2[13] = (v436_data + (v420_data * v434_data));
              float v439_data = r1[42];
              float v441_data = ir2[14];
              ir2[14] = (v441_data + (v420_data * v439_data));
              float v444_data = r1[44];
              float v446_data = ir2[15];
              ir2[15] = (v446_data + (v420_data * v444_data));
              float v448_data = r0[15];
              float v463_data = r1[43];
              float v465_data = ir2[14];
              ir2[14] = (v465_data + (v448_data * v463_data));
              float v468_data = r1[45];
              float v470_data = ir2[15];
              ir2[15] = (v470_data + (v448_data * v468_data));
              #pragma unroll
              for (int32_t v472_n0 = 0; v472_n0 < 1; ++v472_n0) {
                #pragma unroll
                for (int32_t v473_n1 = 0; v473_n1 < 16; ++v473_n1) {
                  int32_t v474_a = v472_n0 + v473_n1;
                  float v475_data = ir2[v474_a];
                  r2[v474_a] = v475_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v476_i0 = 0; v476_i0 < 1; ++v476_i0) {
                int32_t v481_lead = v17_lead + (v476_i0 * 16);
                #pragma unroll
                for (int32_t v477_i1 = 0; v477_i1 < 16; ++v477_i1) {
                  float v479_data = r2[(v476_i0 + v477_i1)];
                  glb_m0[(v481_lead + (v477_i1 * 16))] = v479_data;
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

