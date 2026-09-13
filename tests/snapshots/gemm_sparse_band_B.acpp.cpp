// === base name ===
kernel_49e4db5504f49707

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
inline constexpr tensorforge::LaunchInfo launch_info_kernel_49e4db5504f49707 = {{16, 16, 1}, 16, 16, 1, 16, 1024, false, true, 1};
tensorforge::LaunchConfig launch_config_kernel_49e4db5504f49707(size_t numElements0, void* streamPtr = nullptr);
void launcher_kernel_49e4db5504f49707(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0 = nullptr, void* streamPtr = nullptr);


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
tensorforge::LaunchConfig launch_config_kernel_49e4db5504f49707(size_t numElements0, void* streamPtr) {
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
void launcher_kernel_49e4db5504f49707(float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0, void* streamPtr) {
  const tensorforge::LaunchConfig config = launch_config_kernel_49e4db5504f49707(numElements0, streamPtr);
  sycl::range<3> block (config.block[0], config.block[1], config.block[2]);
  sycl::range<3> grid (config.grid[0], config.grid[1], config.grid[2]);
  if (streamPtr == nullptr) {
    throw std::invalid_argument("stream may not be null!");
  }
  sycl::queue *stream = static_cast<sycl::queue *>(streamPtr);
  kernel_kernel_49e4db5504f49707(stream, grid, block, m0, m0_extraOffset, m1, m1_extraOffset, m2, m2_extraOffset, numElements0, flags0);
  CHECK_ERR;
}


// === kernel ===
inline void kernel_kernel_49e4db5504f49707(sycl::queue *stream, sycl::range<3> group_count, sycl::range<3> group_size, float * m0, size_t m0_extraOffset, const float * m1, size_t m1_extraOffset, const float * m2, size_t m2_extraOffset, size_t numElements0, unsigned * flags0) {
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
              // ir2 = +(r0 * r1)
              // [(0, 16), (0, 16)] [(0, 16)]
              float ir2[16]{};
              float v32_data = r0[0];
              float v33_data = r1[0];
              float v36_data = ir2[0];
              ir2[0] = (v36_data + (v32_data * (sycl::select_from_group(item.get_sub_group(), v33_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v42_data = ir2[1];
              ir2[1] = (v42_data + (v32_data * (sycl::select_from_group(item.get_sub_group(), v33_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v58_data = r0[1];
              float v62_data = ir2[0];
              ir2[0] = (v62_data + (v58_data * (sycl::select_from_group(item.get_sub_group(), v33_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v68_data = ir2[1];
              ir2[1] = (v68_data + (v58_data * (sycl::select_from_group(item.get_sub_group(), v33_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v74_data = ir2[2];
              ir2[2] = (v74_data + (v58_data * (sycl::select_from_group(item.get_sub_group(), v33_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v89_data = r0[2];
              float v94_data = ir2[1];
              ir2[1] = (v94_data + (v89_data * (sycl::select_from_group(item.get_sub_group(), v33_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v100_data = ir2[2];
              ir2[2] = (v100_data + (v89_data * (sycl::select_from_group(item.get_sub_group(), v33_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v106_data = ir2[3];
              ir2[3] = (v106_data + (v89_data * (sycl::select_from_group(item.get_sub_group(), v33_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v120_data = r0[3];
              float v126_data = ir2[2];
              ir2[2] = (v126_data + (v120_data * (sycl::select_from_group(item.get_sub_group(), v33_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v132_data = ir2[3];
              ir2[3] = (v132_data + (v120_data * (sycl::select_from_group(item.get_sub_group(), v33_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v138_data = ir2[4];
              ir2[4] = (v138_data + (v120_data * (sycl::select_from_group(item.get_sub_group(), v33_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v151_data = r0[4];
              float v158_data = ir2[3];
              ir2[3] = (v158_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v33_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v164_data = ir2[4];
              ir2[4] = (v164_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v33_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v170_data = ir2[5];
              ir2[5] = (v170_data + (v151_data * (sycl::select_from_group(item.get_sub_group(), v33_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v182_data = r0[5];
              float v190_data = ir2[4];
              ir2[4] = (v190_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v33_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v196_data = ir2[5];
              ir2[5] = (v196_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v33_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v199_data = r1[1];
              float v202_data = ir2[6];
              ir2[6] = (v202_data + (v182_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v213_data = r0[6];
              float v222_data = ir2[5];
              ir2[5] = (v222_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v228_data = ir2[6];
              ir2[6] = (v228_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v234_data = ir2[7];
              ir2[7] = (v234_data + (v213_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v244_data = r0[7];
              float v254_data = ir2[6];
              ir2[6] = (v254_data + (v244_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v260_data = ir2[7];
              ir2[7] = (v260_data + (v244_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v266_data = ir2[8];
              ir2[8] = (v266_data + (v244_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v275_data = r0[8];
              float v286_data = ir2[7];
              ir2[7] = (v286_data + (v275_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v292_data = ir2[8];
              ir2[8] = (v292_data + (v275_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v298_data = ir2[9];
              ir2[9] = (v298_data + (v275_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v306_data = r0[9];
              float v318_data = ir2[8];
              ir2[8] = (v318_data + (v306_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v324_data = ir2[9];
              ir2[9] = (v324_data + (v306_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v330_data = ir2[10];
              ir2[10] = (v330_data + (v306_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              float v337_data = r0[10];
              float v350_data = ir2[9];
              ir2[9] = (v350_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v356_data = ir2[10];
              ir2[10] = (v356_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (14)))));
              float v359_data = r1[2];
              float v362_data = ir2[11];
              ir2[11] = (v362_data + (v337_data * (sycl::select_from_group(item.get_sub_group(), v359_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (0)))));
              float v368_data = r0[11];
              float v382_data = ir2[10];
              ir2[10] = (v382_data + (v368_data * (sycl::select_from_group(item.get_sub_group(), v199_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (15)))));
              float v388_data = ir2[11];
              ir2[11] = (v388_data + (v368_data * (sycl::select_from_group(item.get_sub_group(), v359_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (1)))));
              float v394_data = ir2[12];
              ir2[12] = (v394_data + (v368_data * (sycl::select_from_group(item.get_sub_group(), v359_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (3)))));
              float v399_data = r0[12];
              float v414_data = ir2[11];
              ir2[11] = (v414_data + (v399_data * (sycl::select_from_group(item.get_sub_group(), v359_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (2)))));
              float v420_data = ir2[12];
              ir2[12] = (v420_data + (v399_data * (sycl::select_from_group(item.get_sub_group(), v359_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (4)))));
              float v426_data = ir2[13];
              ir2[13] = (v426_data + (v399_data * (sycl::select_from_group(item.get_sub_group(), v359_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (6)))));
              float v430_data = r0[13];
              float v446_data = ir2[12];
              ir2[12] = (v446_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v359_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (5)))));
              float v452_data = ir2[13];
              ir2[13] = (v452_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v359_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (7)))));
              float v458_data = ir2[14];
              ir2[14] = (v458_data + (v430_data * (sycl::select_from_group(item.get_sub_group(), v359_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (9)))));
              float v461_data = r0[14];
              float v478_data = ir2[13];
              ir2[13] = (v478_data + (v461_data * (sycl::select_from_group(item.get_sub_group(), v359_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (8)))));
              float v484_data = ir2[14];
              ir2[14] = (v484_data + (v461_data * (sycl::select_from_group(item.get_sub_group(), v359_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (10)))));
              float v490_data = ir2[15];
              ir2[15] = (v490_data + (v461_data * (sycl::select_from_group(item.get_sub_group(), v359_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (12)))));
              float v492_data = r0[15];
              float v510_data = ir2[14];
              ir2[14] = (v510_data + (v492_data * (sycl::select_from_group(item.get_sub_group(), v359_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (11)))));
              float v516_data = ir2[15];
              ir2[15] = (v516_data + (v492_data * (sycl::select_from_group(item.get_sub_group(), v359_data, (item.get_sub_group().get_local_linear_id() / 16) * 16 + (13)))));
              // r2 = ir2
              #pragma unroll
              for (int32_t v518_n0 = 0; v518_n0 < 1; ++v518_n0) {
                #pragma unroll
                for (int32_t v519_n1 = 0; v519_n1 < 16; ++v519_n1) {
                  int32_t v520_a = v518_n0 + v519_n1;
                  float v521_data = ir2[v520_a];
                  r2[v520_a] = v521_data;
                }
              }
              // glb_m0 = store{r>g}(r2);
              #pragma unroll
              for (int32_t v522_i0 = 0; v522_i0 < 1; ++v522_i0) {
                int32_t v527_lead = v17_lead + (v522_i0 * 16);
                #pragma unroll
                for (int32_t v523_i1 = 0; v523_i1 < 16; ++v523_i1) {
                  float v525_data = r2[(v522_i0 + v523_i1)];
                  glb_m0[(v527_lead + (v523_i1 * 16))] = v525_data;
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

