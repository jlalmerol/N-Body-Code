// One MPI rank controls one Wormhole device. One device tile represents
// 32 target particles by 32 source particles.
namespace {

constexpr uint32_t WHG_TILE = 32;
constexpr uint32_t WHG_TILE_ELEMS = WHG_TILE * WHG_TILE;

// TT Float32 tiles are four 16x16 faces. Pack inputs directly in that layout
// and read only the output elements used by the host.
constexpr size_t whg_physical_index(size_t tile, uint32_t row, uint32_t column) {
    const uint32_t face = (row / 16) * 2 + column / 16;
    const uint32_t face_offset = (row % 16) * 16 + column % 16;
    return tile * WHG_TILE_ELEMS + face * 256 + face_offset;
}

inline void whg_set(std::vector<float>& tiles, size_t tile, uint32_t row, uint32_t column, float value) {
    tiles[whg_physical_index(tile, row, column)] = value;
}

inline float whg_get(const std::vector<float>& tiles, size_t tile, uint32_t row, uint32_t column) {
    return tiles[whg_physical_index(tile, row, column)];
}

struct WhgPersistentContext {
    uint32_t global_n = 0;
    uint32_t local_n = 0;
    uint32_t global_target_tile0 = 0;
    uint32_t target_tiles = 0;
    uint32_t source_tiles = 0;
    std::shared_ptr<MeshDevice> mesh_device;
    std::shared_ptr<MeshBuffer> ar_buf, av_buf, br_buf, bv_buf, p_buf, mask_buf, hw_buf, hu_buf;
    MeshWorkload workload;
};

std::unique_ptr<WhgPersistentContext> whg_context;

}  // namespace

void shutdown_force_calculation_backend() {
    if (whg_context && whg_context->mesh_device) {
        whg_context->mesh_device->close();
    }
    whg_context.reset();
}

void force_calculation(double* __restrict h_par_x, double* __restrict h_par_y,
                       double* __restrict h_par_z, double* __restrict h_par_w,
                       double* __restrict h_vel_x, double* __restrict h_vel_y,
                       double* __restrict h_vel_z, double* __restrict h_acc_x,
                       double* __restrict h_acc_y, double* __restrict h_acc_z,
                       double* __restrict h_adot_x, double* __restrict h_adot_y,
                       double* __restrict h_adot_z, Sys& sys) {
    if ((sys.start % WHG_TILE) != 0) {
        throw std::runtime_error("_WH_GEMM_MULTIHOST requires MPI target ranges aligned to 32 particles");
    }

    const uint32_t local_n = static_cast<uint32_t>(sys.end - sys.start);
    const uint32_t target_tiles = (local_n + WHG_TILE - 1) / WHG_TILE;
    const uint32_t source_tiles = (static_cast<uint32_t>(sys.N) + WHG_TILE - 1) / WHG_TILE;
    const uint32_t global_target_tile0 = static_cast<uint32_t>(sys.start) / WHG_TILE;

    // Physical TT-face-layout augmented matrices. P contains mass-weighted
    // moments so W and U depend only on D and C on the SFPU.
    std::vector<float> ar(target_tiles * WHG_TILE_ELEMS, 0.0f);
    std::vector<float> av(target_tiles * WHG_TILE_ELEMS, 0.0f);
    std::vector<float> br(source_tiles * WHG_TILE_ELEMS, 0.0f);
    std::vector<float> bv(source_tiles * WHG_TILE_ELEMS, 0.0f);
    std::vector<float> p(source_tiles * WHG_TILE_ELEMS, 0.0f);

    for (uint32_t it = 0; it < target_tiles; ++it) {
        for (uint32_t ii = 0; ii < WHG_TILE; ++ii) {
            const uint32_t local_i = it * WHG_TILE + ii;
            if (local_i >= local_n) continue;
            const uint32_t i = static_cast<uint32_t>(sys.start) + local_i;
            const float x = h_par_x[i], y = h_par_y[i], z = h_par_z[i];
            const float vx = h_vel_x[i], vy = h_vel_y[i], vz = h_vel_z[i];
            whg_set(ar,it,ii,0,x*x+y*y+z*z); whg_set(ar,it,ii,1,x);
            whg_set(ar,it,ii,2,y); whg_set(ar,it,ii,3,z); whg_set(ar,it,ii,4,1.0f);
            whg_set(av,it,ii,0,1.0f); whg_set(av,it,ii,1,vx); whg_set(av,it,ii,2,vy);
            whg_set(av,it,ii,3,vz); whg_set(av,it,ii,4,x); whg_set(av,it,ii,5,y);
            whg_set(av,it,ii,6,z); whg_set(av,it,ii,7,x*vx+y*vy+z*vz);
        }
    }

    for (uint32_t jt = 0; jt < source_tiles; ++jt) {
        for (uint32_t jj = 0; jj < WHG_TILE; ++jj) {
            const uint32_t j = jt * WHG_TILE + jj;
            if (j >= static_cast<uint32_t>(sys.N)) continue;
            const float x = h_par_x[j], y = h_par_y[j], z = h_par_z[j];
            const float vx = h_vel_x[j], vy = h_vel_y[j], vz = h_vel_z[j];
            const float m = h_par_w[j];
            whg_set(br,jt,0,jj,1.0f); whg_set(br,jt,1,jj,-2*x); whg_set(br,jt,2,jj,-2*y);
            whg_set(br,jt,3,jj,-2*z); whg_set(br,jt,4,jj,x*x+y*y+z*z);
            whg_set(bv,jt,0,jj,x*vx+y*vy+z*vz); whg_set(bv,jt,1,jj,-x);
            whg_set(bv,jt,2,jj,-y); whg_set(bv,jt,3,jj,-z); whg_set(bv,jt,4,jj,-vx);
            whg_set(bv,jt,5,jj,-vy); whg_set(bv,jt,6,jj,-vz); whg_set(bv,jt,7,jj,1.0f);
            whg_set(p,jt,jj,0,m); whg_set(p,jt,jj,1,m*x); whg_set(p,jt,jj,2,m*y);
            whg_set(p,jt,jj,3,m*z); whg_set(p,jt,jj,4,m*vx);
            whg_set(p,jt,jj,5,m*vy); whg_set(p,jt,jj,6,m*vz);
        }
    }

    // Eight reusable masks indexed by: diagonal | target_tail<<1 | source_tail<<2.
    std::vector<float> masks(8 * WHG_TILE_ELEMS, 0.0f);
    const uint32_t target_tail = local_n - (target_tiles - 1) * WHG_TILE;
    const uint32_t source_tail = static_cast<uint32_t>(sys.N) - (source_tiles - 1) * WHG_TILE;
    for (uint32_t kind = 0; kind < 8; ++kind) {
        const bool diagonal = kind & 1;
        const uint32_t valid_i = (kind & 2) ? target_tail : WHG_TILE;
        const uint32_t valid_j = (kind & 4) ? source_tail : WHG_TILE;
        for (uint32_t i = 0; i < valid_i; ++i)
            for (uint32_t j = 0; j < valid_j; ++j)
                whg_set(masks,kind,i,j,(!diagonal || i != j) ? 1.0f : 0.0f);
    }

    if (!whg_context) {
        whg_context = std::make_unique<WhgPersistentContext>();
        auto& context = *whg_context;
        context.global_n = static_cast<uint32_t>(sys.N);
        context.local_n = local_n;
        context.global_target_tile0 = global_target_tile0;
        context.target_tiles = target_tiles;
        context.source_tiles = source_tiles;
        context.mesh_device = MeshDevice::create(MeshDeviceConfig(MeshShape(1, 1)));
        Program program = CreateProgram();
        context.ar_buf = MakeReplicatedBuffer(context.mesh_device, target_tiles);
        context.av_buf = MakeReplicatedBuffer(context.mesh_device, target_tiles);
        context.br_buf = MakeReplicatedBuffer(context.mesh_device, source_tiles);
        context.bv_buf = MakeReplicatedBuffer(context.mesh_device, source_tiles);
        context.p_buf = MakeReplicatedBuffer(context.mesh_device, source_tiles);
        context.mask_buf = MakeReplicatedBuffer(context.mesh_device, 8);
        context.hw_buf = MakeReplicatedBuffer(context.mesh_device, target_tiles);
        context.hu_buf = MakeReplicatedBuffer(context.mesh_device, target_tiles);

        auto grid = context.mesh_device->compute_with_storage_grid_size();
        const uint32_t total_cores = grid.x * grid.y;
        auto all_cores = CoreRange(CoreCoord(0, 0), CoreCoord(grid.x - 1, grid.y - 1));
        for (uint32_t cb=0;cb<=5;++cb) MakeCircularBufferFP32(program,all_cores,static_cast<tt::CBIndex>(cb),2);
        for (uint32_t cb=16;cb<=23;++cb) MakeCircularBufferFP32(program,all_cores,static_cast<tt::CBIndex>(cb),2);
        std::vector<uint32_t> reader_ct={0,1,2,3,4,5}, writer_ct={16,17};
        std::vector<uint32_t> compute_ct={0,1,2,3,4,5,16,17,18,19,20,21,22,23};
        auto reader=CreateKernel(program,"N-Body-Code/kernels/wh_gemm_read.cpp",all_cores,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_0,.noc=NOC::RISCV_0_default,.compile_args=reader_ct});
        auto writer=CreateKernel(program,"N-Body-Code/kernels/wh_gemm_write.cpp",all_cores,
            DataMovementConfig{.processor=DataMovementProcessor::RISCV_1,.noc=NOC::RISCV_1_default,.compile_args=writer_ct});
        auto compute=CreateKernel(program,"N-Body-Code/kernels/wh_gemm_compute.cpp",all_cores,
            ComputeConfig{.math_fidelity=MathFidelity::HiFi4,.fp32_dest_acc_en=true,
                          .dst_full_sync_en=true,.compile_args=compute_ct,.opt_level=KernelBuildOptLevel::O3});
        constexpr bool row_major=true;
        auto [num_cores,cores_used,group1,group2,per1,per2]=split_work_to_cores(grid,target_tiles,row_major);
        sys.num_cores=static_cast<int>(num_cores);
        auto cores=grid_to_cores(total_cores,grid.x,grid.y,row_major);
        for(uint32_t ci=0,start_tile=0;ci<total_cores;++ci){
            const auto& core=cores[ci];
            uint32_t count=group1.contains(core)?per1:(group2.contains(core)?per2:0);
            if(!count){
                SetRuntimeArgs(program,reader,core,std::vector<uint32_t>(14,0));
                SetRuntimeArgs(program,writer,core,std::vector<uint32_t>(6,0));
                SetRuntimeArgs(program,compute,core,std::vector<uint32_t>(4,0));
                continue;
            }
            SetRuntimeArgs(program,reader,core,{count,start_tile,source_tiles,global_target_tile0,
                context.ar_buf->address(),context.br_buf->address(),context.av_buf->address(),context.bv_buf->address(),
                context.p_buf->address(),context.mask_buf->address(),target_tiles,source_tiles,local_n,static_cast<uint32_t>(sys.N)});
            SetRuntimeArgs(program,writer,core,{count,start_tile,context.hw_buf->address(),context.hu_buf->address(),target_tiles,0});
            const float eps2=static_cast<float>(EPS2); uint32_t eps2_bits;
            std::memcpy(&eps2_bits,&eps2,sizeof(eps2_bits));
            SetRuntimeArgs(program,compute,core,{count,start_tile,source_tiles,eps2_bits});
            start_tile+=count;
        }
        auto device_range=MeshCoordinateRange(context.mesh_device->shape());
        context.workload.add_program(device_range,std::move(program));
    }

    auto& context = *whg_context;
    if (context.global_n != static_cast<uint32_t>(sys.N) || context.local_n != local_n ||
        context.global_target_tile0 != global_target_tile0 || context.target_tiles != target_tiles ||
        context.source_tiles != source_tiles) {
        throw std::runtime_error("_WH_GEMM_MULTIHOST persistent context cannot change problem or MPI partition size");
    }
    sys.num_cores = static_cast<int>(std::min<uint32_t>(target_tiles,
        context.mesh_device->compute_with_storage_grid_size().x * context.mesh_device->compute_with_storage_grid_size().y));
    auto& cq = context.mesh_device->mesh_command_queue();
    EnqueueWriteMeshBuffer(cq,context.ar_buf,ar,false); EnqueueWriteMeshBuffer(cq,context.av_buf,av,false);
    EnqueueWriteMeshBuffer(cq,context.br_buf,br,false); EnqueueWriteMeshBuffer(cq,context.bv_buf,bv,false);
    EnqueueWriteMeshBuffer(cq,context.p_buf,p,false); EnqueueWriteMeshBuffer(cq,context.mask_buf,masks,false);

    auto start = std::chrono::high_resolution_clock::now();
    EnqueueMeshWorkload(cq, context.workload, false);
    std::vector<float> hw_physical(target_tiles*WHG_TILE_ELEMS), hu_physical(target_tiles*WHG_TILE_ELEMS);
    EnqueueReadMeshBuffer(cq, hw_physical, context.hw_buf, true);
    EnqueueReadMeshBuffer(cq, hu_physical, context.hu_buf, true);
    auto end = std::chrono::high_resolution_clock::now();
    sys.kernel_time = std::chrono::duration<double>(end-start).count();
    std::fill(h_acc_x,h_acc_x+sys.N,0.0); std::fill(h_acc_y,h_acc_y+sys.N,0.0);
    std::fill(h_acc_z,h_acc_z+sys.N,0.0); std::fill(h_adot_x,h_adot_x+sys.N,0.0);
    std::fill(h_adot_y,h_adot_y+sys.N,0.0); std::fill(h_adot_z,h_adot_z+sys.N,0.0);
    for (uint32_t li=0; li<local_n; ++li) {
        const uint32_t i=sys.start+li, tile=li/WHG_TILE, row=li%WHG_TILE;
        const double sw=whg_get(hw_physical,tile,row,0), su=whg_get(hu_physical,tile,row,0);
        h_acc_x[i]=whg_get(hw_physical,tile,row,1)-h_par_x[i]*sw;
        h_acc_y[i]=whg_get(hw_physical,tile,row,2)-h_par_y[i]*sw;
        h_acc_z[i]=whg_get(hw_physical,tile,row,3)-h_par_z[i]*sw;
        h_adot_x[i]=whg_get(hw_physical,tile,row,4)-h_vel_x[i]*sw+whg_get(hu_physical,tile,row,1)-h_par_x[i]*su;
        h_adot_y[i]=whg_get(hw_physical,tile,row,5)-h_vel_y[i]*sw+whg_get(hu_physical,tile,row,2)-h_par_y[i]*su;
        h_adot_z[i]=whg_get(hw_physical,tile,row,6)-h_vel_z[i]*sw+whg_get(hu_physical,tile,row,3)-h_par_z[i]*su;
    }
}
