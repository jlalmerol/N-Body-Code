#include <cstdint>
#include "api/compute/compute_kernel_hw_startup.h"
#include "api/compute/compute_kernel_api.h"
#include "api/compute/matmul.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/binary_max_min.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_unary/fill.h"
#include "api/compute/eltwise_unary/rsqrt.h"
#include "tools/profiler/kernel_profiler.hpp"

static inline void one_tile_mm(uint32_t a,uint32_t b,uint32_t out){
    cb_wait_front(a,1); cb_wait_front(b,1); cb_reserve_back(out,1);
    matmul_init(a,b); tile_regs_acquire(); matmul_tiles(a,b,0,0,0);
    tile_regs_commit(); tile_regs_wait(); pack_tile(0,out); tile_regs_release();
    cb_push_back(out,1);
}

void kernel_main(){
    const uint32_t count=get_arg_val<uint32_t>(0), source_tiles=get_arg_val<uint32_t>(2);
    const uint32_t eps2_bits=get_arg_val<uint32_t>(3);
    constexpr uint32_t ar=get_compile_time_arg_val(0),br=get_compile_time_arg_val(1);
    constexpr uint32_t av=get_compile_time_arg_val(2),bv=get_compile_time_arg_val(3);
    constexpr uint32_t p=get_compile_time_arg_val(4),mask=get_compile_time_arg_val(5);
    constexpr uint32_t hw=get_compile_time_arg_val(6),hu=get_compile_time_arg_val(7);
    constexpr uint32_t d=get_compile_time_arg_val(8),c=get_compile_time_arg_val(9);
    constexpr uint32_t w=get_compile_time_arg_val(10),u=get_compile_time_arg_val(11);
    constexpr uint32_t accw=get_compile_time_arg_val(12),accu=get_compile_time_arg_val(13);
    compute_kernel_hw_startup<SrcOrder::Reverse>(ar,br,hw);
    // Profile compute after the required hardware startup.
    DeviceZoneScopedN("NBody device compute");
    for(uint32_t it=0;it<count;++it){
        cb_reserve_back(accw,1); cb_reserve_back(accu,1); tile_regs_acquire();
        fill_tile_init();
        fill_tile(0,0.0f); fill_tile(1,0.0f); tile_regs_commit(); tile_regs_wait();
        pack_tile(0,accw); pack_tile(1,accu); tile_regs_release(); cb_push_back(accw,1); cb_push_back(accu,1);
        for(uint32_t jt=0;jt<source_tiles;++jt){
            one_tile_mm(ar,br,d); one_tile_mm(av,bv,c);
            cb_wait_front(d,1); cb_wait_front(c,1); cb_wait_front(mask,1);
            cb_reserve_back(w,1); cb_reserve_back(u,1); tile_regs_acquire();
            copy_tile_to_dst_init_short(d);
            copy_tile(d,0,0); copy_tile(d,0,1); copy_tile(c,0,3); copy_tile(mask,0,4);
            fill_tile_init(); fill_tile_bitcast(2,eps2_bits); fill_tile(6,0.0f);
            binary_max_tile_init(); binary_max_tile(0,6,0); binary_max_tile(1,6,1);
            add_binary_tile_init();
            add_binary_tile(0,2,0); add_binary_tile(1,2,1);
            rsqrt_tile_init(); rsqrt_tile(0); rsqrt_tile(1);
            square_tile_init(); square_tile(1);
            mul_binary_tile_init();
            mul_binary_tile(0,1,0);                 // W = s^-3/2
            mul_binary_tile(3,0,3); mul_binary_tile(3,1,3); // C*s^-5/2
            add_binary_tile_init();
            add_binary_tile(3,3,5); add_binary_tile(5,3,5);
            sub_binary_tile_init();
            sub_binary_tile(6,5,3);                 // U = -3*C*s^-5/2
            mul_binary_tile_init();
            mul_binary_tile(0,4,0); mul_binary_tile(3,4,3);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0,w); pack_tile(3,u);
            tile_regs_release(); cb_push_back(w,1); cb_push_back(u,1);
            cb_pop_front(d,1); cb_pop_front(c,1); cb_pop_front(mask,1);

            one_tile_mm(w,p,d); one_tile_mm(u,p,c);
            cb_wait_front(accw,1); cb_wait_front(accu,1); cb_wait_front(d,1); cb_wait_front(c,1);
            tile_regs_acquire(); copy_tile_to_dst_init_short(accw); copy_tile(accw,0,0); copy_tile(accu,0,1);
            copy_tile(d,0,2); copy_tile(c,0,3); add_binary_tile_init();
            add_binary_tile(0,2,0); add_binary_tile(1,3,1);
            cb_pop_front(accw,1); cb_pop_front(accu,1); cb_pop_front(d,1); cb_pop_front(c,1);
            cb_reserve_back(accw,1); cb_reserve_back(accu,1);
            tile_regs_commit(); tile_regs_wait(); pack_tile(0,accw); pack_tile(1,accu);
            tile_regs_release(); cb_push_back(accw,1); cb_push_back(accu,1);

            cb_pop_front(ar,1); cb_pop_front(br,1); cb_pop_front(av,1); cb_pop_front(bv,1);
            cb_pop_front(p,1); cb_pop_front(w,1); cb_pop_front(u,1);
        }
        cb_wait_front(accw,1); cb_wait_front(accu,1); cb_reserve_back(hw,1); cb_reserve_back(hu,1);
        tile_regs_acquire(); copy_tile_to_dst_init_short(accw); copy_tile(accw,0,0); copy_tile(accu,0,1);
        tile_regs_commit(); tile_regs_wait(); pack_tile(0,hw); pack_tile(1,hu);
        tile_regs_release(); cb_push_back(hw,1); cb_push_back(hu,1);
        cb_pop_front(accw,1); cb_pop_front(accu,1);
    }
}
