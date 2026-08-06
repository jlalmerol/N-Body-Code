#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main() {
    const uint32_t count=get_arg_val<uint32_t>(0), first=get_arg_val<uint32_t>(1);
    const uint32_t source_tiles=get_arg_val<uint32_t>(2), global_i0=get_arg_val<uint32_t>(3);
    const uint32_t ar_addr=get_arg_val<uint32_t>(4), br_addr=get_arg_val<uint32_t>(5);
    const uint32_t av_addr=get_arg_val<uint32_t>(6), bv_addr=get_arg_val<uint32_t>(7);
    const uint32_t p_addr=get_arg_val<uint32_t>(8), mask_addr=get_arg_val<uint32_t>(9);
    const uint32_t target_tiles=get_arg_val<uint32_t>(10), source_tiles_again=get_arg_val<uint32_t>(11);
    const uint32_t local_n=get_arg_val<uint32_t>(12), global_n=get_arg_val<uint32_t>(13);
    (void)source_tiles_again;
    constexpr uint32_t ar_cb=get_compile_time_arg_val(0), br_cb=get_compile_time_arg_val(1);
    constexpr uint32_t av_cb=get_compile_time_arg_val(2), bv_cb=get_compile_time_arg_val(3);
    constexpr uint32_t p_cb=get_compile_time_arg_val(4), mask_cb=get_compile_time_arg_val(5);
    const uint32_t bytes=get_tile_size(ar_cb);
    const InterleavedAddrGenFast<true> ar={.bank_base_address=ar_addr,.page_size=bytes,.data_format=DataFormat::Float32};
    const InterleavedAddrGenFast<true> br={.bank_base_address=br_addr,.page_size=bytes,.data_format=DataFormat::Float32};
    const InterleavedAddrGenFast<true> av={.bank_base_address=av_addr,.page_size=bytes,.data_format=DataFormat::Float32};
    const InterleavedAddrGenFast<true> bv={.bank_base_address=bv_addr,.page_size=bytes,.data_format=DataFormat::Float32};
    const InterleavedAddrGenFast<true> p={.bank_base_address=p_addr,.page_size=bytes,.data_format=DataFormat::Float32};
    const InterleavedAddrGenFast<true> mask={.bank_base_address=mask_addr,.page_size=bytes,.data_format=DataFormat::Float32};
    const uint32_t target_tail=local_n-(target_tiles-1)*32;
    const uint32_t source_tail=global_n-(source_tiles-1)*32;
    for(uint32_t it=first;it<first+count;++it){
        const uint32_t global_it=global_i0+it;
        for(uint32_t jt=0;jt<source_tiles;++jt){
            const bool diag=global_it==jt;
            const bool ti=(it==target_tiles-1)&&(target_tail<32);
            const bool sj=(jt==source_tiles-1)&&(source_tail<32);
            const uint32_t mask_id=(diag?1u:0u)|(ti?2u:0u)|(sj?4u:0u);
            cb_reserve_back(ar_cb,1); cb_reserve_back(br_cb,1); cb_reserve_back(av_cb,1);
            cb_reserve_back(bv_cb,1); cb_reserve_back(p_cb,1); cb_reserve_back(mask_cb,1);
            noc_async_read_tile(it,ar,get_write_ptr(ar_cb)); noc_async_read_tile(jt,br,get_write_ptr(br_cb));
            noc_async_read_tile(it,av,get_write_ptr(av_cb)); noc_async_read_tile(jt,bv,get_write_ptr(bv_cb));
            noc_async_read_tile(jt,p,get_write_ptr(p_cb)); noc_async_read_tile(mask_id,mask,get_write_ptr(mask_cb));
            noc_async_read_barrier(); cb_push_back(ar_cb,1); cb_push_back(br_cb,1);
            cb_push_back(av_cb,1); cb_push_back(bv_cb,1);
            cb_push_back(p_cb,1); cb_push_back(mask_cb,1);
        }
    }
}
