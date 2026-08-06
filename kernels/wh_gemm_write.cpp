#include "api/dataflow/dataflow_api.h"
#include <cstdint>

void kernel_main(){
    const uint32_t count=get_arg_val<uint32_t>(0),first=get_arg_val<uint32_t>(1);
    const uint32_t hw_addr=get_arg_val<uint32_t>(2),hu_addr=get_arg_val<uint32_t>(3);
    constexpr uint32_t hw_cb=get_compile_time_arg_val(0),hu_cb=get_compile_time_arg_val(1);
    const uint32_t bytes=get_tile_size(hw_cb);
    const InterleavedAddrGenFast<true> hw={.bank_base_address=hw_addr,.page_size=bytes,.data_format=DataFormat::Float32};
    const InterleavedAddrGenFast<true> hu={.bank_base_address=hu_addr,.page_size=bytes,.data_format=DataFormat::Float32};
    for(uint32_t i=first;i<first+count;++i){
        cb_wait_front(hw_cb,1); cb_wait_front(hu_cb,1);
        noc_async_write_tile(i,hw,get_read_ptr(hw_cb)); noc_async_write_tile(i,hu,get_read_ptr(hu_cb));
        noc_async_write_barrier(); cb_pop_front(hw_cb,1); cb_pop_front(hu_cb,1);
    }
}
