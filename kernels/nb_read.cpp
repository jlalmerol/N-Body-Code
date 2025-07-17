#include "dataflow_api.h"
#include "debug/dprint.h"
#include <cstdint>

void kernel_main() {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    uint32_t start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t N = get_arg_val<uint32_t>(2);

    uint32_t pxi_addr = get_arg_val<uint32_t>(3);
    uint32_t pxj_addr = get_arg_val<uint32_t>(4);
    uint32_t pyi_addr = get_arg_val<uint32_t>(5);
    uint32_t pyj_addr = get_arg_val<uint32_t>(6);
    uint32_t pzi_addr = get_arg_val<uint32_t>(7);
    uint32_t pzj_addr = get_arg_val<uint32_t>(8);

    uint32_t vxi_addr = get_arg_val<uint32_t>(9);
    uint32_t vxj_addr = get_arg_val<uint32_t>(10);
    uint32_t vyi_addr = get_arg_val<uint32_t>(11);
    uint32_t vyj_addr = get_arg_val<uint32_t>(12);
    uint32_t vzi_addr = get_arg_val<uint32_t>(13);
    uint32_t vzj_addr = get_arg_val<uint32_t>(14);

    uint32_t eps_addr = get_arg_val<uint32_t>(15);
    uint32_t pwj_addr = get_arg_val<uint32_t>(16);
    uint32_t cns_addr = get_arg_val<uint32_t>(17);

    constexpr uint32_t cb_pxi = get_compile_time_arg_val(0);
    constexpr uint32_t cb_pxj = get_compile_time_arg_val(1);
    constexpr uint32_t cb_pyi = get_compile_time_arg_val(2);
    constexpr uint32_t cb_pyj = get_compile_time_arg_val(3);
    constexpr uint32_t cb_pzi = get_compile_time_arg_val(4);
    constexpr uint32_t cb_pzj = get_compile_time_arg_val(5);

    constexpr uint32_t cb_vxi = get_compile_time_arg_val(6);
    constexpr uint32_t cb_vxj = get_compile_time_arg_val(7);
    constexpr uint32_t cb_vyi = get_compile_time_arg_val(8);
    constexpr uint32_t cb_vyj = get_compile_time_arg_val(9);
    constexpr uint32_t cb_vzi = get_compile_time_arg_val(10);
    constexpr uint32_t cb_vzj = get_compile_time_arg_val(11);

    constexpr uint32_t cb_eps = get_compile_time_arg_val(12);
    constexpr uint32_t cb_pwj = get_compile_time_arg_val(13);
    constexpr uint32_t cb_cns = get_compile_time_arg_val(14);

    const uint32_t tile_size_bytes = get_tile_size(cb_pxi);

    const InterleavedAddrGenFast<true> pxi = {.bank_base_address = pxi_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };
    const InterleavedAddrGenFast<true> pxj = {.bank_base_address = pxj_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };
    const InterleavedAddrGenFast<true> pyi = {.bank_base_address = pyi_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };
    const InterleavedAddrGenFast<true> pyj = {.bank_base_address = pyj_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };
    const InterleavedAddrGenFast<true> pzi = {.bank_base_address = pzi_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };
    const InterleavedAddrGenFast<true> pzj = {.bank_base_address = pzj_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };

    const InterleavedAddrGenFast<true> vxi = {.bank_base_address = vxi_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };
    const InterleavedAddrGenFast<true> vxj = {.bank_base_address = vxj_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };
    const InterleavedAddrGenFast<true> vyi = {.bank_base_address = vyi_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };
    const InterleavedAddrGenFast<true> vyj = {.bank_base_address = vyj_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };
    const InterleavedAddrGenFast<true> vzi = {.bank_base_address = vzi_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };
    const InterleavedAddrGenFast<true> vzj = {.bank_base_address = vzj_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };

    const InterleavedAddrGenFast<true> eps = {.bank_base_address = eps_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };
    const InterleavedAddrGenFast<true> pwj = {.bank_base_address = pwj_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };
    const InterleavedAddrGenFast<true> cns = {.bank_base_address = cns_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };

    const uint32_t end_tile_id = start_tile_id + n_tiles;

    for (uint32_t i = start_tile_id; i < end_tile_id; i++) {
        {
        cb_reserve_back(cb_pxi, 1);
        cb_reserve_back(cb_pyi, 1);
        cb_reserve_back(cb_pzi, 1);

        cb_reserve_back(cb_vxi, 1);
        cb_reserve_back(cb_vyi, 1);
        cb_reserve_back(cb_vzi, 1);

        cb_reserve_back(cb_eps, 1);
        cb_reserve_back(cb_cns, 1);

        uint32_t cb_pxi_addr = get_write_ptr(cb_pxi);
        uint32_t cb_pyi_addr = get_write_ptr(cb_pyi);
        uint32_t cb_pzi_addr = get_write_ptr(cb_pzi);

        uint32_t cb_vxi_addr = get_write_ptr(cb_vxi);
        uint32_t cb_vyi_addr = get_write_ptr(cb_vyi);
	    uint32_t cb_vzi_addr = get_write_ptr(cb_vzi);

        uint32_t cb_eps_addr = get_write_ptr(cb_eps);
        uint32_t cb_cns_addr = get_write_ptr(cb_cns);

        noc_async_read_tile(i, pxi, cb_pxi_addr);
        noc_async_read_tile(i, pyi, cb_pyi_addr);
        noc_async_read_tile(i, pzi, cb_pzi_addr);

        noc_async_read_tile(i, vxi, cb_vxi_addr);
        noc_async_read_tile(i, vyi, cb_vyi_addr);
        noc_async_read_tile(i, vzi, cb_vzi_addr);

        noc_async_read_tile(i, eps, cb_eps_addr);
        noc_async_read_tile(i, cns, cb_cns_addr);

        noc_async_read_barrier();

        cb_push_back(cb_pxi, 1);
        cb_push_back(cb_pyi, 1);
        cb_push_back(cb_pzi, 1);

        cb_push_back(cb_vxi, 1);
        cb_push_back(cb_vyi, 1);
        cb_push_back(cb_vzi, 1);

        cb_push_back(cb_eps, 1);
        cb_push_back(cb_cns, 1);
        }
	
        for (uint32_t j = 0; j < N; j++) {
            {
            cb_reserve_back(cb_pxj, 1);
            cb_reserve_back(cb_pyj, 1);
            cb_reserve_back(cb_pzj, 1);
        
            cb_reserve_back(cb_vxj, 1);
            cb_reserve_back(cb_vyj, 1);
            cb_reserve_back(cb_vzj, 1);

	        cb_reserve_back(cb_pwj, 1);

            uint32_t cb_pxj_addr = get_write_ptr(cb_pxj);
            uint32_t cb_pyj_addr = get_write_ptr(cb_pyj);
            uint32_t cb_pzj_addr = get_write_ptr(cb_pzj);
        
            uint32_t cb_vxj_addr = get_write_ptr(cb_vxj);
            uint32_t cb_vyj_addr = get_write_ptr(cb_vyj);
            uint32_t cb_vzj_addr = get_write_ptr(cb_vzj);

	        uint32_t cb_pwj_addr = get_write_ptr(cb_pwj);
                
            noc_async_read_tile(j, pxj, cb_pxj_addr);
            noc_async_read_tile(j, pyj, cb_pyj_addr);
            noc_async_read_tile(j, pzj, cb_pzj_addr);
        
            noc_async_read_tile(j, vxj, cb_vxj_addr);
            noc_async_read_tile(j, vyj, cb_vyj_addr);
            noc_async_read_tile(j, vzj, cb_vzj_addr);

	         noc_async_read_tile(j, pwj, cb_pwj_addr);
                
            noc_async_read_barrier(); 
                
            cb_push_back(cb_pxj, 1);
            cb_push_back(cb_pyj, 1);
            cb_push_back(cb_pzj, 1);
        
            cb_push_back(cb_vxj, 1);
            cb_push_back(cb_vyj, 1);
            cb_push_back(cb_vzj, 1);

	        cb_push_back(cb_pwj, 1);
            }
        }
    }
}

