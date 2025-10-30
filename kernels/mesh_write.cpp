#include <cstdint>

void kernel_main() {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);

    uint32_t ax_addr = get_arg_val<uint32_t>(1);
    uint32_t ay_addr = get_arg_val<uint32_t>(2);
    uint32_t az_addr = get_arg_val<uint32_t>(3);

    uint32_t adx_addr = get_arg_val<uint32_t>(4);
    uint32_t ady_addr = get_arg_val<uint32_t>(5);
    uint32_t adz_addr = get_arg_val<uint32_t>(6);

    constexpr auto cb_ax = tt::CBIndex::c_13;
    constexpr auto cb_ay = tt::CBIndex::c_14;
    constexpr auto cb_az = tt::CBIndex::c_15;

    constexpr auto cb_adx = tt::CBIndex::c_16;
    constexpr auto cb_ady = tt::CBIndex::c_17;
    constexpr auto cb_adz = tt::CBIndex::c_18;

    const uint32_t tile_size_bytes = get_tile_size(cb_ax);

    const InterleavedAddrGenFast<true> ax = { .bank_base_address = ax_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };
    const InterleavedAddrGenFast<true> ay = { .bank_base_address = ay_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };
    const InterleavedAddrGenFast<true> az = { .bank_base_address = az_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };

    const InterleavedAddrGenFast<true> adx = { .bank_base_address = adx_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };
    const InterleavedAddrGenFast<true> ady = { .bank_base_address = ady_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };
    const InterleavedAddrGenFast<true> adz = { .bank_base_address = adz_addr, .page_size = tile_size_bytes, .data_format = DataFormat::Float32, };

    for (uint32_t i = 0; i < n_tiles; i++) {
        cb_wait_front(cb_ax, 1);
        cb_wait_front(cb_ay, 1);
        cb_wait_front(cb_az, 1);

        cb_wait_front(cb_adx, 1);
        cb_wait_front(cb_ady, 1);
        cb_wait_front(cb_adz, 1);

        uint32_t cb_ax_addr = get_read_ptr(cb_ax);
        uint32_t cb_ay_addr = get_read_ptr(cb_ay);
        uint32_t cb_az_addr = get_read_ptr(cb_az);

        uint32_t cb_adx_addr = get_read_ptr(cb_adx);
        uint32_t cb_ady_addr = get_read_ptr(cb_ady);
        uint32_t cb_adz_addr = get_read_ptr(cb_adz);

        noc_async_write_tile(i, ax, cb_ax_addr);
        noc_async_write_tile(i, ay, cb_ay_addr);
        noc_async_write_tile(i, az, cb_az_addr);

        noc_async_write_tile(i, adx, cb_adx_addr);
        noc_async_write_tile(i, ady, cb_ady_addr);
        noc_async_write_tile(i, adz, cb_adz_addr);

        noc_async_write_barrier();

        cb_pop_front(cb_ax, 1);
        cb_pop_front(cb_ay, 1);
        cb_pop_front(cb_az, 1);

        cb_pop_front(cb_adx, 1);
        cb_pop_front(cb_ady, 1);
        cb_pop_front(cb_adz, 1);
    }
}

