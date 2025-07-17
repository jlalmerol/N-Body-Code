#include <cstdint>
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/eltwise_binary.h"
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/eltwise_unary.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "compute_kernel_api/binary_bitwise_sfpu.h"
#include "compute_kernel_api/binary_shift.h"
#include "compute_kernel_api.h"
#include "compute_kernel_api/eltwise_unary/fill.h"

#include "debug/dprint.h"

namespace NAMESPACE {
void MAIN {
    uint32_t n_tiles = get_arg_val<uint32_t>(0);
    uint32_t start_tile_id = get_arg_val<uint32_t>(1);
    uint32_t N = get_arg_val<uint32_t>(2);

    constexpr auto cb_pxi = get_compile_time_arg_val(0);
    constexpr auto cb_pxj = get_compile_time_arg_val(1);
    constexpr auto cb_pyi = get_compile_time_arg_val(2);
    constexpr auto cb_pyj = get_compile_time_arg_val(3);
    constexpr auto cb_pzi = get_compile_time_arg_val(4);
    constexpr auto cb_pzj = get_compile_time_arg_val(5);

    constexpr auto cb_vxi = get_compile_time_arg_val(6);
    constexpr auto cb_vxj = get_compile_time_arg_val(7);
    constexpr auto cb_vyi = get_compile_time_arg_val(8);
    constexpr auto cb_vyj = get_compile_time_arg_val(9);
    constexpr auto cb_vzi = get_compile_time_arg_val(10);
    constexpr auto cb_vzj = get_compile_time_arg_val(11);

    constexpr auto cb_eps = get_compile_time_arg_val(12);
    constexpr auto cb_pwj = get_compile_time_arg_val(13);
    constexpr auto cb_cns = get_compile_time_arg_val(14);

    constexpr auto cb_ax = get_compile_time_arg_val(15);
    constexpr auto cb_ay = get_compile_time_arg_val(16);
    constexpr auto cb_az = get_compile_time_arg_val(17);

    constexpr auto cb_adx = get_compile_time_arg_val(18);
    constexpr auto cb_ady = get_compile_time_arg_val(19);
    constexpr auto cb_adz = get_compile_time_arg_val(20);

    constexpr auto cb_axt = tt::CBIndex::c_21;
    constexpr auto cb_ayt = tt::CBIndex::c_22;
    constexpr auto cb_azt = tt::CBIndex::c_23;

    constexpr auto cb_adxt = tt::CBIndex::c_26;
    constexpr auto cb_adyt = tt::CBIndex::c_27;
    constexpr auto cb_adzt = tt::CBIndex::c_28;

    constexpr auto cb_tmp = tt::CBIndex::c_24;
    constexpr auto cb_tj = tt::CBIndex::c_25;

    constexpr auto cb_dx = tt::CBIndex::c_29;
    constexpr auto cb_dy = tt::CBIndex::c_30;
    constexpr auto cb_dz = tt::CBIndex::c_31;
					
    constexpr uint32_t dst_reg = 0;

    unary_op_init_common(cb_pxi, cb_ax);
    fill_tile_init();
    sub_binary_tile_init();
    add_binary_tile_init();
    square_tile_init();
    rsqrt_tile_init();
    mul_binary_tile_init();

    const uint32_t end_tile_id = start_tile_id + n_tiles;

    for (uint32_t i = start_tile_id; i < end_tile_id; i++) {
		
		cb_wait_front(cb_pxi, 1);
		cb_wait_front(cb_pyi, 1);
		cb_wait_front(cb_pzi, 1);

		cb_wait_front(cb_vxi, 1);
		cb_wait_front(cb_vyi, 1);
		cb_wait_front(cb_vzi, 1);

		cb_wait_front(cb_eps, 1);
		cb_wait_front(cb_cns, 1);
		
		cb_reserve_back(cb_axt, 1);
		cb_reserve_back(cb_ayt, 1);
		cb_reserve_back(cb_azt, 1);

		cb_reserve_back(cb_adxt, 1);
		cb_reserve_back(cb_adyt, 1);
		cb_reserve_back(cb_adzt, 1);
			
		cb_reserve_back(cb_ax, 1);
		cb_reserve_back(cb_ay, 1);
		cb_reserve_back(cb_az, 1);

		tile_regs_acquire();

		fill_tile(0, 0.0f);
		fill_tile(1, 0.0f);
		fill_tile(2, 0.0f);
		fill_tile(3, 0.0f);
		fill_tile(4, 0.0f);
		fill_tile(5, 0.0f);

		tile_regs_commit();
		tile_regs_wait();
		
		pack_tile(0, cb_axt);
		pack_tile(1, cb_ayt);
		pack_tile(2, cb_azt);
		pack_tile(3, cb_adxt);
		pack_tile(4, cb_adyt);
		pack_tile(5, cb_adzt);

		tile_regs_release();

		cb_push_back(cb_axt, 1);
		cb_push_back(cb_ayt, 1);
		cb_push_back(cb_azt, 1);

		cb_push_back(cb_adxt, 1);
		cb_push_back(cb_adyt, 1);
		cb_push_back(cb_adzt, 1);

		for (uint32_t j = 0; j < N; j++) {
			cb_wait_front(cb_pxj, 1);
			cb_wait_front(cb_pyj, 1);
			cb_wait_front(cb_pzj, 1);

			cb_wait_front(cb_vxj, 1);
			cb_wait_front(cb_vyj, 1);
			cb_wait_front(cb_vzj, 1);

			cb_wait_front(cb_pwj, 1);

			// ------------------------ dx, dy, dz

			cb_reserve_back(cb_dx, 1);
			cb_reserve_back(cb_dy, 1);
			cb_reserve_back(cb_dz, 1);

			tile_regs_acquire();

			
			copy_tile(cb_pxj, 0, 0);
			copy_tile(cb_pyj, 0, 1);
			copy_tile(cb_pzj, 0, 2);

			copy_tile(cb_pxi, 0, 3);
			copy_tile(cb_pyi, 0, 4);
			copy_tile(cb_pzi, 0, 5);
			
			sub_binary_tile(0, 3);
			sub_binary_tile(1, 4);
			sub_binary_tile(2, 5);

			tile_regs_commit();
			tile_regs_wait();

			pack_tile(0, cb_dx);
			pack_tile(1, cb_dy);
			pack_tile(2, cb_dz);

			tile_regs_release();

			cb_push_back(cb_dx, 1);
			cb_push_back(cb_dy, 1);
			cb_push_back(cb_dz, 1);

			// ---------------------- | tmp = -3.0 * invs2 | tj = pwj * invs * invs2 |

			cb_wait_front(cb_dx, 1);
			cb_wait_front(cb_dy, 1);
			cb_wait_front(cb_dz, 1);

			cb_reserve_back(cb_tmp, 1);
			cb_reserve_back(cb_tj, 1);

			tile_regs_acquire();

			copy_tile(cb_cns, 0, 0);
			copy_tile(cb_pwj, 0, 1);
			copy_tile(cb_dx, 0, 2);
			copy_tile(cb_dx, 0, 3);
			copy_tile(cb_dy, 0, 4);
			copy_tile(cb_dz, 0, 5);
			copy_tile(cb_eps, 0, 6);

			square_tile(2);
			square_tile(3);
			square_tile(4);
			square_tile(5);

			add_binary_tile(5, 6);
			add_binary_tile(4, 5);
			add_binary_tile(3, 4);
			add_binary_tile(2, 4);

			rsqrt_tile(3);
			rsqrt_tile(2);

			square_tile(3);

			mul_binary_tile(0, 3);
			mul_binary_tile(2, 3);
			mul_binary_tile(1, 2);

			tile_regs_commit();
			tile_regs_wait();

			pack_tile(0, cb_tmp);
			pack_tile(1, cb_tj);

			tile_regs_release();

			cb_push_back(cb_tmp, 1);
			cb_push_back(cb_tj, 1);

			// ------------------------ update ax, ay, ax

			cb_wait_front(cb_axt, 1);
			cb_wait_front(cb_ayt, 1);
			cb_wait_front(cb_azt, 1);

			cb_wait_front(cb_tj, 1);
			cb_wait_front(cb_tmp, 1);

			tile_regs_acquire();

			copy_tile(cb_dx, 0, 0);
			copy_tile(cb_dy, 0, 1);
			copy_tile(cb_dz, 0, 2);

			copy_tile(cb_tj, 0, 3);
			copy_tile(cb_tmp, 0, 4);

			mul_binary_tile(0, 3);
			mul_binary_tile(1, 3);
			mul_binary_tile(2, 3);

			copy_tile(cb_axt, 0, 4);
            		copy_tile(cb_ayt, 0, 5);
            		copy_tile(cb_azt, 0, 6);

	    		add_binary_tile(0, 4);
            		add_binary_tile(1, 5);
            		add_binary_tile(2, 6);

			cb_pop_front(cb_axt, 1);
			cb_pop_front(cb_ayt, 1);
			cb_pop_front(cb_azt, 1);

			tile_regs_commit();
			tile_regs_wait();

			cb_reserve_back(cb_axt, 1);
			cb_reserve_back(cb_ayt, 1);
			cb_reserve_back(cb_azt, 1);

			pack_tile(0, cb_axt);
			pack_tile(1, cb_ayt);
			pack_tile(2, cb_azt);

			tile_regs_release();

			cb_push_back(cb_axt, 1);
			cb_push_back(cb_ayt, 1);
			cb_push_back(cb_azt, 1);

			// ------------------------------------- h_adx
			//

			cb_wait_front(cb_adxt, 1);

			tile_regs_acquire();

			copy_tile(cb_tj, 0, 0);
			
			copy_tile(cb_vxj, 0, 1);
			copy_tile(cb_vxj, 0, 2);
			copy_tile(cb_vyj, 0, 3);
			copy_tile(cb_vzj, 0, 4);
			copy_tile(cb_vxi, 0, 5);
			copy_tile(cb_vyi, 0, 6);
			copy_tile(cb_vzi, 0, 7);

			sub_binary_tile(1, 5); //dvx
			sub_binary_tile(2, 5); //dvx
			sub_binary_tile(3, 6); //dvy
			sub_binary_tile(4, 7); //dvz

			copy_tile(cb_dx, 0, 5); //dx
			copy_tile(cb_dy, 0, 6); //dy
			copy_tile(cb_dz, 0, 7); //dz

			mul_binary_tile(2, 5);
			mul_binary_tile(3, 6);
			mul_binary_tile(4, 7);

			add_binary_tile(3, 4);
			add_binary_tile(2, 3); // vr
				
			copy_tile(cb_tmp, 0, 3);
			copy_tile(cb_dx, 0, 4);

			mul_binary_tile(3, 4); // dx * tmp 
			mul_binary_tile(2, 3); // dx * tmp * vr = dx * qdotq

			add_binary_tile(1, 2); // dvx + (dx * qdotq)

			mul_binary_tile(0, 1); // h_adx

			copy_tile(cb_adxt, 0, 2);
	    		add_binary_tile(0, 2);

			cb_pop_front(cb_adxt, 1);

			tile_regs_commit();
			tile_regs_wait();

			cb_reserve_back(cb_adxt, 1);

			pack_tile(0, cb_adxt);

			tile_regs_release();

			cb_push_back(cb_adxt, 1);

			/*
			if (i == start_tile_id && j == 0) {

				for (int32_t i = 0; i < 5; ++i) {
					SliceRange single_element_slice = {
						.h0 = 0,            // Start row 0
						.h1 = 1,            // End row 1 (exclusive, so only row 0)
						.hs = 1,            // Step 1 for rows
						.w0 = (uint8_t)i,   // Start column 'i' - Cast to uint8_t
						.w1 = (uint8_t)(i + 1),// End column 'i+1' (exclusive, so only column 'i') - Cast to uint8_t
						.ws = 1             // Step 1 for columns
					};

					DPRINT << "dx (0," << i << "): " << TSLICE(cb_dx, 0, single_element_slice) << ENDL();
				DPRINT << "dy (0," << i << "): " << TSLICE(cb_dy, 0, single_element_slice) << ENDL();
				DPRINT << "dz (0," << i << "): " << TSLICE(cb_dz, 0, single_element_slice) << ENDL();
				}
			}
			*/
			// ------------------------------------- h_adx (end)

			// ------------------------------------- h_ady

			cb_wait_front(cb_adyt, 1);

			tile_regs_acquire();

			copy_tile(cb_tj, 0, 0);
			copy_tile(cb_vyj, 0, 1);
			copy_tile(cb_vyj, 0, 2);
			copy_tile(cb_vxj, 0, 3);
			copy_tile(cb_vzj, 0, 4);
			copy_tile(cb_vyi, 0, 5);
			copy_tile(cb_vxi, 0, 6);
			copy_tile(cb_vzi, 0, 7);

			sub_binary_tile(1, 5); //dvy
			sub_binary_tile(2, 5); //dvy
			sub_binary_tile(3, 6); //dvx
			sub_binary_tile(4, 7); //dvz

			copy_tile(cb_dy, 0, 5); //dy
			copy_tile(cb_dx, 0, 6); //dx
			copy_tile(cb_dz, 0, 7); //dz

			mul_binary_tile(2, 5);
			mul_binary_tile(3, 6);
			mul_binary_tile(4, 7);

			add_binary_tile(3, 4);
			add_binary_tile(2, 3); // vr

			copy_tile(cb_tmp, 0, 3);
			copy_tile(cb_dy, 0, 4);

			mul_binary_tile(3, 4); // dy * tmp
			mul_binary_tile(2, 3); // dy * tmp * vr = dy * qdotq

			add_binary_tile(1, 2); // dvy + (dy * qdotq)

			mul_binary_tile(0, 1); // h_ady

			copy_tile(cb_adyt, 0, 2);
            		add_binary_tile(0, 2);

			cb_pop_front(cb_adyt, 1);

			tile_regs_commit();
			tile_regs_wait();

			cb_reserve_back(cb_adyt, 1);

			pack_tile(0, cb_adyt);

			tile_regs_release();

			cb_push_back(cb_adyt, 1);

			// ---------------------------------------- h_ady (end)

			// ------------------------------------- h_ady

			cb_wait_front(cb_adzt, 1);

			tile_regs_acquire();

			copy_tile(cb_tj, 0, 0);
			copy_tile(cb_vzj, 0, 1);
			copy_tile(cb_vzj, 0, 2);
			copy_tile(cb_vxj, 0, 3);
			copy_tile(cb_vyj, 0, 4);
			copy_tile(cb_vzi, 0, 5);
			copy_tile(cb_vxi, 0, 6);
			copy_tile(cb_vyi, 0, 7);

			sub_binary_tile(1, 5); //dvz
			sub_binary_tile(2, 5); //dvz
			sub_binary_tile(3, 6); //dvx
			sub_binary_tile(4, 7); //dvy

			copy_tile(cb_dz, 0, 5); //dz
			copy_tile(cb_dx, 0, 6); //dx
			copy_tile(cb_dy, 0, 7); //dy

			mul_binary_tile(2, 5);
			mul_binary_tile(3, 6);
			mul_binary_tile(4, 7);

			add_binary_tile(3, 4);
			add_binary_tile(2, 3); // vr

			copy_tile(cb_tmp, 0, 3);
			copy_tile(cb_dz, 0, 4);

			mul_binary_tile(3, 4); // dz * tmp
			mul_binary_tile(2, 3); // dz * tmp * vr = dz * qdotq

			add_binary_tile(1, 2); // dvz + (dz * qdotq)

			mul_binary_tile(0, 1); // h_adz

			copy_tile(cb_adzt, 0, 2);
            		add_binary_tile(0, 2);
			
			cb_pop_front(cb_adzt, 1);

			tile_regs_commit();
			tile_regs_wait();

			cb_reserve_back(cb_adzt, 1);

			pack_tile(0, cb_adzt);

			tile_regs_release();

			cb_push_back(cb_adzt, 1);

			// ---------------------------------------- h_adz (end)

			cb_pop_front(cb_dx, 1);
			cb_pop_front(cb_dy, 1);
			cb_pop_front(cb_dz, 1);

			cb_pop_front(cb_tmp, 1);
			cb_pop_front(cb_tj, 1);

			cb_pop_front(cb_pxj, 1);
			cb_pop_front(cb_pyj, 1);
			cb_pop_front(cb_pzj, 1);

			cb_pop_front(cb_vxj, 1);
			cb_pop_front(cb_vyj, 1);
			cb_pop_front(cb_vzj, 1);
			cb_pop_front(cb_pwj, 1); 
		}
		
		cb_wait_front(cb_axt, 1);
		cb_wait_front(cb_ayt, 1);
		cb_wait_front(cb_azt, 1);

		cb_wait_front(cb_adxt, 1);
		cb_wait_front(cb_adyt, 1);
		cb_wait_front(cb_adzt, 1);

		tile_regs_acquire();

		copy_tile(cb_axt, 0, 0);
		copy_tile(cb_ayt, 0, 1);
		copy_tile(cb_azt, 0, 2);

		copy_tile(cb_adxt, 0, 3);
		copy_tile(cb_adyt, 0, 4);
		copy_tile(cb_adzt, 0, 5);

		tile_regs_commit();
		tile_regs_wait();

		pack_tile(0, cb_ax);
		pack_tile(1, cb_ay);
		pack_tile(2, cb_az);

		pack_tile(3, cb_adx);
		pack_tile(4, cb_ady);
		pack_tile(5, cb_adz);
			
		tile_regs_release();

		cb_push_back(cb_ax, 1);
		cb_push_back(cb_ay, 1);
		cb_push_back(cb_az, 1);

		cb_push_back(cb_adx, 1);
		cb_push_back(cb_ady, 1);
		cb_push_back(cb_adz, 1);

		cb_pop_front(cb_axt, 1);
		cb_pop_front(cb_ayt, 1);
		cb_pop_front(cb_azt, 1);

		cb_pop_front(cb_adxt, 1);
		cb_pop_front(cb_adyt, 1);
		cb_pop_front(cb_adzt, 1);

		cb_pop_front(cb_pxi, 1);
		cb_pop_front(cb_pyi, 1);
		cb_pop_front(cb_pzi, 1);

		cb_pop_front(cb_vxi, 1);
		cb_pop_front(cb_vyi, 1);
		cb_pop_front(cb_vzi, 1);

		cb_pop_front(cb_eps, 1);
		cb_pop_front(cb_cns, 1);
    }
}
}  // namespace NAMESPACE


