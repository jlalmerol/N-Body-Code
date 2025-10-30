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
#include "compute_kernel_api/eltwise_unary/rsqrt.h"

#include "debug/dprint.h"
#include "tools/profiler/kernel_profiler.hpp"
#include "llk_math_eltwise_ternary_sfpu_params.h"

#ifdef TRISC_MATH
void qdotq_part_face_unary() {
	for (size_t i = 0; i < 8; i++) {
		vFloat a = -3.0f * dst_reg[i]; 
		dst_reg[i] = a;
	}
}
#endif // TRISC_MATH

void qdotq_part_tile(const uint32_t dst_index_out) {
	MATH(_llk_math_eltwise_unary_sfpu_params_<false>(
			qdotq_part_face_unary, dst_index_out, VectorMode::RC));
}

#ifdef TRISC_MATH
constexpr float DIST_EPS = 1.0e-7f;

void squared_dist_tile_face(
	const uint32_t dst_index_in0,  // dx
	const uint32_t dst_index_in1,  // dy  
	const uint32_t dst_index_in2,  // dz
	const uint32_t dst_index_out)  // output
{
	constexpr uint32_t n_vector_in_tile = 32;

	const uint32_t dx_base_idx = dst_index_in0 * n_vector_in_tile;
	const uint32_t dy_base_idx = dst_index_in1 * n_vector_in_tile;
	const uint32_t dz_base_idx = dst_index_in2 * n_vector_in_tile;
	const uint32_t out_base_idx = dst_index_out * n_vector_in_tile;

	for (size_t i = 0; i < 8; i++) {
		vFloat dx = dst_reg[dx_base_idx + i];
		vFloat dy = dst_reg[dy_base_idx + i];
		vFloat dz = dst_reg[dz_base_idx + i];
		
		vFloat dx_sq = dx * dx;
		vFloat dy_sq = dy * dy; 
		vFloat dz_sq = dz * dz;
		
		dst_reg[out_base_idx + i] = dx_sq + dy_sq + dz_sq + DIST_EPS;
	}
}
#endif // TRISC_MATH

void squared_dist_tile(
	const uint32_t dst_index_dx,
	const uint32_t dst_index_dy, 
	const uint32_t dst_index_dz,
	const uint32_t dst_index_out) 
{
	MATH((_llk_math_eltwise_ternary_sfpu_params_<false>(
			squared_dist_tile_face, 
			dst_index_dx, dst_index_dy, dst_index_dz, dst_index_out)));
}

#ifdef TRISC_MATH
void add_ternary_tile_face(
	const uint32_t dst_index_in0,  // dx
	const uint32_t dst_index_in1,  // dy  
	const uint32_t dst_index_in2,  // dz
	const uint32_t dst_index_out)  // output
{
	constexpr uint32_t n_vector_in_tile = 32;

	const uint32_t dx_base_idx = dst_index_in0 * n_vector_in_tile;
	const uint32_t dy_base_idx = dst_index_in1 * n_vector_in_tile;
	const uint32_t dz_base_idx = dst_index_in2 * n_vector_in_tile;
	const uint32_t out_base_idx = dst_index_out * n_vector_in_tile;

	for (size_t i = 0; i < 8; i++) {
		vFloat dx = dst_reg[dx_base_idx + i];
		vFloat dy = dst_reg[dy_base_idx + i];
		vFloat dz = dst_reg[dz_base_idx + i];

		dst_reg[out_base_idx + i] = dx + dy + dz;
	}
}
#endif // TRISC_MATH

void add_ternary_tile(
	const uint32_t dst_index_dx,
	const uint32_t dst_index_dy, 
	const uint32_t dst_index_dz,
	const uint32_t dst_index_out) 
{
	MATH((_llk_math_eltwise_ternary_sfpu_params_<false>(
			add_ternary_tile_face, 
			dst_index_dx, dst_index_dy, dst_index_dz, dst_index_out)));
}

#ifdef TRISC_MATH
// ax = axt + dx * tj
void update_acc_tile_face(
	const uint32_t dst_index_in0,  // axt
	const uint32_t dst_index_in1,  // dx 
	const uint32_t dst_index_in2,  // tj
	const uint32_t dst_index_out)  // output
{
	constexpr uint32_t n_vector_in_tile = 32;

	const uint32_t dx_base_idx = dst_index_in0 * n_vector_in_tile;
	const uint32_t dy_base_idx = dst_index_in1 * n_vector_in_tile;
	const uint32_t dz_base_idx = dst_index_in2 * n_vector_in_tile;
	const uint32_t out_base_idx = dst_index_out * n_vector_in_tile;

	for (size_t i = 0; i < 8; i++) {
		vFloat axt = dst_reg[dx_base_idx + i];
		vFloat dx = dst_reg[dy_base_idx + i];
		vFloat tj = dst_reg[dz_base_idx + i];
		
		dst_reg[out_base_idx + i] = axt + (dx * tj);
	}
}
#endif // TRISC_MATH

void update_acc_tile(
	const uint32_t dst_index_dx,
	const uint32_t dst_index_dy, 
	const uint32_t dst_index_dz,
	const uint32_t dst_index_out) 
{
	MATH((_llk_math_eltwise_ternary_sfpu_params_<false>(
			update_acc_tile_face, 
			dst_index_dx, dst_index_dy, dst_index_dz, dst_index_out)));
}

#ifdef TRISC_MATH
// A = dvx + (dx * qdotq)
void muladd_tile_face(
	const uint32_t dst_index_in0,  // dvx
	const uint32_t dst_index_in1,  // dx 
	const uint32_t dst_index_in2,  // qdotq
	const uint32_t dst_index_out)  // A
{
	constexpr uint32_t n_vector_in_tile = 32;

	const uint32_t dvx_base_idx = dst_index_in0 * n_vector_in_tile;
	const uint32_t dx_base_idx = dst_index_in1 * n_vector_in_tile;
	const uint32_t qdotq_base_idx = dst_index_in2 * n_vector_in_tile;
	const uint32_t out_base_idx = dst_index_out * n_vector_in_tile;

	for (size_t i = 0; i < 8; i++) {
		vFloat dvx = dst_reg[dvx_base_idx + i];
		vFloat dx = dst_reg[dx_base_idx + i];
		vFloat qdotq = dst_reg[qdotq_base_idx + i];

		dst_reg[out_base_idx + i] = dvx + (qdotq * dx); 
	}
}
#endif // TRISC_MATH

void muladd_tile(
	const uint32_t dst_index_in0,  // dvx
	const uint32_t dst_index_in1,  // dx 
	const uint32_t dst_index_in2,  // qdotq
	const uint32_t dst_index_out)  // A
{
	MATH((_llk_math_eltwise_ternary_sfpu_params_<false>(
			muladd_tile_face, 
			dst_index_in0, dst_index_in1, dst_index_in2, dst_index_out)));
}

namespace NAMESPACE {
void MAIN {
	uint32_t n_tiles = get_arg_val<uint32_t>(0);
	uint32_t N = get_arg_val<uint32_t>(1);

	constexpr auto cb_pxi = tt::CBIndex::c_0;
	constexpr auto cb_pxj = tt::CBIndex::c_1;
	constexpr auto cb_pyi = tt::CBIndex::c_2;
	constexpr auto cb_pyj = tt::CBIndex::c_3;
	constexpr auto cb_pzi = tt::CBIndex::c_4;
	constexpr auto cb_pzj = tt::CBIndex::c_5;

	constexpr auto cb_vxi = tt::CBIndex::c_6;
	constexpr auto cb_vxj = tt::CBIndex::c_7;
	constexpr auto cb_vyi = tt::CBIndex::c_8;
	constexpr auto cb_vyj = tt::CBIndex::c_9;
	constexpr auto cb_vzi = tt::CBIndex::c_10;
	constexpr auto cb_vzj = tt::CBIndex::c_11;

	constexpr auto cb_pwj = tt::CBIndex::c_12;

	constexpr auto cb_ax = tt::CBIndex::c_13;
	constexpr auto cb_ay = tt::CBIndex::c_14;
	constexpr auto cb_az = tt::CBIndex::c_15;

	constexpr auto cb_adx = tt::CBIndex::c_16;
	constexpr auto cb_ady = tt::CBIndex::c_17;
	constexpr auto cb_adz = tt::CBIndex::c_18;

	constexpr auto cb_axt = tt::CBIndex::c_19;
	constexpr auto cb_ayt = tt::CBIndex::c_20;
	constexpr auto cb_azt = tt::CBIndex::c_21;

	constexpr auto cb_adxt = tt::CBIndex::c_24;
	constexpr auto cb_adyt = tt::CBIndex::c_25;
	constexpr auto cb_adzt = tt::CBIndex::c_26;

	constexpr auto cb_tmp = tt::CBIndex::c_22;
	constexpr auto cb_tj = tt::CBIndex::c_23;

	constexpr auto cb_dx = tt::CBIndex::c_27;
	constexpr auto cb_dy = tt::CBIndex::c_28;
	constexpr auto cb_dz = tt::CBIndex::c_29;
		
	constexpr uint32_t dst_reg = 0;

	unary_op_init_common(cb_pxi, cb_ax);
	fill_tile_init();
	sub_binary_tile_init();
	add_binary_tile_init();
	square_tile_init();
	rsqrt_tile_init();
	mul_binary_tile_init();

	for (uint32_t i = 0; i < n_tiles; i++) {
		DeviceZoneScopedN("NBCOMPUTE");
		DeviceTimestampedData("computetime", i + ((uint64_t)1 << 32));
		DeviceRecordEvent(i);

		cb_wait_front(cb_pxi, 1);
		cb_wait_front(cb_pyi, 1);
		cb_wait_front(cb_pzi, 1);

		cb_wait_front(cb_vxi, 1);
		cb_wait_front(cb_vyi, 1);
		cb_wait_front(cb_vzi, 1);

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

		for (uint32_t j = 0; j < N; j++) 
			{
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

			sub_binary_tile(0, 3, 0);
			sub_binary_tile(1, 4, 1);
			sub_binary_tile(2, 5, 2);

			tile_regs_commit();
			tile_regs_wait();

			pack_tile(0, cb_dx);
			pack_tile(1, cb_dy);
			pack_tile(2, cb_dz);

			tile_regs_release();

			cb_push_back(cb_dx, 1);
			cb_push_back(cb_dy, 1);
			cb_push_back(cb_dz, 1);

			// ---------------------- | (tmp) qdotq = -3.0 * invs2 * vr | tj = pwj * invs * invs2 |
			cb_wait_front(cb_dx, 1);
			cb_wait_front(cb_dy, 1);
			cb_wait_front(cb_dz, 1);

			cb_reserve_back(cb_tmp, 1);
			cb_reserve_back(cb_tj, 1);

			tile_regs_acquire();

			copy_tile(cb_pwj, 0, 5);
			copy_tile(cb_dx, 0, 2);
			copy_tile(cb_dy, 0, 3);
			copy_tile(cb_dz, 0, 4);

			squared_dist_tile(2, 3, 4, 0);
			squared_dist_tile(2, 3, 4, 1); 

			rsqrt_tile(0); // invs
			rsqrt_tile(1); // invs

			square_tile(1); // invs2

			mul_binary_tile(0, 1, 0); // q = invs * invs2
			qdotq_part_tile(1); // tmp = -3.0 * invs2
			mul_binary_tile(0, 5, 0); // tj = pwj * q

			// qdotq
			copy_tile(cb_vxj, 0, 2);
			copy_tile(cb_vyj, 0, 3);
			copy_tile(cb_vzj, 0, 4);
			copy_tile(cb_vxi, 0, 5);
			copy_tile(cb_vyi, 0, 6);
			copy_tile(cb_vzi, 0, 7);

			sub_binary_tile(2, 5, 2); //dvx
			sub_binary_tile(3, 6, 3); //dvy
			sub_binary_tile(4, 7, 4); //dvz

			copy_tile(cb_dx, 0, 5);
			copy_tile(cb_dy, 0, 6);
			copy_tile(cb_dz, 0, 7);

			mul_binary_tile(2, 5, 2); // dvx * dx
			mul_binary_tile(3, 6, 3);
			mul_binary_tile(4, 7, 4);

			add_ternary_tile(2, 3, 4, 2); // vr

			mul_binary_tile(1, 2, 1); // qdotq = tmp * vr

			tile_regs_commit();
			tile_regs_wait();

			pack_tile(1, cb_tmp);
			pack_tile(0, cb_tj);

			tile_regs_release();

			cb_push_back(cb_tmp, 1);
			cb_push_back(cb_tj, 1);

			// ------------------------ update ax, ay, ax

			cb_wait_front(cb_axt, 1);
			cb_wait_front(cb_ayt, 1);
			cb_wait_front(cb_azt, 1);

			cb_wait_front(cb_tj, 1);

			tile_regs_acquire();

			copy_tile(cb_dx, 0, 0);
			copy_tile(cb_dy, 0, 1);
			copy_tile(cb_dz, 0, 2);

			copy_tile(cb_tj, 0, 3);

			copy_tile(cb_axt, 0, 4);
			copy_tile(cb_ayt, 0, 5);
			copy_tile(cb_azt, 0, 6);

			update_acc_tile(4, 0, 3, 0);
			update_acc_tile(5, 1, 3, 1);
			update_acc_tile(6, 2, 3, 2);

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

			// ------------------------------------- h_adx, h_ady, h_adz

			cb_wait_front(cb_adxt, 1);
			cb_wait_front(cb_adyt, 1);
			cb_wait_front(cb_adzt, 1);

			tile_regs_acquire();

			copy_tile(cb_tj, 0, 6);
			copy_tile(cb_tmp, 0, 7); // qdotq

			copy_tile(cb_vxj, 0, 0);
			copy_tile(cb_vxi, 0, 1);
			sub_binary_tile(0, 1, 0); //dvx

			copy_tile(cb_vyj, 0, 1);
			copy_tile(cb_vyi, 0, 2);
			sub_binary_tile(1, 2, 1); //dvy

			copy_tile(cb_vzj, 0, 2);
			copy_tile(cb_vzi, 0, 3);
			sub_binary_tile(2, 3, 2); //dvz

			copy_tile(cb_dx, 0, 3); //dx
			copy_tile(cb_dy, 0, 4); //dy
			copy_tile(cb_dz, 0, 5); //dz

			muladd_tile(0, 3, 7, 0); // dvx + (dx * qdotq)
			muladd_tile(1, 4, 7, 1); // dvy + (dy * qdotq)
			muladd_tile(2, 5, 7, 2); // dvz + (dz * qdotq)

			mul_binary_tile(0, 6, 0); // h_adx
			mul_binary_tile(1, 6, 1); // h_ady
			mul_binary_tile(2, 6, 2); // h_adz

			copy_tile(cb_adxt, 0, 3);
			copy_tile(cb_adyt, 0, 4);
			copy_tile(cb_adzt, 0, 5);

			add_binary_tile(0, 3, 0);
			add_binary_tile(1, 4, 1);
			add_binary_tile(2, 5, 2);

			cb_pop_front(cb_adxt, 1);
			cb_pop_front(cb_adyt, 1);
			cb_pop_front(cb_adzt, 1);

			tile_regs_commit();
			tile_regs_wait();

			cb_reserve_back(cb_adxt, 1);
			cb_reserve_back(cb_adyt, 1);
			cb_reserve_back(cb_adzt, 1);

			pack_tile(0, cb_adxt);
			pack_tile(1, cb_adyt);
			pack_tile(2, cb_adzt);

			tile_regs_release();

			cb_push_back(cb_adxt, 1);
			cb_push_back(cb_adyt, 1);
			cb_push_back(cb_adzt, 1);
			// ------------------------------------- h_adx, h_ady, h_adz

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

    }
	}
}  // namespace NAMESPACE