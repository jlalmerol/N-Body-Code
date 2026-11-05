#pragma once

#include <atomic>
#include <cstdlib>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <nlohmann/json.hpp>
#include <tracy/Tracy.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_metal.hpp>

namespace nbody_profiling {

inline void flush_device(tt::tt_metal::distributed::MeshDevice& device) {
    const char* enabled = std::getenv("TT_METAL_DEVICE_PROFILER");
    if (enabled && std::string(enabled) == "1") {
        tt::tt_metal::distributed::Finish(device.mesh_command_queue());
        tt::tt_metal::ReadMeshDeviceProfilerResults(device);
    }
}

// Match TTNN's native-operation protocol without depending on TTNN. Each
// persistent program needs a fresh ID before enqueue, including cache hits.
inline void record_workload(
    tt::tt_metal::distributed::MeshDevice& device,
    tt::tt_metal::distributed::MeshWorkload& workload,
    const char* name, int particles, int cores,
    std::initializer_list<std::size_t> input_elements,
    std::initializer_list<std::size_t> output_elements) {
    static std::atomic<uint32_t> next_id{1};
    for (auto& [range, program] : workload.get_programs()) {
        const auto runtime_id = next_id.fetch_add(1);
        // The SDK packs a 21-bit base ID and a 10-bit device ID into 31 bits.
        if (runtime_id >= (1u << 21)) {
            throw std::overflow_error("N-body profiler operation IDs exhausted");
        }
        program.set_runtime_id(runtime_id);
#ifdef TRACY_ENABLE
        const char* enabled = std::getenv("TTNN_OP_PROFILER");
        if (!enabled || std::string(enabled) != "1") continue;
        for (auto coord : range) {
            const auto devices = device.get_view().get_devices(
                tt::tt_metal::distributed::MeshCoordinateRange(coord, coord));
            if (devices.empty()) continue;
            const auto device_id = devices.front()->id();
            const auto op_id = tt::tt_metal::detail::EncodePerDeviceProgramID(
                runtime_id, device_id, false);
            // These are the actual packed Float32 buffers, not logical N x N
            // tensors. No matmul FLOP model is appropriate for the SFPU kernel.
            auto buffers = [device_id](auto sizes) {
                auto result = nlohmann::json::array();
                for (auto elements : sizes) {
                    result.push_back({
                        {"shape", {{"W", 1}, {"Z", 1}, {"Y", elements / 32}, {"X", 32}}},
                        {"dtype", "FLOAT32"}, {"layout", "TILE"},
                        {"storage_type", "DEV_" + std::to_string(device_id) + "_DRAM_INTERLEAVED"}});
                }
                return result;
            };
            const nlohmann::json metadata = {
                {"global_call_count", op_id}, {"device_id", device_id},
                {"op_code", name}, {"op_type", "tt_dnn_device"},
                {"attributes", {{"particles", particles}, {"buffer_layout", "packed_float32_tiles"}}},
                {"math_fidelity", "HiFi4"}, {"core_usage", {{"count", cores}}},
                {"input_tensors", buffers(input_elements)},
                {"output_tensors", buffers(output_elements)}};
            ZoneScopedN("TT_METAL_DEVICE_OP");
            const auto text = "id:" + std::to_string(op_id);
            ZoneText(text.data(), text.size());
            const auto message = "`TT_METAL_DEVICE_OP ->\n" + metadata.dump() + "`";
            TracyMessage(message.data(), message.size());
        }
#endif
    }
}

}  // namespace nbody_profiling
