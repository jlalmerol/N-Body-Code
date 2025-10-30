#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/device.hpp>

#include "hostdevcommon/profiler_common.h"
#include "tools/mem_bench/host_utils.hpp"
#include <tt-metalium/distributed.hpp>
#include "impl/context/metal_context.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <string_view>
#include <vector>
#include <algorithm>
#include <chrono>

namespace NBodyProject {

    using Program = tt::tt_metal::Program;
    using Buffer = tt::tt_metal::Buffer;
    using CoreCoord = tt::tt_metal::CoreCoord;
    using CoreRange = tt::tt_metal::CoreRange;
    using CoreRangeSet = tt::tt_metal::CoreRangeSet;

    using CBHandle = tt::tt_metal::CBHandle;
    using CBIndex = tt::CBIndex;  

    using CoreSpec = std::variant<CoreCoord, CoreRange, CoreRangeSet>;

} // namespace NBodyProject