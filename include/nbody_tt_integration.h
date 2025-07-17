#pragma once

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/device_impl.hpp>
#include <tt-metalium/work_split.hpp>

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
