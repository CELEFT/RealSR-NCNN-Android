if (MSVC)  # Visual Studio

    # 设置目标属性，指定 MNN 库的路径
    set(mnn_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../3rdparty/mnn_windows_x64_cpu_opencl)
    # set(mnn_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../3rdparty/mnn_windows)
    find_library(MNN_LIB mnn HINTS "${mnn_DIR}/lib/${TARGET_ARCH}/Release/Dynamic/MT")

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(mnn_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../3rdparty/MNN)
    find_library(MNN_LIB MNN HINTS "${mnn_DIR}/libs")

    # set(mnn_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../3rdparty/mnn_linux_x64)
    # find_library(MNN_LIB MNN HINTS "${mnn_DIR}/lib/Release")
    # include_directories(${mnn_DIR}/include)
    message(STATUS "find mnn dir: ${mnn_DIR}")
    message(STATUS "find mnn: ${MNN_LIB}")
    set(ncnn_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../3rdparty/ncnn-ubuntu-shared)
else()

    # 设置目标属性，指定 MNN 库的路径（合并后的单一库）
    set(mnn_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../3rdparty/mnn_android)

    include_directories(${mnn_DIR}/include)

    # 导入合并后的 MNN 主库（包含 CPU、OpenCL、Vulkan 等所有后端）
    add_library(MNN SHARED IMPORTED)
    set_target_properties(MNN PROPERTIES IMPORTED_LOCATION
            ${mnn_DIR}/${ANDROID_ABI}/libMNN.so)

    # 注意：c++_shared 不再需要单独导入，因为 MNN 已静态链接或由系统提供
endif ()

include_directories(${mnn_DIR}/include)

message(STATUS "Find mnn in: ${mnn_DIR}")
if (EXISTS ${MNN_LIB})
    #message(STATUS "find mnn: ${MNN_LIB}")
else ()
    message(STATUS "    mnn library not found!")
    # 设置默认路径为合并后的库
    set(MNN_LIB "${mnn_DIR}/${ANDROID_ABI}/libMNN.so")
endif ()
message(STATUS "Found mnn: ${MNN_LIB}")