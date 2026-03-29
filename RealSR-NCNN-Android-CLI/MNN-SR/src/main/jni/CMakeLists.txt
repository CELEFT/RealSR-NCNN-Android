cmake_minimum_required(VERSION 3.5)
project(mnnsr-ncnn)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN")
#set(CMAKE_BUILD_TYPE Debug)
#set(CMAKE_BUILD_TYPE Release)

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../CMake/arch.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../CMake/libwebp.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../CMake/opencv.cmake)
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../CMake/mnn.cmake)

add_executable(${PROJECT_NAME} main.cpp mnnsr.cpp dcp.cpp)


if (MSVC)  # Visual Studio
    target_link_libraries(${PROJECT_NAME} webp ${OpenCV_LIBS} ${MNN_LIB})
    
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        # 1. 先手动创建目标文件夹
        COMMAND ${CMAKE_COMMAND} -E make_directory ${ASSETS_DIR}
        
        # DLL 路径
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${mnn_DIR}/lib/${TARGET_ARCH}/Release/Dynamic/MT/MNN.dll"
        ${ASSETS_DIR}/

        COMMAND_EXPAND_LISTS
    )

elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(${PROJECT_NAME} webp ${OpenCV_LIBS} ${MNN_LIB})
    set_target_properties(${PROJECT_NAME} PROPERTIES
        BUILD_RPATH "$ORIGIN/lib"
    )

else ()
    # 修改：只链接合并后的 MNN 库，不再单独链接 MNN_CL 和 MNN_Vulkan
    target_link_libraries(${PROJECT_NAME} webp MNN ${OpenCV_LIBS})
endif ()

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../CMake/assets.cmake)
