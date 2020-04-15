if(WIN32)
	set(EXE_NAME "zed_openpose_pcl")
endif ()

SET(EXECUTABLE_OUTPUT_PATH ".")

find_package(ZED 2 REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)

include(${CMAKE_SOURCE_DIR}/cmake/Utils.cmake)
include_directories(${ZED_INCLUDE_DIRS})
include_directories(${GLEW_INCLUDE_DIRS})
include_directories(${GLUT_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIRS})

link_directories(${ZED_LIBRARY_DIR})
link_directories(${GLEW_LIBRARY_DIRS})
link_directories(${GLUT_LIBRARY_DIRS})
link_directories(${OpenGL_LIBRARY_DIRS})


FILE(GLOB_RECURSE SRC_FILES *.cpp)
FILE(GLOB_RECURSE HDR_FILES *.hpp)
message(STATUS "Adding Example ${EXE_NAME}")
add_executable(${EXE_NAME} ${HDR_FILES} ${SRC_FILES})



target_link_libraries(
				${EXE_NAME} openpose 
				${examples_3rdparty_libraries}
				${ZED_LIBRARIES}
				${GLUT_LIBRARY}
				${GLEW_LIBRARY}
				${SPECIAL_OS_LIBS}
				${OPENGL_LIBRARIES}
				${CUDA_CUDA_LIBRARY}
				${CUDA_npps_LIBRARY}
				${CUDA_CUDART_LIBRARY}
				${CUDA_NPP_LIBRARIES_ZED} )


if (WIN32)
	set_property(TARGET ${EXE_NAME} PROPERTY FOLDER "Examples")
	configure_file(${CMAKE_SOURCE_DIR}/cmake/OpenPose${VCXPROJ_FILE_GPU_MODE}.vcxproj.user
        ${CMAKE_CURRENT_BINARY_DIR}/${EXE_NAME}.vcxproj.user @ONLY)
# Properties->General->Output Directory
	set_property(TARGET ${EXE_NAME} PROPERTY RUNTIME_OUTPUT_DIRECTORY_RELEASE ${PROJECT_BINARY_DIR}/$(Platform)/$(Configuration))
	set_property(TARGET ${EXE_NAME} PROPERTY RUNTIME_OUTPUT_DIRECTORY_DEBUG ${PROJECT_BINARY_DIR}/$(Platform)/$(Configuration))
endif (WIN32)

