# opencv.BUILD file

load("@rules_foreign_cc//foreign_cc:defs.bzl", "cmake")

filegroup(
    name = "srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
cmake(
    name = "opencv",
    generate_args = ["-GNinja"],
    additional_inputs=["@opencv_contrib//:modules"],
    cache_entries = {
        "INSTALL_C_EXAMPLES": "OFF",
        "INSTALL_PYTHON_EXAMPLES": "OFF",
        "OPENCV_GENERATE_PKGCONFIG": "ON",
        "OPENCV_ENALBLE_NONFREE": "ON",
        "BUILD_EXAMPLES": "OFF",
        #"WITH_VTK": "ON",
        "BUILD_SHARED_LIBS": "OFF",
        "BUILD_opencv_world": "ON",
        "OPENCV_EXTRA_MODULES_PATH": "$$EXT_BUILD_ROOT$$/external/opencv_contrib/modules",
        "WITH_QT": "OFF",
    },
    lib_source = ":srcs",
    #out_static_libs = ["libopencv_world.a"],
    out_include_dir = "include/opencv4",
    visibility = ["//visibility:public"],
)
