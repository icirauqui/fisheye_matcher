cc_library(
    name = "qt_core",
    hdrs = glob(["QtCore/**"]),
    includes = ["."],
    linkopts = [
        "-lQt5Core",
    ],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "qt_widgets",
    hdrs = glob(["QtWidgets/**"]),
    includes = ["."],
    deps = [":qt_core"],
    linkopts = [
        "-lQt5Widgets",
    ],
    visibility = ["//visibility:public"],
)
cc_library(
    name = "qt_gui",
    hdrs = glob(["QtGui/**"]),
    includes = ["."],
    deps = [":qt_core"],
    linkopts = [
        "-lQt5Gui",
    ],
    visibility = ["//visibility:public"],
)