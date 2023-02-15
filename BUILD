load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")
load("@aabtop_rules_qt//:qt_rules.bzl", "qt_cc_library", "qt_cc_binary", "qt_resource")


cc_library(
    name = "aux",
    hdrs = ["cc/aux.h"],
    srcs = ["cc/aux.cpp"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "json",
    hdrs = ["third_party/nlohmann/json.hpp"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "camera",
    hdrs = ["cc/camera.h"],
    srcs = ["cc/camera.cpp"],
    deps = [
        "@opencv//:opencv",
        ":aux",
        ":json",
    ],
    linkstatic=True,
    strip_include_prefix = "cc",
    visibility = ["//visibility:public"],
)

cc_library(
    name = "ang_matcher",
    hdrs = ["cc/ang_matcher.h"],
    srcs = ["cc/ang_matcher.cpp"],
    deps = [
        "@opencv//:opencv",
        ":aux",
    ],
    linkstatic=True,
    strip_include_prefix = "cc",
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "main",
    srcs = ["cc/main.cpp"],
    deps = [
        ":aux",
        ":json",
        ":camera",
        ":ang_matcher",
        "@opencv//:opencv",
    ],
    includes = ["cc"],
    data = glob(["images/**"]),
    #linkstatic=True,
    visibility = ["//visibility:public"],
)



qt_cc_binary(
  name = "mainqt",
  srcs = [
    "cc/qt/mainqt.cc",
  ],
  deps = [
    ":main_window",
  ],
)

qt_cc_library(
  name = "main_window",
  srcs = [
    "cc/qt/main_window.cc",
  ],
  hdr = "cc/qt/main_window.h",
  ui_src = "cc/qt/main_window.ui",
)