load("@rules_cc//cc:defs.bzl", "cc_binary", "cc_library")


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

cc_library(
    name = "feature_matcher",
    hdrs = ["cc/feature_matcher.h"],
    srcs = ["cc/feature_matcher.cpp"],
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
        ":feature_matcher",
        "@opencv//:opencv",
    ],
    includes = ["cc"],
    linkstatic=True,
    visibility = ["//visibility:public"],
)




exports_files(
    [
        "bazel-out/k8-fastbuild/bin/external/oneDNN/oneDNN/lib/libdnnl.so",
        "bazel-out/k8-fastbuild/bin/external/oneDNN/oneDNN/lib/libdnnl.so.2",
    ]
)