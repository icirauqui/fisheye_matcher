load("@rules_cc//cc:defs.bzl", "cc_library")


cc_library(
    name = "matcher",
    hdrs = ["matcher.hpp"],
    srcs = ["matcher.cpp"],
    deps = [
        "@opencv//:opencv",
        "//src/fe_lens:fe_lens",
    ],
    visibility = ["//visibility:public"],
)