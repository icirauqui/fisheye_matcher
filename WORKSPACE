workspace(name = "main")

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Setup rules_foreign_cc (CMake integration)
http_archive(
    name = "rules_foreign_cc",
    strip_prefix = "rules_foreign_cc-8d540605805fb69e24c6bf5dc885b0403d74746a", # 0.9.0
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/8d540605805fb69e24c6bf5dc885b0403d74746a.tar.gz",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()






_ALL_CONTENT = """\
filegroup(
    name = "all_srcs",
    srcs = glob(["**"]),
    visibility = ["//visibility:public"],
)
"""

http_archive(
      name = "opencv",
      #build_file_content = _ALL_CONTENT,
      build_file = "//:third_party/opencv.BUILD",
      sha256 = "8df0079cdbe179748a18d44731af62a245a45ebf5085223dc03133954c662973",
      strip_prefix = "opencv-4.7.0",
      urls = ["https://github.com/opencv/opencv/archive/refs/tags/4.7.0.tar.gz"],
)

http_archive(
      name = "opencv_contrib",
      #build_file_content = _ALL_CONTENT,
      build_file = "//:third_party/opencv_contrib.BUILD",
      sha256 = "42df840cf9055e59d0e22c249cfb19f04743e1bdad113d31b1573d3934d62584",
      strip_prefix = "opencv_contrib-4.7.0",
      urls = ["https://github.com/opencv/opencv_contrib/archive/refs/tags/4.7.0.tar.gz"],
)


http_archive(
    name = "aabtop_rules_qt",
    strip_prefix = "rules_qt-4703da94a8a996e9372e6ec3d33bb082a2882e8d",
    url = "https://github.com/aabtop/rules_qt/archive/4703da94a8a996e9372e6ec3d33bb082a2882e8d.zip",
    sha256 = "ba7912fe87a6a389bb83f83baa1d89d9f899abf1739b3aaf972169d934da6c9b",
)


load("@aabtop_rules_qt//:rules_qt_deps1.bzl", "rules_qt_deps1")
rules_qt_deps1()
load("@aabtop_rules_qt//:rules_qt_deps2.bzl", "rules_qt_deps2")
rules_qt_deps2()