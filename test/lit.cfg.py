# -*- Python -*-

import os
import platform
import re
import subprocess
import tempfile

import lit.formats
import lit.util

from lit.llvm import llvm_config
from lit.llvm.subst import ToolSubst
from lit.llvm.subst import FindTool

# Configuration file for the 'lit' test runner.

# name: The name of this test suite.
config.name = 'POTATO'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = ['.mlir', '.c', '.cpp', '.ll']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.potato_obj_root, 'test')

config.substitutions.append(('%PATH%', config.environment['PATH']))
config.substitutions.append(('%shlibext', config.llvm_shlib_ext))

llvm_config.with_system_environment(
    ['HOME', 'INCLUDE', 'LIB', 'TMP', 'TEMP'])

# excludes: A list of directories to exclude from the testsuite. The 'Inputs'
# subdirectories contain auxiliary inputs for various tests in their parent
# directories.
config.excludes = ['Inputs', 'Examples', 'CMakeLists.txt', 'README.txt', 'LICENSE.txt']

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.potato_obj_root, 'test')
config.potato_test_util = os.path.join(config.potato_src_root, 'test/utils')
config.potato_bin_dir = os.path.join(config.potato_obj_root, 'bin')
mlir_translate = os.path.join(config.llvm_tools_dir, 'mlir-translate')
tools = [
    ToolSubst('%potato-opt', command = 'potato-opt'),
    ToolSubst('%emit-llvm', command = config.host_cc +' -S -emit-llvm'),
    ToolSubst('%llvm-to-mlir', command = mlir_translate + ' -import-llvm'),
    ToolSubst('%file-check', command = 'FileCheck'),
    ToolSubst('%cc', command = config.host_cc)
]

if 'BUILD_TYPE' in lit_config.params:
    config.potato_build_type = lit_config.params['BUILD_TYPE']
else:
    config.potato_build_type = "Debug"

for tool in tools:
    if tool.command.startswith('potato'):
        path = [config.potato_bin_dir, tool.command]
        tool.command = os.path.join(*path)
    llvm_config.add_tool_substitutions([tool])

if config.host_cc.find('clang') != -1:
    config.available_features.add("clang")
