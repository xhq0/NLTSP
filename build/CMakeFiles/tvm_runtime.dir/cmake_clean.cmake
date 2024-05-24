file(REMOVE_RECURSE
  "libtvm_runtime.pdb"
  "libtvm_runtime.so"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/tvm_runtime.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
