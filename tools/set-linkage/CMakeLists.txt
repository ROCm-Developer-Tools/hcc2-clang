
set(LLVM_LINK_COMPONENTS support BitWriter Core Object )
add_clang_executable(set-linkage set-linkage.cpp)
install(TARGETS set-linkage RUNTIME DESTINATION bin)
