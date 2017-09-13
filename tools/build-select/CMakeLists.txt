
set(LLVM_LINK_COMPONENTS support BitWriter Core Object )
add_clang_executable(build-select build-select.cpp)
install(TARGETS build-select RUNTIME DESTINATION bin)
