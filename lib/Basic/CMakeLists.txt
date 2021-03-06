set(LLVM_LINK_COMPONENTS
  Core
  MC
  Support
  )

# Figure out if we can track VC revisions.
function(find_first_existing_file out_var)
  foreach(file ${ARGN})
    if(EXISTS "${file}")
      set(${out_var} "${file}" PARENT_SCOPE)
      return()
    endif()
  endforeach()
endfunction()

macro(find_first_existing_vc_file out_var path)
  #set(git_path "${path}/.git")
  set(git_path "${path}/../../.git") ## workaround for HCC

  # Normally '.git' is a directory that contains a 'logs/HEAD' file that
  # is updated as modifications are made to the repository. In case the
  # repository is a Git submodule, '.git' is a file that contains text that
  # indicates where the repository's Git directory exists.
  if (EXISTS "${git_path}" AND NOT IS_DIRECTORY "${git_path}")
    FILE(READ "${git_path}" file_contents)
    if("${file_contents}" MATCHES "^gitdir: ([^\n]+)")
      # '.git' is indeed a link to the submodule's Git directory.
      # Use the path to that Git directory.
      set(git_path "${path}/${CMAKE_MATCH_1}")
    endif()
  endif()

  find_first_existing_file(${out_var}
    "${git_path}/logs/HEAD"  # Git or Git submodule
    "${path}/.svn/wc.db"     # SVN 1.7
    "${path}/.svn/entries"   # SVN 1.6
    )
endmacro()

find_first_existing_vc_file(llvm_vc "${LLVM_MAIN_SRC_DIR}")
find_first_existing_vc_file(clang_vc "${CLANG_SOURCE_DIR}")

# The VC revision include that we want to generate.
set(version_inc "${CMAKE_CURRENT_BINARY_DIR}/SVNVersion.inc")

set(get_svn_script "${LLVM_CMAKE_PATH}/GetSVN.cmake")

if(DEFINED llvm_vc AND DEFINED clang_vc)
  # Create custom target to generate the VC revision include.
  add_custom_command(OUTPUT "${version_inc}"
    DEPENDS "${llvm_vc}" "${clang_vc}" "${get_svn_script}"
    COMMAND
    ${CMAKE_COMMAND} "-DFIRST_SOURCE_DIR=${LLVM_MAIN_SRC_DIR}"
                     "-DFIRST_NAME=LLVM"
                     "-DSECOND_SOURCE_DIR=${CLANG_SOURCE_DIR}"
                     "-DSECOND_NAME=SVN"
                     "-DHEADER_FILE=${version_inc}"
                     -P "${get_svn_script}")

  # Mark the generated header as being generated.
  set_source_files_properties("${version_inc}"
    PROPERTIES GENERATED TRUE
               HEADER_FILE_ONLY TRUE)

  # Tell Version.cpp that it needs to build with -DHAVE_SVN_VERSION_INC.
  set_source_files_properties(Version.cpp
    PROPERTIES COMPILE_DEFINITIONS "HAVE_SVN_VERSION_INC")
else()
  # Not producing a VC revision include.
  set(version_inc)

  # Being able to force-set the SVN revision in cases where it isn't available
  # is useful for performance tracking, and matches compatibility from autoconf.
  if(SVN_REVISION)
    set_source_files_properties(Version.cpp
      PROPERTIES COMPILE_DEFINITIONS "SVN_REVISION=\"${SVN_REVISION}\"")
  endif()
endif()

add_clang_library(clangBasic
  Attributes.cpp
  Builtins.cpp
  CharInfo.cpp
  Cuda.cpp
  Diagnostic.cpp
  DiagnosticIDs.cpp
  DiagnosticOptions.cpp
  FileManager.cpp
  FileSystemStatCache.cpp
  IdentifierTable.cpp
  LangOptions.cpp
  MemoryBufferCache.cpp
  Module.cpp
  ObjCRuntime.cpp
  OpenMPKinds.cpp
  OperatorPrecedence.cpp
  SanitizerBlacklist.cpp
  Sanitizers.cpp
  SourceLocation.cpp
  SourceManager.cpp
  TargetInfo.cpp
  Targets.cpp
  Targets/AArch64.cpp
  Targets/AMDGPU.cpp
  Targets/ARM.cpp
  Targets/AVR.cpp
  Targets/BPF.cpp
  Targets/Hexagon.cpp
  Targets/Lanai.cpp
  Targets/Le64.cpp
  Targets/MSP430.cpp
  Targets/Mips.cpp
  Targets/NVPTX.cpp
  Targets/Nios2.cpp
  Targets/OSTargets.cpp
  Targets/PNaCl.cpp
  Targets/PPC.cpp
  Targets/SPIR.cpp
  Targets/Sparc.cpp
  Targets/SystemZ.cpp
  Targets/TCE.cpp
  Targets/WebAssembly.cpp
  Targets/X86.cpp
  Targets/XCore.cpp
  TokenKinds.cpp
  Version.cpp
  VersionTuple.cpp
  VirtualFileSystem.cpp
  Warnings.cpp
  XRayLists.cpp
  ${version_inc}
  )

