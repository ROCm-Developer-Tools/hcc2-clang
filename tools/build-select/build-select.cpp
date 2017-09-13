//===-- build-select/build-select.cpp ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief This file implements a tool run at the link phase for amdgcn.
/// The tool builds a short function called select_outline_wrapper. 
/// The function calls the function based on a hash code generated 
/// at compile time for the particular function thus avoiding the need
/// for a function pointer and allowing inline optimization of the parallel
/// region. 
///
//===----------------------------------------------------------------------===//

#include "llvm/Bitcode/BitcodeReader.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Endian.h"
#include "llvm/Support/Format.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/ManagedStatic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Support/ErrorOr.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IR/DataLayout.h"

#include <fstream>
#include <string>
#include <vector>
#include <system_error>

using namespace llvm;

static llvm::Module * processFile(const std::string InputFilename,
  LLVMContext &Context, char**argv){
  llvm::Module *M = nullptr;
  std::string ErrorMessage;
  { 
    ErrorOr<std::unique_ptr<MemoryBuffer>> BufferOrErr =
      MemoryBuffer::getFile(InputFilename);
    std::unique_ptr<MemoryBuffer> &BufferPtr = BufferOrErr.get();
    if (std::error_code  ec = BufferOrErr.getError())
      ErrorMessage = ec.message();
    else {
      ErrorOr<std::unique_ptr<Module>> ModuleOrErr =
          expectedToErrorOrAndEmitErrors(Context,
          parseBitcodeFile(BufferPtr.get()->getMemBufferRef(), Context));
      if (std::error_code ec = ModuleOrErr.getError())
        ErrorMessage = ec.message();

      M = ModuleOrErr.get().release();
    }
  }
  if (!M) {
    errs() << argv[0] << ": ";
    if (ErrorMessage.size())
      errs() << ErrorMessage << "\n";
    else
      errs() << "bitcode didn't read correctly.\n";
  }
  return M;
}

static cl::list<std::string>
InputFilenames(cl::Positional, cl::OneOrMore,
               cl::desc("<input bitcode files>"));

static cl::opt<std::string>
OutputFilename("o", cl::desc("Output filename"),
               cl::value_desc("filename"));


int main(int argc, char **argv) {
  LLVMContext Context;
  llvm_shutdown_obj Y;
  cl::ParseCommandLineOptions(argc, argv, "Build select-output-wrapper\n");
  const Module *M = nullptr;
  llvm::SmallVector<const Module *,16> Modules;

  // Read all modules for all input bc files
  int fcount=0;
  for(auto InputFilename : InputFilenames) {
    if( !(M = processFile(InputFilename, Context, argv)) )
      return 1;
    Modules.push_back(M);
    for (auto &glob : M->globals()) {
      std::string name = glob.getName().str();
      if(name.compare(0,7,"_HASHW_")==0) {
        std::string wrapperName = name.substr(7);
#if WRITE_C_FUNCTION 
        printf("void %s(unsigned short a, unsigned int b);\n",wrapperName.c_str());
#endif
        fcount++;
      }
    }
  }

#if WRITE_C_FUNCTION 
  // Write an equivalent c function
  printf("void select_outline_wrapper(unsigned short a, unsigned int b, long long c) {\n");
  fcount = 0;
  for(const Module * M : Modules) {
    for (auto &glob : M->globals()) {
      std::string name = glob.getName().str();
      if(name.compare(0,7,"_HASHW_")==0) {
        std::string wrapperName = name.substr(7);
        const APInt &value = glob.getInitializer()->getUniqueInteger();
        const uint64_t*rawvalue = value.getRawData();
        if (fcount==0)
          printf("  if(c==%ld) %s(a,b);\n",*rawvalue,wrapperName.c_str());
        else
          printf("  else if(c==%ld) %s(a,b);\n",*rawvalue,wrapperName.c_str());
        fcount++;
      }
    }
  }
  printf("}\n");
#endif

  const DataLayout &DL = M->getDataLayout();
  const std::string &TT =  M->getTargetTriple();
    
  if (OutputFilename.empty()) {
    errs() << "no output file\n";
    return 1;
  }

  std::error_code EC;
  std::unique_ptr<tool_output_file> Out
  (new tool_output_file(OutputFilename, EC, sys::fs::F_None));
  if (EC) {
    errs() << EC.message() << '\n';
    exit(1);
  }

  LLVMContext Ctx;
  Module *MOUT = new Module("select-outline-wrapper",Ctx);
  llvm::IRBuilder<> Builder(Ctx);
  MOUT->setDataLayout(DL);
  MOUT->setTargetTriple(TT);
  llvm::Type *VoidTy = llvm::Type::getVoidTy(Ctx);
  llvm::Type *Int16Ty = llvm::Type::getInt16Ty(Ctx);
  llvm::Type *Int32Ty = llvm::Type::getInt32Ty(Ctx);
  llvm::Type *Int64Ty = llvm::Type::getInt64Ty(Ctx);
  llvm::FunctionType *FnTy = llvm::FunctionType::get(VoidTy,
          {Int16Ty, Int32Ty, Int64Ty}, /*isVarArg*/ false);
  llvm::FunctionType *FTy = llvm::FunctionType::get(VoidTy,
          {Int16Ty, Int32Ty }, /*isVarArg*/ false);
  llvm::Function *Fn = llvm::Function::Create(
     FnTy, llvm::GlobalVariable::ExternalLinkage,
    "select_outline_wrapper", MOUT);
  llvm::BasicBlock * entry = llvm::BasicBlock::Create(
    Ctx, "entry", Fn, nullptr);
  llvm::BasicBlock * exitbb = llvm::BasicBlock::Create(
    Ctx, "exit", Fn, nullptr);

  Builder.SetInsertPoint(entry);
  llvm::BasicBlock * defaultbb = llvm::BasicBlock::Create(
    Ctx, "default", Fn, nullptr);
  Builder.SetInsertPoint(defaultbb);
  Builder.CreateBr(exitbb);

  SmallVector<llvm::Value* ,4> PArgs;
  for(auto &Arg : Fn->args()) PArgs.push_back(&Arg);
  SmallVector<llvm::Value* ,4> CArgs = {PArgs[0],PArgs[1]};
  Builder.SetInsertPoint(entry);
  llvm::SwitchInst * Switch = Builder.CreateSwitch(PArgs[2],defaultbb,fcount); 

  fcount=0;
  for(const Module * M : Modules) {
    for (auto &glob : M->globals()) {
      std::string name = glob.getName().str();
      if(name.compare(0,7,"_HASHW_")==0) {
        std::string wrapperName = name.substr(7);
        const APInt &value = glob.getInitializer()->getUniqueInteger();
        const uint64_t*rawvalue = value.getRawData();
        Builder.SetInsertPoint(entry);
        llvm::BasicBlock * BB = llvm::BasicBlock::Create(
          Ctx, "BB" + wrapperName, Fn, nullptr);
        Builder.SetInsertPoint(BB);
        llvm::Function *F = llvm::Function::Create(FTy, 
           llvm::GlobalVariable::ExternalLinkage, wrapperName, MOUT);
/*
        llvm::Function *F = M->getFunction(wrapperName);
        if (!F) {
          // Wrapper should already be defined in the module unless its name changed
          // otherwise create a new one
          F = llvm::Function::Create(FTy, 
            llvm::GlobalVariable::ExternalLinkage, wrapperName, MOUT);
        }
*/
        Builder.CreateCall(F, CArgs);
        Builder.CreateBr(exitbb);
        llvm::Value* val = llvm::ConstantInt::get(Int64Ty,(long long) *rawvalue); 
        Switch->addCase(cast<llvm::ConstantInt>(val),BB);
        fcount++;
      }
    }
  }

  Builder.SetInsertPoint(exitbb);
  llvm::ReturnInst::Create(Ctx,nullptr,exitbb);
  Fn->setCallingConv(CallingConv::C);

  if (llvm::verifyFunction(*Fn)) {
    llvm::errs() << "Error in verifying function:\n";
    llvm::errs() << Fn->getName().str().c_str() << "\n";
  }

  WriteBitcodeToFile(MOUT, Out->os());
  Out->keep();
  return 0;
}

