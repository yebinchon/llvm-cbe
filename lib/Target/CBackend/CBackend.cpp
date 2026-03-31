//===-- CBackend.cpp - Library for converting LLVM code to C --------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This library converts LLVM code to C code, compilable by GCC and other C
// compilers.
//
//===----------------------------------------------------------------------===//

#include "CBackend.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetRegistry.h"

#include "TopologicalSorter.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>

#include <cstdlib>
#include <iostream>
#include <string>
#include <utility>

// SUSAN: added libs
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"

// YEBIN: added libs
#include "llvm/Demangle/Demangle.h"

// Jackson Korba 9/29/14
#ifndef DEBUG_TYPE
#define DEBUG_TYPE ""
#endif
// End Modification

// Some ms header decided to define setjmp as _setjmp, undo this for this file
// since we don't need it
#ifdef setjmp
#undef setjmp
#endif
#ifdef _MSC_VER
#define alloca _alloca
#endif
// On LLVM 10 and later, include intrinsics files.
#if LLVM_VERSION_MAJOR >= 10
#include "llvm/IR/IntrinsicsPowerPC.h"
#include "llvm/IR/IntrinsicsX86.h"
#endif

// Debug output helper
#ifndef NDEBUG
#define DBG_ERRS(x) errs() << x << " (#" << __LINE__ << ")\n"
#else
#define DBG_ERRS(x)
#endif

namespace llvm_cbe {

using namespace llvm;

static cl::opt<bool> DeclareLocalsLate(
    "cbe-declare-locals-late",
    cl::desc("C backend: Declare local variables at the point they're first "
             "assigned, "
             "if possible, rather than always at the start of the function. "
             "Note that "
             "this is not legal in standard C prior to C99."));

static cl::opt<bool> DisableAllErrsPrinting(
    "cbe-disable-all-errs-printing",
    cl::desc("C backend: Suppress all C backend stderr output."),
    cl::init(false));

// Route all unqualified errs() calls in this translation unit through a
// runtime-selectable sink.
static raw_ostream &errs() {
  if (DisableAllErrsPrinting)
    return nulls();
  return llvm::errs();
}

extern "C" void LLVMInitializeCBackendTarget() {
  // Register the target.
  RegisterTargetMachine<CTargetMachine> X(TheCBackendTarget);
}
#if LLVM_VERSION_MAJOR >= 12
bool IsPowerOfTwo(unsigned long x) { return (x & (x - 1)) == 0; }
#endif

unsigned int NumberOfElements(VectorType *TheType) {
#if LLVM_VERSION_MAJOR >= 12
  return TheType->getElementCount().getValue();
#else
  return TheType->getNumElements();
#endif
}
char CWriter::ID = 0;

// extra (invalid) Ops tags for tracking unary ops as a special case of the
// available binary ops
enum UnaryOps {
  BinaryNeg = Instruction::OtherOpsEnd + 1,
  BinaryNot,
};

#ifdef NDEBUG
#define cwriter_assert(expr)                                                   \
  do {                                                                         \
  } while (0)
#else
#define cwriter_assert(expr)                                                   \
  if (!(expr)) {                                                               \
    this->errorWithMessage(#expr);                                             \
  }
#endif

static bool
changeMapValue(std::map<Instruction *, std::map<std::string, Instruction *>>
                   prevMRVar2ValMap,
               std::map<Instruction *, std::map<std::string, Instruction *>>
                   currMRVar2ValMap,
               Function &F) {

  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    auto prev = prevMRVar2ValMap[&*I];
    auto curr = currMRVar2ValMap[&*I];
    for (auto &[prev_var, prev_val] : prev) {
      if (curr.find(prev_var) == curr.end())
        return false;
      if (curr[prev_var] != prev_val)
        return false;
    }
  }

  return true;
}

static bool isCudaBuiltinStyleArgName(StringRef Name);
static bool isShortIVStyleName(StringRef Name);
static std::string makeArgConflictFreeName(const std::string &Base,
                                           const std::set<std::string> &Reserved,
                                           std::set<std::string> &Used);

static bool isConstantNull(Value *V) {
  if (Constant *C = dyn_cast<Constant>(V))
    return C->isNullValue();
  return false;
}

static bool isNegative(Value *V) {
  if (ConstantInt *C = dyn_cast<ConstantInt>(V))
    return C->isNegative();
  return false;
}

static bool isEmptyType(Type *Ty) {
  if (!Ty)
    return false;
  if (StructType *STy = dyn_cast<StructType>(Ty))
    return STy->getNumElements() == 0 ||
           std::all_of(STy->element_begin(), STy->element_end(), isEmptyType);

  if (VectorType *VTy = dyn_cast<VectorType>(Ty))
    return NumberOfElements(VTy) == 0 || isEmptyType(VTy->getElementType());
  if (ArrayType *ATy = dyn_cast<ArrayType>(Ty))
    return ATy->getNumElements() == 0 || isEmptyType(ATy->getElementType());

  return Ty->isVoidTy();
}

bool CWriter::isEmptyType(Type *Ty) const { return llvm_cbe::isEmptyType(Ty); }

/// Peel off outer array types which have zero elements.
/// This is useful for pointers types. It isn't reasonable for values.
Type *CWriter::skipEmptyArrayTypes(Type *Ty) const {
  while (Ty->isArrayTy() && Ty->getArrayNumElements() == 0)
    Ty = Ty->getArrayElementType();
  return Ty;
}

/// isAddressExposed - Return true if the specified value's name needs to
/// have its address taken in order to get a C value of the correct type.
/// This happens for global variables, byval parameters, and direct allocas.
bool CWriter::isAddressExposed(Value *V) const {
  if (Argument *A = dyn_cast<Argument>(V))
    return ByValParams.count(A) > 0;
  else
    return isa<GlobalVariable>(V) || isDirectAlloca(V);
}

// isInlinableInst - Attempt to inline instructions into their uses to build
// trees as much as possible.  To do this, we have to consistently decide
// what is acceptable to inline, so that variable declarations don't get
// printed and an extra copy of the expr is not emitted.
bool CWriter::isInlinableInst(Instruction &I) const {
  // If the instruction is used by a PHI node, we cannot inline it because
  // the PHI node expects a defined variable in the predecessor block.
  // for (const User *U : I.users()) {
  //   if (isa<PHINode>(U))
  //     return false;
  // }

  // Always inline cmp instructions, even if they are shared by multiple
  // expressions.  GCC generates horrible code if we don't.
  //

  // if it's a store that is over written by another store in the same basic
  // block, then it's inlinable
  if (StoreInst *store = dyn_cast<StoreInst>(&I)) {
    Value *dest = store->getPointerOperand();
    for (auto it = store->getIterator(); it != store->getParent()->end();
         ++it) {
      if (&*it == store)
        continue;
      if (isa<StoreInst>(&*it) && dest == (&*it)->getOperand(1))
        return true;
    }
  }

  // Address-only arithmetic can be safely duplicated into inlined GEPs. Without
  // this, a multi-use offset feeding several inlined GEPs can end up printed as
  // a bare SSA-derived name with no emitted C temporary (e.g. deep-fission's
  // df.state.block.base used by active-mask/spill block pointers).
  if (auto *BO = dyn_cast<BinaryOperator>(&I)) {
    bool GEPOnlyUsers = !BO->user_empty();
    for (const User *U : BO->users()) {
      if (!isa<GetElementPtrInst>(U)) {
        GEPOnlyUsers = false;
        break;
      }
    }
    if (GEPOnlyUsers)
      return true;
  }

  if (isa<CmpInst>(I) || isa<GetElementPtrInst>(I) || isa<CastInst>(I))
    return true;

  // Check if we can inline a LoadInst.
  if (LoadInst *LI = dyn_cast<LoadInst>(&I)) {

    const DataLayout &DL = I.getModule()->getDataLayout();
    Value *Ptr = LI->getPointerOperand();
    Value *Underlying = GetUnderlyingObject(Ptr, DL);

    // DEBUG: Trace inlining decisions
    errs() << "CBE_DEBUG: Checking Load: " << *LI << "\n";
    errs() << "CBE_DEBUG: Underlying Object: " << *Underlying << "\n";

    // ANDREW: Only inline global variables unconditionally (e.g. constants, bounds).
    // This allows clean code like "var = Global" instead of "tmp = Global; var = tmp".
    // Also handle loads through global pointers (e.g., rowstr[j] where rowstr is
    // stored in a global). In that case, GetUnderlyingObject returns the LoadInst
    // that loaded the pointer from the global, not the global itself.
    // ALSO: Handle Arguments that are known to be read-only structure arrays in the context
    // of specific functions (like conj_grad).
    bool isGlobalBased = false;
    if (isa<GlobalVariable>(Underlying)) {
      isGlobalBased = true;
    } else if (LoadInst *UnderlyingLoad = dyn_cast<LoadInst>(Underlying)) {
      // Check if the underlying load itself loads from a global variable.
      Value *UnderlyingPtr = UnderlyingLoad->getPointerOperand();
      const DataLayout &DL2 = I.getModule()->getDataLayout();
      Value *DeepUnderlying = GetUnderlyingObject(UnderlyingPtr, DL2);
      if (isa<GlobalVariable>(DeepUnderlying)) {
        isGlobalBased = true;
      }
    } else if (Argument *Arg = dyn_cast<Argument>(Underlying)) {
      // Check if the argument is ReadOnly (e.g. "int * restrict const rowstr").
      // If the function promises not to write to it, we can treat it as safe to inline
      // (assuming no aliasing issues that would break strict C semantics, which readonly implies).
      if (Arg->onlyReadsMemory()) {
         errs() << "CBE_DEBUG:   -> ReadOnly Argument, Allowlisting for INLINE.\n";
         isGlobalBased = true;
      }
    }

    if (isGlobalBased) {
      if (LI->isVolatile()) {
        errs() << "CBE_DEBUG:   -> Volatile Global, NOT inlining.\n";
        return false;
      }
      errs() << "CBE_DEBUG:   -> Global Variable (or load through global ptr), INLINING.\n";
      return true;
    }

    if (!I.hasOneUse()) {
      errs() << "CBE_DEBUG:   -> Not Global and Multiple Uses, NOT inlining.\n";
      return false;
    }

    // We can only inline the load if there are no instructions between the load
    // and the use that could potentialy modify the memory location.
    // We use AliasAnalysis to check this.
    Instruction *User = cast<Instruction>(I.user_back());
    if (User->getParent() != LI->getParent()) {
       // If standard check fails, try AA-based "Invariant" check?
       // If AA says no instruction in the function modifies this memory, it is effectively global/invariant.
       // But checking "all instructions" is expensive.
       // For now, respect strict block scope unless isGlobalBased found it.
       // Actually, if AA proves that NO stores in the current function alias with this location,
       // and NO calls modify it, then it IS invariant.
       // But we only iterate between Load and Use here.
       // Cross-block inlining requires LoopInvariantCodeMotion logic or similar.
       // For rowstr in conj_grad, the use (phi) and load are in different blocks.
       // So we MUST return TRUE from isGlobalBased logic to bypass this.
       //
       // Wait! If AA says it points to constant memory, we can treat it as GlobalBased!
       if (AA && AA->pointsToConstantMemory(LI->getPointerOperand())) {
          errs() << "CBE_DEBUG:   -> AA proves Constant Memory, INLINING (GlobalBased).\n";
          return true; // Treat as global
       }

      errs() << "CBE_DEBUG:   -> Diff Parent Block, NOT inlining.\n";
      return false;
    }
      
    for (BasicBlock::iterator It = std::next(LI->getIterator()), E = User->getIterator();
         It != E; ++It) {
      // If we have AA, use it to check for ModRef.
      if (AA) {
        if (It->mayReadOrWriteMemory()) {
           // Check if this instruction modifies the memory location of the load.
           // ModRefInfo: Mod=Modify, Ref=Read, ModRef=Both, NoModRef=None.
           // We only care if it MODIFIES (Mod or ModRef).
           // If it only Reads, it doesn't clobber the load.
           llvm::MemoryLocation Loc = llvm::MemoryLocation::get(LI);
           if (isModSet(AA->getModRefInfo(&*It, Loc))) {
              errs() << "CBE_DEBUG:   -> AA Clobber detected (Mod), NOT inlining. Inst: " << *It << "\n";
              return false;
           }
        }
      } else {
        // Fallback to strict check
        if (It->mayWriteToMemory()) {
          errs() << "CBE_DEBUG:   -> Clobber detected (Write), NOT inlining.\n";
          return false;
        }
        if (isa<CallInst>(&*It) || isa<InvokeInst>(&*It)) {
           errs() << "CBE_DEBUG:   -> Clobber detected (Call), NOT inlining.\n";
          return false;
        }
      }
    }
    errs() << "CBE_DEBUG:   -> Safe Local, INLINING.\n";
    return true;
  }

  //
  //  if(isa<BinaryOperator>(&I) && notInlinableBinOps.find(&I) ==
  //  notInlinableBinOps.end())
  //    return true;

  // exit condition can be inlined
  if (isa<CallInst>(I) &&
      loopCondCalls.find(dyn_cast<CallInst>(&I)) != loopCondCalls.end())
    return true;

  // Must be an expression, must be used exactly once.  If it is dead, we
  // emit it inline where it would go.
  if (isEmptyType(I.getType()) || !I.hasOneUse() || I.isTerminator() ||
      // isa<CallInst>(I) || isa<PHINode>(I) || isa<LoadInst>(I) ||
      isa<CallInst>(I) || isa<PHINode>(I) || isa<VAArgInst>(I) ||
      isa<InsertElementInst>(I) || isa<InsertValueInst>(I))
    // Don't inline a load across a store or other bad things!
    return false;

  // Must not be used in inline asm, extractelement, or shufflevector.
  if (I.hasOneUse()) {
    Instruction &User = cast<Instruction>(*I.user_back());
    if (isInlineAsm(User))
      return false;
  }

  // Only inline instruction if its use is in the same BB as the inst.
  return I.getParent() == cast<Instruction>(I.user_back())->getParent();
}

bool CWriter::isAtomicAddLoweringCandidate(Instruction *I) const {
  if (!I)
    return false;

  if (I->getMetadata("tulip.atomic.add"))
    return true;

  // Fallback path while benchmarks are being migrated to metadata tagging.
  if (CallInst *CI = dyn_cast<CallInst>(I))
    if (Function *F = CI->getCalledFunction())
      if (F->getName().contains("atomicAdd"))
        return true;

  if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(I))
    return RMW->getOperation() == AtomicRMWInst::Add ||
           RMW->getOperation() == AtomicRMWInst::FAdd;

  return false;
}

bool CWriter::emitOpenMPAtomicForInst(Instruction *I) {
  if (!isAtomicAddLoweringCandidate(I))
    return false;

  Value *Ptr = nullptr;
  Value *Val = nullptr;
  bool resultUsed = !I->user_empty() && !isEmptyType(I->getType());

  if (CallInst *CI = dyn_cast<CallInst>(I)) {
    if (CI->arg_size() < 2)
      return false;
    Ptr = CI->getArgOperand(0);
    Val = CI->getArgOperand(1);
  } else if (AtomicRMWInst *RMW = dyn_cast<AtomicRMWInst>(I)) {
    if (!(RMW->getOperation() == AtomicRMWInst::Add ||
          RMW->getOperation() == AtomicRMWInst::FAdd))
      return false;
    Ptr = RMW->getPointerOperand();
    Val = RMW->getValOperand();
  } else {
    return false;
  }

  if (!Ptr || !Val)
    return false;

  auto printAtomicLHS = [&]() {
    // Only aggregate-based GEPs (struct/array stepping) print as direct scalar
    // lvalues like "(sums+iteration)->field". Scalar-pointer GEPs still print
    // as address expressions "(ptr+idx)" and must remain wrapped by "*("...")".
    auto gepPrintsDirectLValue = [&](Value *V) -> bool {
      if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
        if (Type *SrcTy = GEP->getSourceElementType())
          return SrcTy->isAggregateType();
      } else if (auto *GEPOp = dyn_cast<GEPOperator>(V)) {
        if (Type *SrcTy = GEPOp->getSourceElementType())
          return SrcTy->isAggregateType();
      }
      return false;
    };
    if (gepPrintsDirectLValue(Ptr)) {
      writeOperandInternal(Ptr);
      return;
    }
    Out << "*(";
    writeOperand(Ptr, ContextCasted);
    Out << ")";
  };

  auto isMinusOneInt = [&](Value *V) -> bool {
    if (auto *CI = dyn_cast<ConstantInt>(V))
      return CI->isMinusOne();
    return false;
  };
  auto isOneInt = [&](Value *V) -> bool {
    if (auto *CI = dyn_cast<ConstantInt>(V))
      return CI->isOne();
    return false;
  };

  if (resultUsed) {
    auto varName = GetValueName(I);
    if (declaredLocals.find(varName) == declaredLocals.end() &&
        omp_declaredLocals.find(varName) == omp_declaredLocals.end()) {
      auto declName = GetValueName(I, true);
      printTypeName(Out, I->getType(), false) << ' ';
      Out << declName << ";\n";
      if (!IS_OPENMP_FUNCTION)
        declaredLocals.insert(declName);
      else
        omp_declaredLocals.insert(declName);
      nameDict.LocalVars[demangleFunctionName(I->getFunction()->getName())]
          .insert(declName);
    }

    bool preferPostDecCapture =
        I->getMetadata("tulip.atomic.capture.postdec") || isMinusOneInt(Val);
    bool preferPostIncCapture =
        I->getMetadata("tulip.atomic.capture.postinc") || isOneInt(Val);

    Out << "  #pragma omp atomic capture\n";
    if (preferPostDecCapture) {
      // Emit a single capture expression to preserve "return old value then
      // decrement" semantics for atomicAdd(ptr, -1)-style patterns.
      Out << "  " << varName << " = (";
      printAtomicLHS();
      Out << ")--;\n";
    } else if (preferPostIncCapture) {
      // Emit a single capture expression to preserve "return old value then
      // increment" semantics for atomicAdd(ptr, +1)-style patterns.
      Out << "  " << varName << " = (";
      printAtomicLHS();
      Out << ")++;\n";
    } else {
      Out << "  {\n";
      Out << "    " << varName << " = ";
      printAtomicLHS();
      Out << ";\n";
      Out << "    ";
      printAtomicLHS();
      Out << " = " << varName << " + ";
      writeOperand(Val, ContextCasted);
      Out << ";\n";
      Out << "  }\n";
    }
  } else {
    Out << "  #pragma omp atomic update\n";
    Out << "  ";
    printAtomicLHS();
    Out << " += ";
    writeOperand(Val, ContextCasted);
    Out << ";\n";
  }

  return true;
}

// isDirectAlloca - Define fixed sized allocas in the entry block as direct
// variables which are accessed with the & operator.  This causes GCC to
// generate significantly better code than to emit alloca calls directly.
AllocaInst *CWriter::isDirectAlloca(Value *V) const {
  AllocaInst *AI = dyn_cast<AllocaInst>(V);
  if (!AI)
    return nullptr;
  if (AI->isArrayAllocation())
    return nullptr; // FIXME: we can also inline fixed size array allocas!
  if (AI->getParent() != &AI->getParent()->getParent()->getEntryBlock())
    return nullptr;
  return AI;
}

// isInlineAsm - Check if the instruction is a call to an inline asm chunk.
bool CWriter::isInlineAsm(Instruction &I) const {
  if (CallInst *CI = dyn_cast<CallInst>(&I)) {
    return isa<InlineAsm>(CI->getCalledOperand());
  } else
    return false;
}

// an 'if' or 'switch' returns only if the branch's returning or its successor
// has return statement
BasicBlock *isExitingFunction(BasicBlock *bb) {
  Instruction *term = bb->getTerminator();
  if (isa<ReturnInst>(term))
    return bb;

  if (term->getNumSuccessors() > 1)
    return nullptr;

  if (isa<UnreachableInst>(term))
    return bb;

  BasicBlock *succ = term->getSuccessor(0);
  Instruction *ret = succ->getTerminator();

  if (isa<ReturnInst>(ret))
    return succ;
  else
    return nullptr;
}

void directPathFromAtoBwithoutC(BasicBlock *fromBB, BasicBlock *toBB,
                                BasicBlock *avoidBB,
                                std::set<BasicBlock *> &visited,
                                std::set<BasicBlock *> &path,
                                bool &foundPathWithoutC) {

  visited.insert(fromBB);
  path.insert(fromBB);

  if (fromBB == toBB) {
    if (path.find(avoidBB) == path.end())
      foundPathWithoutC = true;
  } else {
    for (auto succ = succ_begin(fromBB); succ != succ_end(fromBB); ++succ) {
      BasicBlock *succBB = *succ;
      if (visited.find(succBB) == visited.end())
        directPathFromAtoBwithoutC(succBB, toBB, avoidBB, visited, path,
                                   foundPathWithoutC);
    }
  }
  visited.erase(fromBB);
}

bool directPathFromAtoBwithoutC(BasicBlock *fromBB, BasicBlock *toBB,
                                BasicBlock *avoidBB) {

  std::set<BasicBlock *> visited;
  std::set<BasicBlock *> path;
  bool foundPathWithoutC = false;

  // if(!isPotentiallyReachable(fromBB, avoidBB)) return true;

  directPathFromAtoBwithoutC(fromBB, toBB, avoidBB, visited, path,
                             foundPathWithoutC);
  return foundPathWithoutC;
}

CBERegion *CWriter::findRegionOfBlock(BasicBlock *BB) {
  std::queue<CBERegion *> toVisit;
  toVisit.push(topRegion);
  while (!toVisit.empty()) {
    CBERegion *currNode = toVisit.front();
    toVisit.pop();

    if (currNode->entryBlock == BB)
      return currNode;

    CBERegionMap[currNode->entryBlock] = currNode;
    for (CBERegion *subRegion : currNode->thenSubRegions) {
      toVisit.push(subRegion);
    }
    for (CBERegion *subRegion : currNode->elseSubRegions) {
      toVisit.push(subRegion);
    }
  }
  return nullptr;
}

bool CWriter::alreadyVisitedRegion(BasicBlock *bbUT) {
  std::set<CBERegion *> regions;
  for (auto &[region, bb] : recordedRegionBBs) {
    if (bb == bbUT)
      return true;
  }
  return false;
}

void CWriter::CountTimes2bePrintedByRegionPath() {
  std::stack<CBERegion *> toVisit;

  toVisit.push(topRegion);

  while (!toVisit.empty()) {
    CBERegion *currRegion = toVisit.top();
    toVisit.pop();

    // if(currRegion->thenSubRegions.empty() &&
    // currRegion->elseSubRegions.empty()){
    //   errs() << "currRegion: " << *currRegion->br << "\n";
    //   CBERegion* parent = currRegion->parentRegion;
    //   std::set<BasicBlock*> blocks2cnt;
    //   while(parent){
    //     for(auto bb : currRegion->thenBBs){
    //       if(std::count(parent->thenBBs.begin(), parent->thenBBs.end(), bb))
    //         blocks2cnt.insert(bb);
    //       if(std::count(parent->elseBBs.begin(), parent->elseBBs.end(), bb))
    //         blocks2cnt.insert(bb);
    //     }
    //     for(auto bb : currRegion->elseBBs){
    //       if(std::count(parent->thenBBs.begin(), parent->thenBBs.end(), bb))
    //         blocks2cnt.insert(bb);
    //       if(std::count(parent->elseBBs.begin(), parent->elseBBs.end(), bb))
    //         blocks2cnt.insert(bb);
    //     }
    //     child = parent;
    //     parent = parent->parentRegion;
    //   }

    //}

    for (auto bb : currRegion->thenBBs) {
      if (returnDominated && currRegion == topRegion &&
          isa<ReturnInst>(bb->getTerminator())) {
        errs() << "SUSAN: found duplicated then return BB\n";
        continue;
      }
      times2bePrinted[bb]++;
    }
    for (auto bb : currRegion->elseBBs) {
      if (returnDominated && currRegion == topRegion &&
          isa<ReturnInst>(bb->getTerminator())) {
        errs() << "SUSAN: found duplicated else return BB\n";
        continue;
      }
      times2bePrinted[bb]++;
    }

    for (auto subRegion : currRegion->thenSubRegions)
      toVisit.push(subRegion);
    for (auto subRegion : currRegion->elseSubRegions)
    toVisit.push(subRegion);
  }
}

CBERegion *CWriter::createNewRegion(BasicBlock *entryBB, CBERegion *parentR,
                                    bool isElseRegion) {
  // create a new region
  CBERegion *newR = new CBERegion();
  newR->parentRegion = parentR;
  if (isElseRegion)
    parentR->elseSubRegions.push_back(newR);
  else
    parentR->thenSubRegions.push_back(newR);
  newR->entryBlock = entryBB;
  recordedRegionBBs[newR] = entryBB;
  return newR;
}

void CWriter::markBranchRegion(Instruction *br, CBERegion *targetRegion) {
  errs() << "=================SUSAN: START OF marking region : "
         << br->getParent()->getName() << "==================\n";
  BasicBlock *currBB = br->getParent();

  // analyse the branch properties
  BasicBlock *exitingBB = currBB;
  BasicBlock *exitLoopTrueBB = nullptr;
  BasicBlock *exitLoopFalseBB = nullptr;
  for (unsigned int i_succ = 0; i_succ < br->getNumSuccessors(); ++i_succ) {
    BasicBlock *exitBB = br->getSuccessor(i_succ);
    for (auto edge : irregularLoopExits) {
      if (edge.first == exitingBB && edge.second == exitBB) {
        if (i_succ == 0)
          exitLoopTrueBB = exitBB;
        else if (i_succ == 1)
          exitLoopFalseBB = exitBB;
      }
    }
  }
  BasicBlock *brBB = currBB;
  BasicBlock *trueStartBB = br->getSuccessor(0);
  BasicBlock *falseStartBB = br->getSuccessor(1);
  bool exitFunctionTrueBr = isExitingFunction(trueStartBB);
  bool exitFunctionFalseBr = isExitingFunction(falseStartBB);
  bool trueBrOnly = noElseRegion(true, brBB);
  bool falseBrOnly = noElseRegion(false, brBB);
  returnDominated = dominatedByReturn(brBB);
  if (!trueBrOnly && !falseBrOnly && returnDominated == -1) {
    trueBrOnly = (exitFunctionTrueBr && !exitFunctionFalseBr) || exitLoopTrueBB;
    falseBrOnly =
        (exitFunctionFalseBr && !exitFunctionTrueBr) || exitLoopFalseBB;
  }
  // end of analysis

  CBERegion *currRegion = targetRegion;
  if (exitLoopFalseBB || exitLoopTrueBB) {
    BasicBlock *exitBB = exitLoopFalseBB ? exitLoopFalseBB : exitLoopTrueBB;
    currRegion->thenEdges.push_back(std::make_pair(brBB, exitBB));
    currRegion->thenBBs.push_back(exitBB);
    // if succBB of exitBB is returning, don't print break, print return block
    for (auto ret = succ_begin(exitBB); ret != succ_end(exitBB); ++ret) {
      BasicBlock *retBB = *ret;
      if (retBB && !nodeBelongsToRegion(retBB, currRegion)) {
        currRegion->thenEdges.push_back(std::make_pair(exitBB, retBB));
        currRegion->thenBBs.push_back(retBB);
      }
    }

    if (exitLoopFalseBB)
      recordTimes2bePrintedForBranch(trueStartBB, brBB, falseStartBB,
                                     currRegion, true);
    else
      recordTimes2bePrintedForBranch(falseStartBB, brBB, trueStartBB,
                                     currRegion, true);

    return;
  }

  if (trueBrOnly && returnDominated == -1) {
    recordTimes2bePrintedForBranch(trueStartBB, brBB, falseStartBB, currRegion);

    BasicBlock *ret = isExitingFunction(trueStartBB);
    if (ret && !nodeBelongsToRegion(ret, currRegion) && ret != trueStartBB) {
      if (ret == trueStartBB)
        currRegion->thenEdges.push_back(std::make_pair(brBB, ret));
      else
        currRegion->thenEdges.push_back(std::make_pair(trueStartBB, ret));

      currRegion->thenBBs.push_back(ret);
    }

    // the other branch belongs to TopRegion if not belong to subregion
    recordTimes2bePrintedForBranch(falseStartBB, brBB, trueStartBB, currRegion,
                                   true);
  }
  // Case 3: only print if body with reveresed case
  else if (falseBrOnly && returnDominated == -1) {
    recordTimes2bePrintedForBranch(falseStartBB, brBB, trueStartBB, currRegion);

    BasicBlock *ret = isExitingFunction(falseStartBB);
    if (ret && !nodeBelongsToRegion(ret, currRegion) && ret != falseStartBB) {
      if (ret == trueStartBB)
        currRegion->thenEdges.push_back(std::make_pair(brBB, ret));
      else
        currRegion->thenEdges.push_back(std::make_pair(falseStartBB, ret));
      currRegion->thenBBs.push_back(ret);
    }

    // the other branch belongs to TopRegion if not belong to subregion
    recordTimes2bePrintedForBranch(trueStartBB, brBB, falseStartBB, currRegion,
                                   true);
  }
  // Case 4: print if & else;
  else {
    recordTimes2bePrintedForBranch(trueStartBB, brBB, falseStartBB, currRegion);

    BasicBlock *ret = isExitingFunction(trueStartBB);
    if (ret &&
        !nodeBelongsToRegion(ret, currRegion)) { // && ret != trueStartBB){
      if (ret == falseStartBB)
        currRegion->thenEdges.push_back(std::make_pair(brBB, ret));
      else
        currRegion->thenEdges.push_back(std::make_pair(trueStartBB, ret));
      currRegion->thenBBs.push_back(ret);
    }
    recordTimes2bePrintedForBranch(falseStartBB, brBB, trueStartBB, currRegion,
                                   true);

    ret = isExitingFunction(falseStartBB);
    if (ret && !nodeBelongsToRegion(ret, currRegion,
                                    true)) { // && ret != falseStartBB){
      if (ret == trueStartBB)
        currRegion->elseEdges.push_back(std::make_pair(brBB, ret));
      else
        currRegion->elseEdges.push_back(std::make_pair(falseStartBB, ret));
      currRegion->elseBBs.push_back(ret);
    }
  }

  errs() << "=================SUSAN: END OF marking region : "
         << br->getParent()->getName() << "==================\n";
}

int CBERegion2::whichRegion(BasicBlock *entryBB, LoopInfo *LI) {
  Loop *L = LI->getLoopFor(entryBB);
  if (L) {
    errs() << "CBackend: entryBB is a loop: " << entryBB->getName() << "\n";

    if (L->getHeader() == entryBB)
      return 2;
    errs() << "but not a header!\n";
  }

  errs() << *entryBB->getTerminator() << "\n";
  if (BranchInst *br = dyn_cast<BranchInst>(entryBB->getTerminator()))
    if (br->isConditional()) {
      // special case: do-while loop latch, not if-else
      if(L && L->getLoopLatch() == entryBB)
        return 0;
      return 1;
    }

  return 0;
}

void CWriter::markBBwithNumOfVisits(Function &F) {

  // set up top region
  topRegion = new CBERegion();
  topRegion->entryBlock = nullptr;
  topRegion->parentRegion = nullptr;
  for (auto &BB : F) {
    // topRegion.thenBBs.push_back(&BB);
    times2bePrinted[&BB] = 0;
  }

  returnDominated = -1;
  recordTimes2bePrintedForBranch(&F.getEntryBlock(), nullptr, nullptr,
                                 topRegion);

  // despite root node, each leaf-to-child_of_root path will contain a set of
  // BBs, these BBs times3bePrinted need to be incrememnted, lastly any node
  // with times2bePrinted = 0 means it belong to the entry node and therefore
  // times2bePrinted = 1
  std::vector<CBERegion *> regionPath;
  CountTimes2bePrintedByRegionPath();

  // view the tree:
  // record times2bePrinted
  // record RegionMap
  std::queue<CBERegion *> toVisit;
  toVisit.push(topRegion);
  while (!toVisit.empty()) {
    CBERegion *currNode = toVisit.front();
    toVisit.pop();
    if (currNode->entryBlock)
      errs() << "SUSAN: Node " << (currNode->entryBlock->getName()) << "\n";
    else
      errs() << "SUSAN: Node: topRegion\n";

    errs() << "then SubNodes: \n";
    for (auto subNode : currNode->thenSubRegions) {
      errs() << (subNode->entryBlock->getName()) << "\n";
    }

    errs() << "else SubNodes: \n";
    for (auto subNode : currNode->elseSubRegions) {
      errs() << (subNode->entryBlock->getName()) << "\n";
    }

    errs() << "current region then bbs:\n";
    for (auto BB : currNode->thenBBs) {
      errs() << BB->getName() << "\n";
    }

    errs() << "current region else bbs:\n";
    for (auto BB : currNode->elseBBs) {
      errs() << BB->getName() << "\n";
    }

    errs() << "current region then edges:\n";
    for (auto edge : currNode->thenEdges) {
      BasicBlock *from = edge.first;
      BasicBlock *to = edge.second;
      if (from && to)
        errs() << from->getName() << " -> " << to->getName() << "\n";
    }

    errs() << "current region else edges:\n";
    for (auto edge : currNode->elseEdges) {
      BasicBlock *from = edge.first;
      BasicBlock *to = edge.second;
      if (from && to)
        errs() << from->getName() << " -> " << to->getName() << "\n";
    }

    CBERegionMap[currNode->entryBlock] = currNode;
    for (CBERegion *subRegion : currNode->thenSubRegions) {
      toVisit.push(subRegion);
    }
    for (CBERegion *subRegion : currNode->elseSubRegions) {
      toVisit.push(subRegion);
    }
  }

  for (auto &BB : F) {
    if (!times2bePrinted[&BB]) {
      std::vector<BasicBlock *> preds(pred_begin(&BB), pred_end(&BB));
      times2bePrinted[&BB] = preds.size() ? preds.size() : 1;
    }
    // if(isa<ReturnInst>(BB.getTerminator())){
    //   std::vector<BasicBlock*> preds(pred_begin(&BB), pred_end(&BB));
    //   times2bePrinted[&BB] = preds.size() ? preds.size() : 1;
    // }
    errs() << "SUSAN: BB " << BB.getName()
           << " times2bePrinted: " << times2bePrinted[&BB] << "\n";
  }
}

void collectNoneArrayGEPsDownStream(
    GetElementPtrInst *gepInst, std::set<GetElementPtrInst *> &NoneArrayGEPs) {
  // collect StructGeps DownStream
  GetElementPtrInst *gep = gepInst;
  Value *opnd = gep->getPointerOperand();
  while (GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(opnd)) {
    NoneArrayGEPs.insert(gep);
    opnd = gep->getPointerOperand();
  }
}

void CheckAndAddArrayGep2NoneArrayGEPs(
    GetElementPtrInst *gepInst, std::set<GetElementPtrInst *> &NoneArrayGEPs) {
  GetElementPtrInst *gep = gepInst;
  Value *opnd = gep->getPointerOperand();
  while (GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(opnd)) {
    NoneArrayGEPs.insert(gep);
    if (NoneArrayGEPs.find(gep) != NoneArrayGEPs.end()) {
      NoneArrayGEPs.insert(gepInst);
      break;
    }
  }
}

void CWriter::findVariableDepth(Type *Ty, Value *UO, int depths) {
  if (++depths > 20)
    return;

  if (Times2Dereference.find(UO) == Times2Dereference.end())
    Times2Dereference[UO] = 0;
  else
    Times2Dereference[UO]++;

  if (isa<IntegerType>(Ty) || Ty->isFloatTy() || Ty->isDoubleTy())
    return;

  if (PointerType *ptrTy = dyn_cast<PointerType>(Ty)) {
    Type *nextTy = ptrTy->getPointerElementType();
    findVariableDepth(nextTy, UO, depths);
  } else if (ArrayType *arrTy = dyn_cast<ArrayType>(Ty)) {
    Type *nextTy = arrTy->getArrayElementType();
    findVariableDepth(nextTy, UO, depths);
  } else if (StructType *strucTy = dyn_cast<StructType>(Ty)) {
    for (StructType::element_iterator I = strucTy->element_begin(),
                                      E = strucTy->element_end();
         I != E; ++I) {
      Type *nextTy = *I;
      if (nextTy == Ty) // recursive case
        Times2Dereference[UO] = 20;
      else if (PointerType *ptrTy = dyn_cast<PointerType>(nextTy)) {
        if (ptrTy->getPointerElementType() == Ty) // recursive case
          Times2Dereference[UO] = 20;
      } else if (isa<PointerType>(nextTy) || isa<ArrayType>(nextTy) ||
                 isa<StructType>(nextTy))
        findVariableDepth(nextTy, UO, depths);
    }
  }
}

void CWriter::collectVariables2Deref(Function &F) {
  // SUSAN: build the table of local variable : times to be dereferenced
  // GEP might not be directly connected to a site
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    Type *instTy = (*I).getType();
    findVariableDepth(instTy, cast<Value>(&*I), 0);
  }

  static bool collected = false;
  // add global variables to dereftable
  // FIXME: only need to do it once...
  //

  if (!collected) {
    Module *M = F.getParent();
    for (Module::global_iterator I = M->global_begin(), E = M->global_end();
         I != E; ++I) {
      GlobalVariable *glob = &*I;
      if (glob->hasInitializer()) {
        Constant *globVal = glob->getInitializer();
        Type *globTy = globVal->getType();
        if (isa<ArrayType>(globTy) || isa<StructType>(globTy) ||
            isa<PointerType>(globTy)) {
          errs() << "global: " << *glob << "\n";
          errs() << "type: " << *globTy << "\n";
          findVariableDepth(globTy, cast<Value>(glob), 0);
        }
      }
    }
  }

  collected = true;

  // collect phi nodes that might be pointers/array/structs
}

void CWriter::collectLateDeclares(Function &F) {
  std::list<Loop *> loops(LI->begin(), LI->end());
  for (auto L : loops) {
    Instruction *term = L->getHeader()->getTerminator();
    // if(!term->getMetadata("splendid.doall.loop")) continue;
    for (unsigned i = 0, e = L->getBlocks().size(); i != e; ++i) {
      BasicBlock *BB = L->getBlocks()[i];
      for (auto &I : *BB) {
        // if (isInductionVariable(&I) || isExtraInductionVariable(&I) ||
        // isIVIncrement(&I)) continue; if (isIVIncrement(&I)) continue;
        if (isSkipableInst(&I))
          continue;
        if (isDirectAlloca(&I))
          continue;
        toDeclareLocals.insert(&I);
      }
    }
  }
}

void CWriter::collectNoneArrayGEPs(Function &F) {

  std::set<GetElementPtrInst *> arrayGeps;
  // collect array geps, then the rest goes into struct geps, anything in the
  // downstream of struct geps also goes to struct geps
  for (auto &BB : F) {
    for (auto &I : BB) {
      if (GetElementPtrInst *gepInst = dyn_cast<GetElementPtrInst>(&I)) {
        if (isa<ArrayType>(gepInst->getSourceElementType())) {
          arrayGeps.insert(gepInst);
        } else {
          NoneArrayGEPs.insert(gepInst);
          collectNoneArrayGEPsDownStream(gepInst, NoneArrayGEPs);
        }
      }
    }
  }

  // collect array geps into none array geps if array geps has struct geps down
  // stream
  for (auto gep : arrayGeps) {
    CheckAndAddArrayGep2NoneArrayGEPs(gep, NoneArrayGEPs);
    collectNoneArrayGEPsDownStream(gep, NoneArrayGEPs);
  }
}

void CWriter::markBackEdges(Function &F) {
  std::set<BasicBlock *> visited;
  std::queue<BasicBlock *> toVisit;
  visited.insert(&F.getEntryBlock());
  toVisit.push(&F.getEntryBlock());

  while (!toVisit.empty()) {
    BasicBlock *currBB = toVisit.front();
    toVisit.pop();

    for (auto succ = succ_begin(currBB); succ != succ_end(currBB); ++succ) {
      BasicBlock *succBB = *succ;
      if (DT->dominates(succBB, currBB)) {
        backEdges.insert(std::make_pair(currBB, succBB));
      }

      if (visited.find(succBB) == visited.end()) {
        visited.insert(succBB);
        toVisit.push(succBB);
      }
    }
  }
}

std::set<BasicBlock *> CWriter::findRegionEntriesOfBB(BasicBlock *BB) {
  std::set<BasicBlock *> entries;
  std::queue<CBERegion *> toVisit;
  toVisit.push(topRegion);

  while (!toVisit.empty()) {
    CBERegion *currNode = toVisit.front();
    toVisit.pop();

    for (auto regionBB : currNode->thenBBs)
      if (regionBB == BB && currNode->entryBlock)
        entries.insert(currNode->entryBlock);

    for (auto regionBB : currNode->elseBBs)
      if (regionBB == BB && currNode->entryBlock)
        entries.insert(currNode->entryBlock);

    for (CBERegion *subRegion : currNode->thenSubRegions) {
      toVisit.push(subRegion);
    }
    for (CBERegion *subRegion : currNode->elseSubRegions) {
      toVisit.push(subRegion);
    }
  }

  return entries;
}

void CWriter::determineControlFlowTranslationMethod(Function &F) {
  NATURAL_CONTROL_FLOW = true;
  // markBBwithNumOfVisits(F);

  BasicBlock *returnBB = nullptr;
  for (auto &BB : F)
    if (isa<ReturnInst>(BB.getTerminator())) {
      returnBB = &BB;
      break;
    }
  assert(returnBB && "didn't find returnBB\n");

  TopRegion = new CBERegion2(LI, PDT, DT, this);
  TopRegion->createCBERegionDAG(&F.getEntryBlock(), nullptr, returnBB);
}

Instruction *CWriter::getIVIncrement(Loop *L, PHINode *IV) {
  if (!IV)
    return nullptr;
  for (unsigned i = 0; i < IV->getNumIncomingValues(); ++i) {
    BasicBlock *predBB = IV->getIncomingBlock(i);
    if (LI->getLoopFor(predBB) == L)
      return dyn_cast<Instruction>(IV->getIncomingValue(i));
  }
  return nullptr;
}

PHINode *CWriter::getInductionVariable(Loop *L) {
  if (!L) {
    errs() << "ANDREW: getInductionVariable got null loop\n";
    return nullptr;
  }

  errs() << "ANDREW: getInductionVariable\n";
  errs() << "trying to get IV for Loop:" << L->getHeader()->getName() << "\n";
  PHINode *InnerIndexVar = L->getCanonicalInductionVariable();
  if (InnerIndexVar) {
    errs() << "SUSAN: found IV 784" << *InnerIndexVar << "\n";
    return InnerIndexVar;
  }
  if (L->getLoopLatch() == nullptr || L->getLoopPredecessor() == nullptr) {
    errs() << "SUSAN: didn't find IV 788\n";
    return nullptr;
  }
  for (BasicBlock::iterator I = L->getHeader()->begin(); isa<PHINode>(I); ++I) {
    PHINode *PhiVar = cast<PHINode>(I);
    errs() << "SUSAN: phi: " << *PhiVar << "\n";
    Type *PhiTy = PhiVar->getType();
    if (!PhiTy->isIntegerTy() && !PhiTy->isFloatingPointTy() &&
        !PhiTy->isPointerTy()) {
      errs() << "SUSAN: didn't find IV 796\n";
      return nullptr;
    }

    const SCEVAddRecExpr *AddRec = nullptr;
    if (SE->isSCEVable(PhiVar->getType()))
      AddRec = dyn_cast<SCEVAddRecExpr>(SE->getSCEV(PhiVar));
    if (!AddRec || !AddRec->isAffine()) {
      errs() << "SUSAN: can't find addRec\n";
      continue;
    }
    // const SCEV *Step = AddRec->getStepRecurrence(*SE);
    // if (!isa<SCEVConstant>(Step) || !isa<SCEVSequentialMinMaxExpr>(Step)){
    //   errs() << "SUSAN: step isn't constant\n";
    //   continue;
    // }

    // Found the induction variable.
    // FIXME: Handle loops with more than one induction variable. Note that,
    // currently, legality makes sure we have only one induction variable.
    errs() << "SUSAN: find IV 809\n";
    return PhiVar;
  }
  errs() << "SUSAN: didn't find IV 812\n";
  return nullptr;
}

void CWriter::CreateOmpLoops(Loop *L, Value *ub, Value *lb, Value *incr) {
  LoopProfile *ompLI = new LoopProfile();
  ompLI->L = L;
  ompLI->LoopHeader = L ? L->getHeader() : nullptr;
  ompLI->ub = ub;
  ompLI->lb = lb;
  ompLI->incr = incr;
  ompLI->IV = getInductionVariable(L);
  ompLI->isOmpLoop = true;
  ompLI->isForLoop = true;
  assert(ompLI->IV && "SUSAN: only translate for loop for omp right now\n");
  LoopProfiles.insert(ompLI);
}

Loop *CWriter::findLoopAccordingTo(Function &F, Value *bound) {
  Instruction *boundI = dyn_cast<Instruction>(bound);
  if (!boundI) {
    return nullptr;
  }

  std::queue<Instruction *> toVisit;
  std::set<Instruction *> visited;
  toVisit.push(boundI);
  visited.insert(boundI);
  while (!toVisit.empty()) {
    Instruction *currInst = toVisit.front();
    toVisit.pop();

    if (isa<CmpInst>(currInst)) {
      Loop *L = LI->getLoopFor(currInst->getParent());
      if (L)
        return L;
    }

    for (User *U : currInst->users()) {
      if (Instruction *inst = dyn_cast<Instruction>(U)) {
        if (visited.find(inst) == visited.end()) {
          visited.insert(inst);
          toVisit.push(inst);
        }
      }
    }
  }

  return nullptr;
}

void CWriter::preprossesPHIs2Print(Function &F) {
  std::map<PHINode *, PHINode *> phiLoops;

  // Helper lambda to check if adding key -> value would create a cycle
  // by checking if value transitively maps back to key
  auto wouldCreateCycle = [this](Value *key, Value *value) -> bool {
    std::set<Value*> visited;
    Value *current = value;
    while (current) {
      if (current == key) return true;  // Cycle detected
      if (!visited.insert(current).second) return false; // Hit a different cycle, not involving key
      auto it = InstsToReplaceByPhi.find(current);
      if (it == InstsToReplaceByPhi.end()) return false; // End of chain
      current = it->second;
    }
    return false;
  };

  auto isLoopIVOrRelated = [this](Value *V) -> bool {
    if (!V)
      return false;
    if (PHINode *phi = dyn_cast<PHINode>(V))
      return isInductionVariable(phi) || isExtraInductionVariable(phi);
    if (Instruction *inst = dyn_cast<Instruction>(V))
      return isIVIncrement(inst) || isExtraIVIncrement(inst);
    return false;
  };

  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I)
    if (PHINode *phi = dyn_cast<PHINode>(&*I)) {

      Value *ldPtr = nullptr;
      Value *stPtr = nullptr;
      LoadInst *ldInst = nullptr;
      StoreInst *stInst = nullptr;

      // For canonical for-loops, IV initialization is emitted in loop syntax.
      // For while-style loops we still need PHI materialization (e.g. lg = 1).
      Loop *phiLoop = LI->getLoopFor(phi->getParent());
      LoopProfile *phiLP = phiLoop ? findLoopProfile(phiLoop) : nullptr;
      bool isLoopIV = isInductionVariable(phi);
      if (phiLP && phiLP->isForLoop && isLoopIV) {
        errs() << "ANDREW: skipping loop IV PHI materialization for for-loop: "
               << *phi << "\n";
        continue;
      }
      for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
        BasicBlock *predBB = phi->getIncomingBlock(i);
        Value *phiVal = phi->getIncomingValue(i);
        if (!isa<UndefValue>(phiVal) && !isEmptyType(phi->getType())) {
          bool skipStderr = false;
          if (LoadInst *ld = dyn_cast<LoadInst>(phiVal))
            if (ld->getPointerOperand()->getName() == "stderr") {
              skipStderr = true;
              IRNaming.insert(std::make_pair(ld, "stderr"));
            }
          if (!skipStderr) {
            errs() << "YEBIN: PHI to print: " << *phi << "\n";
            errs() << "     from predBB: " << predBB->getName() << "\n";
            errs() << "CBE_DEBUG: incoming value for PHI print: " << *phiVal
                   << "\n";
            PHIValues2Print.insert(std::make_pair(predBB, phi));
          }
        }

        if (PHINode *incomingPhi = dyn_cast<PHINode>(phiVal)) {
          // detect a circle
          for (unsigned i = 0; i < incomingPhi->getNumIncomingValues(); ++i)
            if (incomingPhi->getIncomingValue(i) == phi) {
              phiLoops[incomingPhi] = phi;
              break;
            }
        }

        PHINode *replaceVal = phi;
        if (phiLoops.find(phi) != phiLoops.end()) {
          if (phiLoops[phi] == phiVal)
            continue;
          replaceVal = phiLoops[phi];
        }

        // Keep explicit return-merge PHIs stable. Coalescing incoming values
        // into the return PHI can rewrite "return retval" into one branch-local
        // variable and drop required merge-edge copies.
        bool phiFeedsReturn = false;
        for (User *U : phi->users()) {
          if (isa<ReturnInst>(U)) {
            phiFeedsReturn = true;
            break;
          }
        }
        if (phiFeedsReturn) {
          errs() << "ANDREW: SKIPPING return-merge PHI coalescing: " << *phi
                 << "\n";
          continue;
        }

        // Keep loop IVs and extra IVs distinct. Collapsing these via
        // InstsToReplaceByPhi can rewrite loop-carried variables (e.g. nza)
        // into a different IV (e.g. k), corrupting indexed assignments.
        if (isLoopIVOrRelated(phiVal) || isLoopIVOrRelated(replaceVal)) {
          errs() << "ANDREW: SKIPPING IV-related PHI coalescing: " << *phiVal
                 << " -> " << *replaceVal;
          if (Loop *phiLoop = LI->getLoopFor(phi->getParent()))
            errs() << " in loop " << phiLoop->getHeader()->getName();
          errs() << "\n";
          continue;
        }

        if (Instruction *incomingInst = dyn_cast<Instruction>(phiVal)) {
          // Guard PHI->PHI coalescing across unrelated blocks.
          // Without dominance safety, a later PHI name can leak backward into
          // earlier blocks (e.g. while condition using a value only defined in
          // a post-loop/if-merge PHI), producing incorrect C variables.
          if (auto *incomingPhi = dyn_cast<PHINode>(incomingInst)) {
            if (auto *replacePhi = dyn_cast<PHINode>(replaceVal)) {
              bool sameBlock =
                  incomingPhi->getParent() == replacePhi->getParent();
              bool dominatesIncoming =
                  DT && DT->dominates(replacePhi->getParent(),
                                      incomingPhi->getParent());
              if (!sameBlock && !dominatesIncoming) {
                errs() << "ANDREW: SKIPPING non-dominating PHI coalescing: "
                       << *incomingPhi << " -> " << *replacePhi << "\n";
                continue;
              }
            }
          }

          // ANDREW: Skip coalescing for loop-carried PHI values where the 
          // incoming instruction is computed inside the same loop.
          // This prevents incorrect substitution like %div -> %kk.0 when
          // %div is computed from %kk.0 inside the loop body.
          Loop *phiLoop = LI->getLoopFor(phi->getParent());
          Loop *incomingLoop = LI->getLoopFor(incomingInst->getParent());
          if (phiLoop && incomingLoop && phiLoop == incomingLoop) {
            errs() << "ANDREW: SKIPPING loop-carried coalescing: " << *incomingInst
                   << " -> " << *replaceVal << "\n";
            continue;  // Skip this - they need to be separate variables
          }
          
          if (deleteAndReplaceInsts.find(incomingInst) !=
              deleteAndReplaceInsts.end()) {
            Value *keyVal = deleteAndReplaceInsts[incomingInst];
            if (isLoopIVOrRelated(keyVal)) {
              errs() << "ANDREW: SKIPPING IV-related deleteInst mapping: "
                     << *keyVal << " -> " << *replaceVal << "\n";
              continue;
            }
            // Only add mapping if it doesn't create a cycle
            if (!wouldCreateCycle(keyVal, replaceVal)) {
              errs() << "YEBIN: replacing PHI deleteInsts: " << *keyVal
                     << " with " << *replaceVal << "\n";
              InstsToReplaceByPhi[keyVal] = replaceVal;
            } else {
              errs() << "YEBIN: SKIPPING cycle-creating mapping: " << *keyVal
                     << " -> " << *replaceVal << "\n";
            }
          }
          else {
            // Only add mapping if it doesn't create a cycle
            if (!wouldCreateCycle(phiVal, replaceVal)) {
              errs() << "YEBIN: replacing PHI incomingInst: " << *incomingInst
                     << " with " << *replaceVal << "\n";
              InstsToReplaceByPhi[phiVal] = replaceVal;
            } else {
              errs() << "YEBIN: SKIPPING cycle-creating mapping: " << *phiVal
                     << " -> " << *replaceVal << "\n";
            }
          }
        }
      }
    }
}

Value *CWriter::findOriginalUb(Function &F, Value *ub, CallInst *initCI,
                               CallInst *prevFini, int &offset) {
  bool startSearching = prevFini ? false : true;
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    if (&*I == initCI)
      break;
    if (&*I == prevFini)
      startSearching = true;
    if (!startSearching)
      continue;

    if (StoreInst *store = dyn_cast<StoreInst>(&*I))
      if (store->getOperand(1) == ub) {
        BinaryOperator *subInst =
            dyn_cast<BinaryOperator>(store->getOperand(0));
        if (subInst && (subInst->getOpcode() == Instruction::Add ||
                        subInst->getOpcode() == Instruction::FAdd)) {
          if (ConstantInt *OffSetMinusOne =
                  dyn_cast<ConstantInt>(subInst->getOperand(1))) {
            // if(minusOne->getSExtValue() == -1){
            Value *opnd0 = subInst->getOperand(0);
            offset = OffSetMinusOne->getSExtValue() + 1;
            // if(LoadInst* ld = dyn_cast<LoadInst>(opnd0)) return
            // ld->getPointerOperand(); else return opnd0;
            return opnd0;
            //}
          }
        }

        Argument *arg = dyn_cast<Argument>(store->getOperand(0));
        if (arg) {
          UpperBoundArgs.insert(arg);
          return arg;
        }
      }
  }
  // errs() << "SUSAN: ub: " << *ub << "\n";
  return ub;
}

void CWriter::omp_preprossesing(Function &F) {
  // FIXME: currently only searching for the loop to be processed
  // find __kmpc_for_static_init and associated loop info
  Value *lb, *ub, *incr;
  int schedtype, chunksize;
  CallInst *initCI, *finiCI;
  initCI = nullptr;
  finiCI = nullptr;
  lb = nullptr;
  ub = nullptr;
  incr = nullptr;
  LoopProfile *currLP = nullptr;
  int countBarrier = 0;
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    if (CallInst *CI = dyn_cast<CallInst>(&*I)) {
      if (Function *ompCall = CI->getCalledFunction()) {

        /*
         * OpenMP: translate omp parallel for schedule (static)
         */
        if (ompCall->getName().contains("__kmpc_for_static_init")) {
          LoopProfile *ompLP = new LoopProfile();
          ompLP->LoopHeader = nullptr;
          initCI = CI;
          omp_SkipVals.insert(cast<Value>(CI));

          // find the value stored into lb
          lb = CI->getArgOperand(4);
          if (schedtype == 33 || schedtype == 34) {
            errs() << "SUSAN: schedtype is static!\n";
          }
          ompLP->lbAlloca = lb;
          ompLP->schedtype =
              cast<ConstantInt>(CI->getArgOperand(2))->getSExtValue();
          ompLP->chunksize =
              cast<ConstantInt>(CI->getArgOperand(8))->getSExtValue();
          errs() << "SUSAN: lbAlloca: " << *lb << "\n";
          for (User *U : lb->users()) {
            if (StoreInst *store = dyn_cast<StoreInst>(U))
              lb = store->getOperand(0);
          }
          ompLP->lb = lb;

          // find ub & incr
          int ubOffset = 0;
          ompLP->ub =
              findOriginalUb(F, CI->getArgOperand(5), initCI, finiCI, ubOffset);
          errs() << "SUSAN: original ub: " << *(ompLP->ub) << "\n";
          ompLP->ubOffset = ubOffset;
          ompLP->incr = CI->getArgOperand(7);
          currLP = ompLP;
        } else if (ompCall->getName().contains("__kmpc_for_static_fini")) {
          finiCI = CI;
          Loop *ompLoop = nullptr;
          // find loop in between init and fini
          for (auto &BB : F) {
            if (DT->dominates(initCI->getParent(), &BB) &&
                PDT->dominates(finiCI->getParent(), &BB)) {
              Loop *dominatedLoop = LI->getLoopFor(&BB);
              if (!dominatedLoop)
                continue;

              bool loopIsDominated = true;
              for (unsigned i = 0, e = dominatedLoop->getBlocks().size();
                   i != e; ++i) {
                BasicBlock *domBB = dominatedLoop->getBlocks()[i];
                if (!DT->dominates(initCI->getParent(), domBB) ||
                    !PDT->dominates(finiCI->getParent(), domBB)) {
                  loopIsDominated = false;
                  break;
                }
              }

              if (!loopIsDominated)
                continue;

              ompLoop = dominatedLoop;
              break;
            }
          }
          assert(ompLoop && "didn't find omp loop?\n");
          errs() << "SUSAN: omploop:" << *ompLoop << "\n";
          currLP->L = ompLoop;
          currLP->LoopHeader = ompLoop ? ompLoop->getHeader() : nullptr;
          currLP->isOmpLoop = true;
          currLP->barrier = false;
          countBarrier = 0;
          currLP->IV = getInductionVariable(ompLoop);
          currLP->IVInc = getIVIncrement(currLP->L, currLP->IV);
          currLP->isForLoop = true;
          currLP->nestlevel = -1;
          LoopProfiles.insert(currLP);
        }

        /*
         * OpenMP: search for barrier call
         */
        if (ompCall->getName().contains("__kmpc_barrier")) {
          errs() << "SUSAN: barrier call!!\n";
          countBarrier++;
          if (countBarrier > 1)
            currLP->barrier = true;
        }
      }
    }
  }

  // CreateOmpLoops(ompLoop, ub, lb, incr);
  // omp_searchForUsesToDelete(values2delete, F);
}

Value *CWriter::findOriginalValue(Value *val) {
  Instruction *valInst = dyn_cast<Instruction>(val);
  if (!valInst)
    return val;

  Value *newVal = val;

  if (isa<AllocaInst>(val))
    for (auto user : val->users())
      if (StoreInst *store = dyn_cast<StoreInst>(user))
        if (store->getPointerOperand() == val)
          newVal = store->getOperand(0);

  std::set<Value*> visited;
  while (isa<CastInst>(newVal) || isa<LoadInst>(newVal) ||
         (isa<PHINode>(newVal) && !isInductionVariable(newVal) &&
          !isExtraInductionVariable(newVal))) {
    if (!visited.insert(newVal).second) break; // Cycle detected

    Instruction *currInst = cast<Instruction>(newVal);
    if (isa<CastInst>(newVal) || isa<LoadInst>(newVal))
      newVal = currInst->getOperand(0);
    else if (isa<PHINode>(newVal)) {
      PHINode *phi = dyn_cast<PHINode>(currInst);
      if (phi->getNumIncomingValues() > 0)
        newVal = phi->getIncomingValue(0); // Take first incoming value as representative heuristic
      else
        break;
      // Note: The previous logic iterated all incoming values and took the last one.
      // This seems arbitrary and likely unintended. Taking the first one is simpler and deterministic.
      // Ideally we would search for a non-PHI source, but without context (e.g. which path taken), 
      // traversing PHIs statically is ambiguous.
    }
  }

  valInst = dyn_cast<Instruction>(newVal);
  if (!valInst)
    return newVal;

  if (deleteAndReplaceInsts.find(valInst) != deleteAndReplaceInsts.end())
    newVal = deleteAndReplaceInsts[valInst];
  return newVal;
}

bool CWriter::isReturnOrExit(BasicBlock *BB) {
  Instruction *term = BB->getTerminator();
  if (isa<UnreachableInst>(term))
    return true;
  if (isa<ReturnInst>(term))
    return true;
  // branch to a return-only block
  if(auto* branch = dyn_cast<BranchInst>(term)) {
    if(branch->isUnconditional()) {
      BasicBlock* succ = branch->getSuccessor(0);
      // Check if it is a return-only block
      if(isa<ReturnInst>(succ->getFirstNonPHIOrDbgOrLifetime()))
        return true;
    }
  }
  return false;
}

LoopType CWriter::getLoopType(Loop* loop) {
  // TODO: assume single exit block for now...
  errs() << "Getting loop type...\n";
  BasicBlock* exitBB = getSingleExitBlock(loop);
  BasicBlock* latchBB = loop->getLoopLatch();
  BasicBlock* exitPredBB = exitBB->getUniquePredecessor();
  BasicBlock* entryBlock = loop->getHeader();
  assert(exitPredBB && "No unique predecessor of loop exit!!\n");
  if(exitPredBB == latchBB)
    return doWhileLoop;

  BranchInst* brInst = dyn_cast<BranchInst>(entryBlock->getTerminator());
  // Handle loop rotation patterns where header has unconditional branch
  // This can indicate a while loop with continue statements (outer/inner pattern)
  if (!brInst || !brInst->isConditional()) {
    errs() << "ANDREW: Loop header has unconditional branch\n";
    
    // Check if this is a while-with-continue pattern:
    // Header (while.cond.outer) has unconditional branch to condition block (while.cond)
    // The condition block contains the actual loop condition
    if (brInst && brInst->isUnconditional()) {
      BasicBlock *condBlock = brInst->getSuccessor(0);
      if (loop->contains(condBlock) && condBlock != latchBB) {
        BranchInst *condBrInst = dyn_cast<BranchInst>(condBlock->getTerminator());
        if (condBrInst && condBrInst->isConditional()) {
          // Check if one successor exits the loop (the exit block or leads to it)
          BasicBlock *succ0 = condBrInst->getSuccessor(0);
          BasicBlock *succ1 = condBrInst->getSuccessor(1);
          bool succ0ExitsLoop = !loop->contains(succ0) || succ0 == exitBB;
          bool succ1ExitsLoop = !loop->contains(succ1) || succ1 == exitBB;
          
          if (succ0ExitsLoop || succ1ExitsLoop) {
            errs() << "ANDREW: Detected whileLoopWithContinue pattern\n";
            errs() << "  Header: " << entryBlock->getName() << "\n";
            errs() << "  CondBlock: " << condBlock->getName() << "\n";
            return whileLoopWithContinue;
          }
        }
      }
    }
    
    // Fallback to doWhileLoop
    errs() << "ANDREW: treating as doWhileLoop\n";
    return doWhileLoop;
  }

  errs() << *brInst->getCondition() << "\n";
  CmpInst* cmp = dyn_cast<CmpInst>(brInst->getCondition());
  if(!cmp) return whileLoop;

  if(auto IV = getInductionVariable(loop)) {
    errs() << "Found IV!\n\t"<< *IV << "\n";
    if(cmp->getOperand(0) == IV || cmp->getOperand(1) == IV)
      return forLoop;
    // TODO: check for more than one level of casting
    // Does this happen??
    if(auto castOp0 = dyn_cast<CastInst>(cmp->getOperand(0)))
      if(castOp0->getOperand(0) == IV)
        return forLoop;
    if(auto castOp1 = dyn_cast<CastInst>(cmp->getOperand(1)))
      if(castOp1->getOperand(0) == IV)
        return forLoop;
  }

  return whileLoop;
}

BasicBlock* CWriter::getSingleExitBlock(Loop *L) {
  if(auto* singleExitBB = L->getExitBlock())
    return singleExitBB;

  SmallVector<BasicBlock*, 2> exitBBs;
  L->getExitBlocks(exitBBs);

  BasicBlock* exitBlock = nullptr;
  unsigned numExitBBs = 0;
  for(auto* exitBB: exitBBs) {
    errs() << *exitBB << "\n";
    if(BasicBlock* exitPredBB = exitBB->getUniquePredecessor()) {
        if(exitPredBB == L->getLoopLatch() || exitPredBB == L->getHeader()) {
          exitBlock = exitBB;
          numExitBBs++;
        }
      }
    else
     errs() << "Multiple preds\n";
    //if(isReturnOrExit(exitBB)) continue;
    //numExitBBs++;
    //exitBlock = exitBB;
  }
  assert(exitBlock && "No valid exit found for the loop!!\n");
  assert(numExitBBs == 1 && "Multiple valid exits that are not return/unreachables!!\n");
  return exitBlock;
}

void CWriter::preprocessLoopProfiles(Function &F) {
  errs() << "In preprocessLoopProfiles(" << F.getName() << ")\n";
  std::list<Loop *> loops(LI->begin(), LI->end());

  while (!loops.empty()) {
    Loop *L = loops.front();
    loops.pop_front();

    bool skipLoop = false;
    for (auto LI : LoopProfiles)
      if (LI->L == L && LI->isOmpLoop) {
        skipLoop = true;
        break;
      }

    if (skipLoop) {
      errs() << "SUSAN: skipping omp loop: " << *L << "\n";
      loops.insert(loops.end(), L->getSubLoops().begin(),
                   L->getSubLoops().end());
      continue;
    }

    errs() << "Not skipping " << L->getName() << "\n";
    PHINode *IV = getInductionVariable(L);
    // handle case where loop is while but has an indvar
    errs() << "Header and latch\n";
    errs() << *L->getHeader() << *L->getLoopLatch() << "\n";
    BranchInst* lBr = dyn_cast<BranchInst>(L->getHeader()->getTerminator());
    bool cmpWithIV = false;
    if(lBr && lBr->isConditional()) {
      if(CmpInst* lCmp = dyn_cast<CmpInst>(lBr->getCondition())) {
        errs() << "Found lCmp: " << *lCmp << "\n";
        if (IV) {
          errs() << "IV: " << *IV << "\n";
          errs() << "Operands: " << *(lCmp->getOperand(0)) << " , "
                 << *(lCmp->getOperand(1)) << "\n";
          if(lCmp->getOperand(0) == IV || lCmp->getOperand(1) == IV)
              cmpWithIV = true;
          else if(auto castOp0 = dyn_cast<CastInst>(lCmp->getOperand(0)))
            if(castOp0->getOperand(0) == IV)
              cmpWithIV = true;
          else if(auto castOp1 = dyn_cast<CastInst>(lCmp->getOperand(1)))
            if(castOp1->getOperand(0) == IV)
              cmpWithIV = true;
        } else {
          errs() << "IV: null (while loop without induction variable)\n";
        }
      }
    }
    // it should be enough to just check whether the exit condition is from IV
    bool isaForLoop = cmpWithIV;
    if (!isaForLoop) {
      errs() << "SUSAN: recording while loop profile:" << *L << "\n";
      LoopProfile *LP = new LoopProfile();
      LP->isForLoop = false;
      LP->L = L;
      LP->LoopHeader = L ? L->getHeader() : nullptr;
      LP->IV = nullptr;
      LP->IVInc = nullptr;
      LP->ub = nullptr;
      LP->lb = nullptr;
      LP->incr = nullptr;
      LP->lbAlloca = nullptr;
      bool negateCondition = false;
      Instruction *condInst = findCondInst(LP->L, negateCondition);
      if (condInst)
        errs() << "while loop condInst" << *condInst << "\n";
      LoopProfiles.insert(LP);

      loops.insert(loops.end(), L->getSubLoops().begin(),
                   L->getSubLoops().end());
      continue;
    }
    errs() << "YEBIN: We are here\n";

    LoopProfile *LP = new LoopProfile();
    LP->isForLoop = true;
    LP->L = L;
    LP->LoopHeader = L ? L->getHeader() : nullptr;
    LP->IV = IV;
    LP->IVInc = getIVIncrement(L, IV);

    if (LI->getLoopFor(IV->getIncomingBlock(0)) != L)
      LP->lb = IV->getIncomingValue(0);
    else if ((LI->getLoopFor(IV->getIncomingBlock(0)) == L))
      LP->incr = IV->getIncomingValue(0);

    if (LI->getLoopFor(IV->getIncomingBlock(1)) != L)
      LP->lb = IV->getIncomingValue(1);
    else if ((LI->getLoopFor(IV->getIncomingBlock(1)) == L))
      LP->incr = IV->getIncomingValue(1);

    bool negateCondition = false;
    Instruction *condInst = findCondInst(LP->L, negateCondition);
    Value *ub = condInst->getOperand(1);
    // LP->ub = findOriginalValue(ub);
    LP->ub = ub;
    errs() << "none omp loop ub: " << *LP->ub << "\n";

    LP->lbAlloca = nullptr;
    // LP->ub = nullptr; //note: ub is included in condinst unless it is a omp
    // loop
    LP->isOmpLoop = false;

    LP->nestlevel = -1;
    LoopProfiles.insert(LP);

    loops.insert(loops.end(), L->getSubLoops().begin(), L->getSubLoops().end());
  }

  errs() << "=========LOOP PROFILES=========\n";
  for (auto LP : LoopProfiles) {
    if (!LP->isForLoop)
      continue;
    errs() << "Loop: " << *LP->L << "\n";
    errs() << "isomp: " << LP->isOmpLoop << "\n";
    // errs() << "ub: " << *LP->ub << "\n";
  }
}

void CWriter::removeBranchTarget(BranchInst *br, int destIdx) {
  errs() << "SUSAN: removing branch target: " << *br << "\n";
  IRBuilder<> brBuilder(br);
  std::set<BasicBlock *> succBB2Remove;
  for (auto succ = succ_begin(br); succ != succ_end(br); ++succ) {
    BasicBlock *succBB = *succ;
    if (br->getSuccessor(destIdx) == succBB)
      continue;
    for (pred_iterator PI = pred_begin(succBB), E = pred_end(succBB); PI != E;
         ++PI)
      if (*PI == br->getParent())
        succBB2Remove.insert(succBB);
    errs() << "SUSAN: inserting succBB: " << succBB->getName() << "\n";
  }
  for (auto succBB : succBB2Remove) {
    errs() << "SUSAN: removing succBB" << *succBB << "\n";
    succBB->removePredecessor(br->getParent());
  }

  Value *newBr = brBuilder.CreateBr(br->getSuccessor(destIdx));
}

void CWriter::preprocessSkippableBranches(Function &F) {
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    BranchInst *br = dyn_cast<BranchInst>(&*I);
    if (!br)
      continue;
    if (!br->isConditional())
      continue;
    Loop *L = LI->getLoopFor(br->getParent());
    if (L && L->getLoopLatch()->getTerminator() == br)
      continue;
    ICmpInst *cmp = dyn_cast<ICmpInst>(br->getCondition());
    if (!cmp)
      continue;

    errs() << "Preprocessing branch \n\t" << *br << "\nwith comparitor\n\t" << *cmp << "\nin " << F.getName() << "\n"; 

    Value *opnd0 = cmp->getOperand(0);
    Value *opnd1 = cmp->getOperand(1);
    errs() << "SUSAN: opnd0" << *opnd0 << "\n";
    errs() << "SUSAN: opnd1" << *opnd1 << "\n";
    if (isInductionVariable(opnd0) || isInductionVariable(opnd1))
      continue;

    opnd0 = findOriginalValue(opnd0);
    opnd1 = findOriginalValue(opnd1);

    // check if it's a call to omp master
    bool isMasterCall = false;
    if (CallInst *CI = dyn_cast<CallInst>(opnd0))
      if (Function *ompCall = CI->getCalledFunction())
        if (ompCall->getName().contains("__kmpc_master"))
          if (cmp->getPredicate() == CmpInst::ICMP_EQ)
            isMasterCall = true;
    if (isMasterCall) {
      deadBranches[br] = 1;
      continue;
    }

    for (auto LP : LoopProfiles) {
      errs() << "Checking LP " << LP->L->getName() << "\n";
      if (LP->isForLoop) {
        errs() << "It's a for loop!\n";
        Value *UpperBound = findOriginalValue(LP->ub);
        errs() << "SUSAN: LP->ub: " << *LP->ub << "\n";
        // if(LoadInst *ldUB = dyn_cast<LoadInst>(LP->ub))
        //   UpperBound = ldUB->getPointerOperand();
        errs() << "SUSAN: upperbound: " << *LP->ub << "\n";

        // if(UpperBound == opnd0){
        //    if ((cmp->getPredicate() == CmpInst::ICMP_SGT
        //         || cmp->getPredicate() == CmpInst::ICMP_UGT)){
        //     errs() << "SUSAN: deadbranch: " << *br << "\n";
        //     deadBranches[br] = 0;
        //    }
        // }
        // else if(UpperBound == opnd1){
        //   bool negateCondition = false;
        //   Instruction *condInst = findCondInst(LP->L, negateCondition);
        //   if(cmp == condInst) continue;

        //  if (cmp->getPredicate() == CmpInst::ICMP_SLT
        //      || cmp->getPredicate() == CmpInst::ICMP_ULT){
        //    deadBranches[br] = 0;
        //  }
        //}
        // else if(LP->lbAlloca == opnd0
        //    && (cmp->getPredicate() == CmpInst::ICMP_SGT
        //        || cmp->getPredicate() == CmpInst::ICMP_UGT)){
        //   Value *opnd1 = cmp->getOperand(1);
        //   if(isa<SelectInst>(opnd1)){
        //     deadBranches[br] = 1;
        //   }
        //}
      } else {
        errs() << "Not a for loop\n";
        bool negateCondition;
        Instruction *condInst = findCondInst(LP->L, negateCondition);
        if (!condInst) {
          errs() << "condInst is null, skipping\n";
          continue;
        }
        errs() << *condInst << "\n";
        if(!isa<CmpInst>(condInst)) continue;
        Value *loopCondOpnd0 = condInst->getOperand(0);
        Value *loopCondOpnd1 = condInst->getOperand(1);
        if (loopCondOpnd1 != opnd1 && loopCondOpnd0 != opnd0)
          continue;

        bool isDoWhileReversed = false;
        if (loopCondOpnd1 == opnd1) {
          for (User *user : opnd0->users()) {
            PHINode *userPhi = dyn_cast<PHINode>(user);
            if (!userPhi)
              continue;
            for (unsigned i = 0; i < userPhi->getNumIncomingValues(); ++i)
              if (userPhi->getIncomingValue(i) == loopCondOpnd0)
                isDoWhileReversed = true;
          }
        }

        if (!isDoWhileReversed)
          continue;

        if (!negateCondition) {
          errs() << "SUSAN: found deadbranch for while loop: " << *br << "\n";
          deadBranches[br] = 0;
        } else {
          deadBranches[br] = 1;
        }
      }
    }
  }

  for (auto [branch, dest] : deadBranches)
    removeBranchTarget(branch, dest);
  for (auto [branch, dest] : deadBranches)
    branch->eraseFromParent();

  for (auto &BB : F) {
    errs() << "SUSAN: BB:" << BB << "\n";
  }
}

void CWriter::preprocessSkippableInsts(Function &F) {
  // skip ands with xFFFFFFFF
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    BinaryOperator *binop = dyn_cast<BinaryOperator>(&*I);
    if (!binop)
      continue;

    auto opcode = binop->getOpcode();
    if (opcode != Instruction::And)
      continue;

    if (ConstantInt *consInt = dyn_cast<ConstantInt>(binop->getOperand(0)))
      if (consInt->getZExtValue() == 4294967295 ||
          consInt->getZExtValue() == 34359738360) {
        deleteAndReplaceInsts[&*I] = binop->getOperand(1);
        continue;
      }

    if (ConstantInt *consInt = dyn_cast<ConstantInt>(binop->getOperand(1))) {
      if (consInt->getZExtValue() == 4294967295 ||
          consInt->getZExtValue() == 34359738360) {
        errs() << "SUSAN: resgitering deleteAndReplaceInsts: " << *I << "\n";
        deleteAndReplaceInsts[&*I] = binop->getOperand(0);
        continue;
      }
    }
  }
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    LoadInst *ld = dyn_cast<LoadInst>(&*I);
    if (!ld)
      continue;
    if (ld->getPointerOperand()->getName() == "stderr") {
      errs() << "SUSAN: added stderr to delete insts\n";
      deleteAndReplaceInsts[&*I] = ld->getPointerOperand();
    }
  }
}

void CWriter::EliminateDeadInsts(Function &F) {
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    Instruction *inst = &*I;
    if (isAtomicAddLoweringCandidate(inst))
      continue;

    if (CallInst *CI = dyn_cast<CallInst>(inst)) {
      errs() << "SUSAN: CI at 1400: " << *CI << "\n";
      if (Function *F = CI->getCalledFunction()) {
        if (F->getName() == "malloc") {
          errs() << "SUSAN: found malloc 1403: " << *inst << "\n";
          for (User *U : inst->users())
            if (isa<StoreInst>(U)) {
              errs() << "SUSAN: found storeinst 1404: " << *U << "\n";
              //deadInsts.insert(cast<Instruction>(U));
            }
        } else if (F->getName() == "strtol") {
          for (User *U : inst->users())
            if (isa<TruncInst>(U)) {
              TruncInst *trunc = cast<TruncInst>(U);
              for (User *truncU : trunc->users())
                if (isa<StoreInst>(truncU)) {
                  errs() << "SUSAN: dead trunc: " << *truncU << "\n";
                  deadInsts.insert(cast<Instruction>(truncU));
                  StoreInst *store = cast<StoreInst>(truncU);
                  Instruction *ptrOpnd =
                      dyn_cast<Instruction>(store->getPointerOperand());
                  if (!ptrOpnd)
                    continue;
                  deadInsts.insert(ptrOpnd);
                }
            }
        }
      }
    } else if (IS_OPENMP_FUNCTION && isa<ReturnInst>(inst)) {
      errs() << "SUSAN: add return to deadinst: " << *inst << "\n";
      deadInsts.insert(inst);
    }

    if (inst->getName().contains("kmpc_loc")) {
      errs() << "SUSAN: kmpc_loc found!!!\n";
      deadInsts.insert(inst);

      std::set<Instruction *> visited;
      std::queue<Instruction *> toVisit;
      visited.insert(inst);
      toVisit.push(inst);
      deadInsts.insert(inst);

      while (!toVisit.empty()) {
        Instruction *currInst = toVisit.front();
        toVisit.pop();

        for (User *U : currInst->users()) {
          Instruction *userInst = dyn_cast<Instruction>(U);
          if (userInst && visited.find(userInst) == visited.end()) {
            visited.insert(userInst);
            toVisit.push(userInst);
            CallInst *CI = dyn_cast<CallInst>(userInst);
            if (CI && ompFuncs.find(CI) != ompFuncs.end())
              continue;
            deadInsts.insert(userInst);
          }
        }
      }
      continue;
    }
  }
  // first record all the liveins for openmp functions
  // any instructions that's not in an omp loop or not an livein should be
  // eliminated
  if (IS_OPENMP_FUNCTION) {
    for (auto LP : LoopProfiles)
      if (LP->isForLoop)
        OMP_RecordLiveIns(LP);

    errs() << "==========omp liveins========\n";
    for (auto [L, liveins] : omp_liveins) {
      errs() << "loop: " << *L << "\n";
      for (auto livein : liveins)
        errs() << "livein: " << *livein << "\n";
    }
    errs() << "==========omp liveins end========\n";

    for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
      Instruction *inst = &*I;
      if (isAtomicAddLoweringCandidate(inst))
        continue;

      bool isLoopLiveIn = false;
      for (auto [loop, liveins] : omp_liveins)
        if (liveins.find(inst) != liveins.end()) {
          errs() << "SUSAN: livein: " << *inst << "\n";
          isLoopLiveIn = true;
          break;
        }
      if (isLoopLiveIn)
        continue;
      Loop *L = LI->getLoopFor(inst->getParent());
      bool isOmpLoop = false;
      for (auto LP : LoopProfiles) {
        if (LP->L == L && LP->isOmpLoop) {
          isOmpLoop = true;
        }
      }
      if (isOmpLoop)
        continue;

      bool nestedInOmpLoop = false;
      while (L) {
        for (auto LP : LoopProfiles) {
          if (LP->L == L && LP->isOmpLoop) {
            errs() << "nested in omploop:" << *inst << "\n";
            nestedInOmpLoop = true;
            break;
          }
        }
        L = L->getParentLoop();
      }
      if (nestedInOmpLoop)
        continue;

      errs() << "SUSAN: adding to deadInsts" << *inst << "\n";
      deadInsts.insert(inst);
    }

    for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
      PHINode *phi = dyn_cast<PHINode>(&*I);
      if (!phi)
        continue;

      bool isDead = true;
      for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
        Instruction *val = dyn_cast<Instruction>(phi->getIncomingValue(i));
        if (!val) {
          isDead = false;
          break;
        }
        if (deadInsts.find(val) == deadInsts.end()) {
          isDead = false;
          break;
        }
      }

      if (isDead) {
        errs() << "SUSAN: add phi to deadInst: " << *I << "\n";
        deadInsts.insert(&*I);
      }
    }
  }
}

void CWriter::FindInductionVariableRelationships() {
  std::list<Loop *> loops(LI->begin(), LI->end());
  while (!loops.empty()) {
    Loop *L = loops.front();
    loops.pop_front();

    BasicBlock *header = L->getHeader();
    PHINode *headIV = nullptr;
    for (auto &I : *header) {
      PHINode *phi = dyn_cast<PHINode>(&I);
      if (!phi)
        break;
      if (isInductionVariable(phi)) {
        headIV = phi;
        break;
      }
    }

    if (!headIV) {
      loops.insert(loops.end(), L->getSubLoops().begin(),
                   L->getSubLoops().end());
      continue;
    }

    for (auto &I : *header) {
      PHINode *phi = dyn_cast<PHINode>(&I);
      if (!phi)
        break;
      if (isExtraInductionVariable(phi))
        IVMap[headIV].insert(phi);
    }

    loops.insert(loops.end(), L->getSubLoops().begin(), L->getSubLoops().end());
  }

  errs() << "========== IV MAP==========\n";
  for (auto [phi, extraPHIs] : IVMap) {
    errs() << "SUSAN: headPHI: " << *phi << "\n";
    for (auto extraPhi : extraPHIs)
      errs() << "SUSAN: phi: " << *extraPhi << "\n";
  }
}

void CWriter::preprocessIVIncrements() {
  std::list<Loop *> loops(LI->begin(), LI->end());
  while (!loops.empty()) {
    Loop *L = loops.front();
    loops.pop_front();
    LoopProfile *LP = findLoopProfile(L);
    IVInc2IV[LP->IVInc] = LP->IV;
    loops.insert(loops.end(), L->getSubLoops().begin(), L->getSubLoops().end());
  }
}

bool CWriter::hasHigherOrderOps(Instruction *I,
                                std::set<unsigned> higherOrderOpcodes) {

  std::queue<Instruction *> toVisit;
  std::set<Instruction *> visited;
  toVisit.push(I);
  visited.insert(I);

  while (!toVisit.empty()) {
    Instruction *currInst = toVisit.front();
    toVisit.pop();

    errs() << "currInst :" << *currInst << "\n";
    for (auto op : higherOrderOpcodes) {
      if (currInst->getOpcode() == op)
        return true;
    }

    if (!isInlinableInst(*currInst))
      break;

    for (User *U : currInst->users()) {
      if (Instruction *inst = dyn_cast<Instruction>(U)) {
        if (visited.find(inst) == visited.end()) {
          visited.insert(inst);
          toVisit.push(inst);
        }
      }
    }
  }
  return false;
}

void CWriter::preprocessInsts2AddParenthesis(Function &F) {
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    std::set<unsigned> level5operators, level4operators, level3operators,
        level2operators;
    level5operators.insert(
        {Instruction::Shl, Instruction::LShr, Instruction::AShr});
    level4operators.insert({Instruction::Add, Instruction::Sub});
    level3operators.insert({Instruction::Mul, Instruction::FMul,
                            Instruction::SDiv, Instruction::UDiv,
                            Instruction::FDiv, Instruction::URem,
                            Instruction::SRem, Instruction::FRem});
    level2operators.insert(
        {Instruction::FNeg, Instruction::GetElementPtr, Instruction::Load});

    if (!isInlinableInst(*I))
      continue;
    if (Instruction *op = dyn_cast<Instruction>(&*I)) {
      auto opcode = op->getOpcode();
      std::set<unsigned> higherOrderOpcodes;
      if (level3operators.find(opcode) != level3operators.end()) {
        higherOrderOpcodes.insert(level2operators.begin(),
                                  level2operators.end());
      } else if (level4operators.find(opcode) != level4operators.end()) {
        higherOrderOpcodes.insert(level2operators.begin(),
                                  level2operators.end());
        higherOrderOpcodes.insert(level3operators.begin(),
                                  level3operators.end());
      } else if (level5operators.find(opcode) != level5operators.end()) {
        higherOrderOpcodes.insert(level2operators.begin(),
                                  level2operators.end());
        higherOrderOpcodes.insert(level3operators.begin(),
                                  level3operators.end());
        higherOrderOpcodes.insert(level4operators.begin(),
                                  level4operators.end());
      }

      if (hasHigherOrderOps(&*I, higherOrderOpcodes) && !isIVIncrement(&*I)) {
        errs() << "SUSAN: add () to inst: " << *op << "\n";
        addParenthesis.insert(op);
      }
    }
  }
}

void CWriter::buildIVNames() {
  // for(auto [callInst, utask] : ompFuncs){
  //   auto L = LI->getLoopFor(callInst->getParent());
  //   if(!L) continue;
  //   errs() << "SUSAN: found loop over callInst " << *callInst << "\n";
  //   errs() << "depth " << L->getLoopDepth() << "\n";
  //   errs() << "SUSAN: utask: " << *utask << "\n";
  //   FunctionTopLoopLevels[utask] = L->getLoopDepth();
  // }

  std::map<Function *, std::set<std::string>> ReservedNamesByFunction;
  auto reserveSourceNames = [&](Function *F) {
    auto &Reserved = ReservedNamesByFunction[F];
    for (auto &Arg : F->args()) {
      if (!Arg.getName().empty())
        Reserved.insert(Arg.getName().str());
    }
    for (auto &BB : *F) {
      for (auto &I : BB) {
        if (!I.getName().empty())
          Reserved.insert(I.getName().str());
      }
    }
  };
  auto allocateUniqueIVName = [](std::set<std::string> &Reserved,
                                 const std::string &Base) {
    std::string Candidate = Base;
    if (Candidate.empty())
      Candidate = "iv";
    if (Reserved.find(Candidate) == Reserved.end()) {
      Reserved.insert(Candidate);
      return Candidate;
    }

    Candidate = Base + "_iv";
    if (Reserved.find(Candidate) == Reserved.end()) {
      Reserved.insert(Candidate);
      return Candidate;
    }

    unsigned Suffix = 1;
    while (true) {
      std::string Numbered = Candidate + "_" + utostr(Suffix++);
      if (Reserved.find(Numbered) == Reserved.end()) {
        Reserved.insert(Numbered);
        return Numbered;
      }
    }
  };

  for (auto LP : LoopProfiles) {
    if (!LP->isForLoop || !LP->IV)
      continue;
    errs() << "LP->LV 1694: " << *(LP->IV) << "\n";
    errs() << "LP->L 1694: " << *(LP->L) << "\n";

    Function *ParentF = LP->L->getHeader()->getParent();
    if (ReservedNamesByFunction.find(ParentF) == ReservedNamesByFunction.end())
      reserveSourceNames(ParentF);
    auto &Reserved = ReservedNamesByFunction[ParentF];

    char nestLevel = LP->L->getLoopDepth() - 1;
    std::string BaseName(1, static_cast<char>('i' + nestLevel));
    std::string Name = allocateUniqueIVName(Reserved, BaseName);
    errs() << "nestlevel: " << Name << "\n";
    IV2Name[LP->IV] = Name;
    if (LP->IVInc)
      IV2Name[LP->IVInc] = Name;
  }
}

void CWriter::collectNotInlinableBinOps(Function &F) {
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    PHINode *phi = dyn_cast<PHINode>(&*I);
    if (!phi)
      continue;
    for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
      BinaryOperator *incomingI =
          dyn_cast<BinaryOperator>(phi->getIncomingValue(i));
      if (incomingI)
        notInlinableBinOps.insert(incomingI);
    }
  }
}

void CWriter::findDoubleGEP(Function &F) {
  for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
    LoadInst *ld = dyn_cast<LoadInst>(&*I);
    StoreInst *st = dyn_cast<StoreInst>(&*I);
    if (!ld && !st)
      continue;
    if (ld) {
      if (GetElementPtrInst *gep =
              dyn_cast<GetElementPtrInst>(ld->getPointerOperand())) {
        auto indices = gep->getNumIndices();
        if (GetElementPtrInst *gep2 =
                dyn_cast<GetElementPtrInst>(gep->getPointerOperand())) {
          auto indices2 = gep2->getNumIndices();
          if (indices == indices2 && indices == 1)
            errs() << "SUSAN: found double gep: 1703: " << *ld << "\n";
          errs() << "SUSAN: gep 1704: " << *gep << "\n";
          errs() << "SUSAN: gep2 1705: " << *gep2 << "\n";
          PointerType *ptrTy =
              dyn_cast<PointerType>(gep2->getPointerOperand()->getType());
          // if(!isa<GetElementPtrInst>(gep2->getPointerOperand()) &&
          //     (valuesCast2Double.find(gep2->getPointerOperand()) ==
          //     valuesCast2Double.end()) &&
          //     !ptrTy->getPointerElementType()->isDoubleTy())
          doubleGeps[ld] = gep2->getPointerOperand();
        }
      }
    } else {
      if (GetElementPtrInst *gep =
              dyn_cast<GetElementPtrInst>(st->getPointerOperand())) {
        auto indices = gep->getNumIndices();
        if (GetElementPtrInst *gep2 =
                dyn_cast<GetElementPtrInst>(gep->getPointerOperand())) {
          auto indices2 = gep2->getNumIndices();
          if (indices == indices2 && indices == 1)
            errs() << "SUSAN: found double gep: 1713: " << *st << "\n";
          errs() << "SUSAN: gep 1715: " << *gep << "\n";
          errs() << "SUSAN: gep2 1716: " << *gep2 << "\n";
          PointerType *ptrTy =
              dyn_cast<PointerType>(gep2->getPointerOperand()->getType());
          // if(!isa<GetElementPtrInst>(gep2->getPointerOperand()) &&
          //     !ptrTy->getPointerElementType()->isDoubleTy())
          //     (valuesCast2Double.find(gep2->getPointerOperand()) ==
          //     valuesCast2Double.end()) &&
          //     !ptrTy->getPointerElementType()->isDoubleTy())
          doubleGeps[st] = gep2->getPointerOperand();
        }
      }
    }
  }
}

bool CWriter::runOnModule(Module &M) {
  cnt_totalVariables = 0;
  cnt_reconstructedVariables = 0;
  bool Modified = false;
  findOMPFunctions(M);
  std::set<Function *> functionUsed;

  // TODO: do not print any functions not being called
  for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI) {
    Function *F = &*FI;
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      if (CallInst *callInst = dyn_cast<CallInst>(&*I)) {
        functionUsed.insert(callInst->getCalledFunction());
      }
    }
  }

  for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI) {
    declaredLocals.clear();

    Function *F = &*FI;
    errs() << "CBackend: iterating function 1759: " << F->getName() << "\n";
    if (F->isIntrinsic())
      continue;
    if (F->isDeclaration())
      continue;
    if (F->getName() == "xmalloc")
      continue;
    if (functionUsed.find(F) == functionUsed.end() && F->getName() != "main")
      continue;

    IS_OPENMP_FUNCTION = false;
    for (auto [call, utask] : ompFuncs) {
      errs() << "OMP FUNC: " << *utask << "\n";
      if (utask == F)
        IS_OPENMP_FUNCTION = true;
    }
    if (IS_OPENMP_FUNCTION)
      continue;
    errs() << "CBackend: printing function 1770" << F->getName() << "\n";

    // Do not codegen any 'available_externally' functions at all, they have
    // definitions outside the translation unit.
    if (F->hasAvailableExternallyLinkage())
      continue;

    errs() << "ANDREW: CBackend: runOnModule function: " << F->getName()
           << "\n";
    Modified |= RunAllAnalysis(*F);

    // Output all floating point constants that cannot be printed accurately.
    printFloatingPointConstants(*F);
    printFunction(*F);

    LI = nullptr;
    PDT = nullptr;
  }
  errs() << "TOTAL VARIABLES: " << cnt_totalVariables << "\n";
  errs() << "RECONSTRUCTED VARIABLES: " << cnt_reconstructedVariables << "\n";
  return Modified;
}

static std::string CBEMangle(const std::string &S) {
  std::string Result;

  for (auto c : S) {
    if (isalnum(c) || c == '_') {
      Result += c;
    } else {
      Result += '_';
      Result += 'A' + (c & 15);
      Result += 'A' + ((c >> 4) & 15);
      Result += '_';
    }
  }

  return Result;
}

raw_ostream &CWriter::printTypeString(raw_ostream &Out, Type *Ty,
                                      bool isSigned) {
  if (StructType *ST = dyn_cast<StructType>(Ty)) {
    cwriter_assert(!isEmptyType(ST));
    TypedefDeclTypes.insert(Ty);

    if (!ST->isLiteral() && !ST->getName().empty()) {
      std::string Name{ST->getName()};
      return Out << "struct_" << CBEMangle(Name);
    }

    unsigned id = UnnamedStructIDs.getOrInsert(ST);
    return Out << "unnamed_" + utostr(id);
  }

  if (Ty->isPointerTy()) {
    Out << "p";
    return printTypeString(Out, Ty->getPointerElementType(), isSigned);
  }

  switch (Ty->getTypeID()) {
  case Type::VoidTyID:
    return Out << "void";
  case Type::IntegerTyID: {
    unsigned NumBits = cast<IntegerType>(Ty)->getBitWidth();
    if (NumBits == 1)
      return Out << "bool";
    else {
      cwriter_assert(NumBits <= 128 && "Bit widths > 128 not implemented yet");
      return Out << (isSigned ? "i" : "u") << NumBits;
    }
  }
  case Type::FloatTyID:
    return Out << "f32";
  case Type::DoubleTyID:
    return Out << "f64";
  case Type::X86_FP80TyID:
    return Out << "f80";
  case Type::PPC_FP128TyID:
  case Type::FP128TyID:
    return Out << "f128";

  case Type::X86_MMXTyID:
    return Out << (isSigned ? "i32y2" : "u32y2");
#if LLVM_VERSION_MAJOR > 10
  case Type::FixedVectorTyID:
  case Type::ScalableVectorTyID:
#else
  case Type::VectorTyID:
#endif
  {
    TypedefDeclTypes.insert(Ty);
    VectorType *VTy = cast<VectorType>(Ty);
    cwriter_assert(VTy->getNumElements() != 0);
    printTypeString(Out, VTy->getElementType(), isSigned);
    return Out << "x" << NumberOfElements(VTy);
  }

  case Type::ArrayTyID: {
    TypedefDeclTypes.insert(Ty);
    ArrayType *ATy = cast<ArrayType>(Ty);
    cwriter_assert(ATy->getNumElements() != 0);
    printTypeString(Out, ATy->getElementType(), isSigned);
    return Out << "a" << ATy->getNumElements();
  }

  default:
    DBG_ERRS("Unknown primitive type: " << *Ty);
    errorWithMessage("unknown primitive type");
  }
}

// YEBIN: append FIXME to all structs
std::string CWriter::getStructName(StructType *ST) {
  cwriter_assert(ST->getNumElements() != 0);
  if (!ST->isLiteral() && !ST->getName().empty()) {
    // special case for FILE type
    if(CBEMangle(ST->getName().str()) == "struct_OC__IO_FILE") return "FILE";
    return "struct __FIXME__l_struct_" + CBEMangle(ST->getName().str());
  }

  unsigned id = UnnamedStructIDs.getOrInsert(ST);
  return "struct __FIXME__l_unnamed_" + utostr(id);
}

// YEBIN: get struct fields to support FIXME
std::string CWriter::getFieldName(StructType *ST) {
  std::string StrName = getStructName(ST);
  StrName.erase(0, 7);
  return StrName + "_field";
}

std::string
CWriter::getFunctionName(FunctionType *FT,
                         std::pair<AttributeList, CallingConv::ID> PAL) {
  unsigned id = UnnamedFunctionIDs.getOrInsert(std::make_pair(FT, PAL));
  return "l_fptr_" + utostr(id);
}

std::string CWriter::getArrayName(ArrayType *AT) {
  std::string astr;
  raw_string_ostream ArrayInnards(astr);
  // Arrays are wrapped in structs to allow them to have normal
  // value semantics (avoiding the array "decay").
  cwriter_assert(!isEmptyType(AT));
  printTypeName(ArrayInnards, AT->getElementType(), false);
  return "struct __FIXME__l_array_" + utostr(AT->getNumElements()) + '_' +
         CBEMangle(ArrayInnards.str());
}

std::string CWriter::getVectorName(VectorType *VT, bool Aligned) {
  std::string astr;
  raw_string_ostream VectorInnards(astr);
  // Vectors are handled like arrays
  cwriter_assert(!isEmptyType(VT));
  if (Aligned) {
    headerUseMsAlign();
    Out << "__MSALIGN__(" << TD->getABITypeAlignment(VT) << ") ";
  }
  printTypeName(VectorInnards, VT->getElementType(), false);
  return "struct __FIXME__l_vector_" + utostr(NumberOfElements(VT)) + '_' +
         CBEMangle(VectorInnards.str());
}

static const std::string getCmpPredicateName(CmpInst::Predicate P) {
  switch (P) {
  case FCmpInst::FCMP_FALSE:
    return "0";
  case FCmpInst::FCMP_OEQ:
    return "oeq";
  case FCmpInst::FCMP_OGT:
    return "ogt";
  case FCmpInst::FCMP_OGE:
    return "oge";
  case FCmpInst::FCMP_OLT:
    return "olt";
  case FCmpInst::FCMP_OLE:
    return "ole";
  case FCmpInst::FCMP_ONE:
    return "one";
  case FCmpInst::FCMP_ORD:
    return "ord";
  case FCmpInst::FCMP_UNO:
    return "uno";
  case FCmpInst::FCMP_UEQ:
    return "ueq";
  case FCmpInst::FCMP_UGT:
    return "ugt";
  case FCmpInst::FCMP_UGE:
    return "uge";
  case FCmpInst::FCMP_ULT:
    return "ult";
  case FCmpInst::FCMP_ULE:
    return "ule";
  case FCmpInst::FCMP_UNE:
    return "une";
  case FCmpInst::FCMP_TRUE:
    return "1";
  case ICmpInst::ICMP_EQ:
    return "eq";
  case ICmpInst::ICMP_NE:
    return "ne";
  case ICmpInst::ICMP_ULE:
    return "ule";
  case ICmpInst::ICMP_SLE:
    return "sle";
  case ICmpInst::ICMP_UGE:
    return "uge";
  case ICmpInst::ICMP_SGE:
    return "sge";
  case ICmpInst::ICMP_ULT:
    return "ult";
  case ICmpInst::ICMP_SLT:
    return "slt";
  case ICmpInst::ICMP_UGT:
    return "ugt";
  case ICmpInst::ICMP_SGT:
    return "sgt";
  default:
    DBG_ERRS("Invalid icmp predicate!" << P);
    // TODO: cwriter_assert
    llvm_unreachable(0);
  }
}

static const char *getFCmpImplem(CmpInst::Predicate P) {
  switch (P) {
  case FCmpInst::FCMP_FALSE:
    return "0";
  case FCmpInst::FCMP_OEQ:
    return "X == Y";
  case FCmpInst::FCMP_OGT:
    return "X >  Y";
  case FCmpInst::FCMP_OGE:
    return "X >= Y";
  case FCmpInst::FCMP_OLT:
    return "X <  Y";
  case FCmpInst::FCMP_OLE:
    return "X <= Y";
  case FCmpInst::FCMP_ONE:
    return "X != Y && llvm_fcmp_ord(X, Y);";
  case FCmpInst::FCMP_ORD:
    return "X == X && Y == Y";
  case FCmpInst::FCMP_UNO:
    return "X != X || Y != Y";
  case FCmpInst::FCMP_UEQ:
    return "X == Y || llvm_fcmp_uno(X, Y)";
  case FCmpInst::FCMP_UGT:
    return "X >  Y || llvm_fcmp_uno(X, Y)";
    return "ugt";
  case FCmpInst::FCMP_UGE:
    return "X >= Y || llvm_fcmp_uno(X, Y)";
  case FCmpInst::FCMP_ULT:
    return "X <  Y || llvm_fcmp_uno(X, Y)";
  case FCmpInst::FCMP_ULE:
    return "X <= Y || llvm_fcmp_uno(X, Y)";
  case FCmpInst::FCMP_UNE:
    return "X != Y";
  case FCmpInst::FCMP_TRUE:
    return "1";
  default:
    DBG_ERRS("Invalid fcmp predicate!" << P);
    // TODO: cwriter_assert
    llvm_unreachable(0);
  }
}

static void defineFCmpOp(raw_ostream &Out, CmpInst::Predicate const P) {
  Out << "static __forceinline int llvm_fcmp_" << getCmpPredicateName(P)
      << "(double X, double Y) { ";
  Out << "return " << getFCmpImplem(P) << "; }\n";
}

void CWriter::headerUseFCmpOp(CmpInst::Predicate P) {
  switch (P) {
  case FCmpInst::FCMP_ONE:
    FCmpOps.insert(CmpInst::FCMP_ORD);
    break;
  case FCmpInst::FCMP_UEQ:
  case FCmpInst::FCMP_UGT:
  case FCmpInst::FCMP_UGE:
  case FCmpInst::FCMP_ULT:
  case FCmpInst::FCMP_ULE:
    FCmpOps.insert(CmpInst::FCMP_UNO);
    break;
  default:
    break;
  }
  FCmpOps.insert(P);
}

raw_ostream &CWriter::printSimpleType(raw_ostream &Out, Type *Ty,
                                      bool isSigned) {
  cwriter_assert((Ty->isSingleValueType() || Ty->isVoidTy()) &&
                 "Invalid type for printSimpleType");
  switch (Ty->getTypeID()) {
  case Type::VoidTyID:
    return Out << "void";
  case Type::IntegerTyID: {
    unsigned NumBits = cast<IntegerType>(Ty)->getBitWidth();
    if (NumBits == 1)
      return Out << "bool";
    else if (NumBits <= 8)
      return Out << (isSigned ? "int8_t" : "uint8_t");
    else if (NumBits <= 16)
      return Out << (isSigned ? "int16_t" : "uint16_t");
    else if (NumBits <= 32)
      return Out << (isSigned ? "int32_t" : "uint32_t");
    else if (NumBits <= 64)
      return Out << (isSigned ? "int64_t" : "uint64_t");
    else {
      cwriter_assert(NumBits <= 128 && "Bit widths > 128 not implemented yet");
      return Out << (isSigned ? "int128_t" : "uint128_t");
    }
  }
  case Type::FloatTyID:
    return Out << "float";
  case Type::DoubleTyID:
    return Out << "double";
  // Lacking emulation of FP80 on PPC, etc., we assume whichever of these is
  // present matches host 'long double'.
  case Type::X86_FP80TyID:
  case Type::PPC_FP128TyID:
  case Type::FP128TyID:
    return Out << "long double";

  case Type::X86_MMXTyID:
    return Out << (isSigned ? "int32_t" : "uint32_t")
               << " __attribute__((vector_size(8)))";

  default:
    DBG_ERRS("Unknown primitive type: " << *Ty);
    errorWithMessage("unknown primitive type");
  }
}

raw_ostream &CWriter::printTypeNameForAddressableValue(raw_ostream &Out,
                                                       Type *Ty,
                                                       bool isSigned) {
  // We can't directly declare a zero-sized variable in C, so we have to
  // use a single-byte type instead, in case a pointer to it is taken.
  // We can then fix the pointer type in writeOperand.
  if (!isEmptyType(Ty))
    return printTypeName(Out, Ty, isSigned);
  else
    return Out << "char /* (empty) */";
}

// Pass the Type* and the variable name and this prints out the variable
// declaration.
raw_ostream &
CWriter::printTypeName(raw_ostream &Out, Type *Ty, bool isSigned,
                       std::pair<AttributeList, CallingConv::ID> PAL) {
  if (Ty->isSingleValueType() || Ty->isVoidTy()) {
    if (!Ty->isPointerTy() && !Ty->isVectorTy())
      return printSimpleType(Out, Ty, isSigned);
  }

  if (isEmptyType(Ty))
    return Out << "void";

  switch (Ty->getTypeID()) {
  case Type::FunctionTyID: {
    FunctionType *FTy = cast<FunctionType>(Ty);
    return Out << getFunctionName(FTy, PAL);
  }
  case Type::StructTyID: {
    TypedefDeclTypes.insert(Ty);
    return Out << getStructName(cast<StructType>(Ty));
  }

  case Type::PointerTyID: {
    Type *ElTy = Ty->getPointerElementType();
    ElTy = skipEmptyArrayTypes(ElTy);
    return printTypeName(Out, ElTy, false) << '*';
  }

  case Type::ArrayTyID: {
    // TypedefDeclTypes.insert(Ty);
    Type *elTy = Ty->getArrayElementType();
    return printTypeName(Out, elTy, false);
    // return Out << getArrayName(cast<ArrayType>(Ty));
  }
#if LLVM_VERSION_MAJOR > 10
  case Type::FixedVectorTyID:
  case Type::ScalableVectorTyID:
#else
  case Type::VectorTyID:
#endif
  {
    TypedefDeclTypes.insert(Ty);
    return Out << getVectorName(cast<VectorType>(Ty), true);
  }

  default:
    DBG_ERRS("Unexpected type: " << *Ty);
    errorWithMessage("unexpected type");
  }
}

raw_ostream &CWriter::printTypeNameUnaligned(raw_ostream &Out, Type *Ty,
                                             bool isSigned) {
  if (VectorType *VTy = dyn_cast<VectorType>(Ty)) {
    // MSVC doesn't handle __declspec(align) on parameters,
    // but we specify it for Vector (hoping the compiler will vectorize it)
    // so we need to avoid it sometimes
    TypedefDeclTypes.insert(VTy);
    return Out << getVectorName(VTy, false);
  }
  return printTypeName(Out, Ty, isSigned);
}

raw_ostream &CWriter::printStructDeclaration(raw_ostream &Out,
                                             StructType *STy) {
  // The IO struct is in stdio.h, should not be exposed
  if (getStructName(STy) == "FILE")
    return Out;
  if (STy->isPacked())
    Out << "#ifdef _MSC_VER\n#pragma pack(push, 1)\n#endif\n";
  std::vector<std::string> fieldNames;
  Out << getStructName(STy) << " {\n";
  unsigned Idx = 0;
  for (StructType::element_iterator I = STy->element_begin(),
                                    E = STy->element_end();
       I != E; ++I, Idx++) {
    Out << "  ";
    bool empty = isEmptyType(*I);
    if (empty)
      Out << "/* "; // skip zero-sized types
    // printTypeName(Out, *I, false) << " field" << utostr(Idx);
    //  append struct name to field for FIXME
    printTypeName(Out, *I, false) << " " + getFieldName(STy) << utostr(Idx);
    fieldNames.push_back(getFieldName(STy)+utostr(Idx));
    ArrayType *ArrTy = dyn_cast<ArrayType>(*I);
    while (ArrTy) {
      Out << "[" << ArrTy->getNumElements() << "]";
      ArrTy = dyn_cast<ArrayType>(ArrTy->getElementType());
    }
    if (empty)
      Out << " */"; // skip zero-sized types
    else
      Out << ";\n";
  }
  Out << '}';
  
  nameDict.StructDefs.insert({getStructName(STy).substr(7), fieldNames});
  if (STy->isPacked())
    Out << " __attribute__ ((packed))";
  Out << ";\n";
  if (STy->isPacked())
    Out << "#ifdef _MSC_VER\n#pragma pack(pop)\n#endif\n";
  return Out;
}

bool CWriter::isInductionVariable(Value *V) {
  if (!V)
    return false;
  PHINode *phi = dyn_cast<PHINode>(V);
  if (!phi)
    return false;

  Loop *L = LI->getLoopFor(phi->getParent());
  if (L && getInductionVariable(L) == phi)
    return true;

  return false;
}

bool CWriter::isExtraInductionVariable(Value *V) {

  errs() << "ANDREW: isExtraInductionVariable: " << *V << "\n";
  if (!V) {
    errs() << "ANDREW: isExtraInductionVariable: V is null\n";
    return false;
  }
  PHINode *phi = dyn_cast<PHINode>(V);
  
  if (!phi) {
    errs() << "ANDREW: isExtraInductionVariable: not a PHI\n";
    return false;
  }
  Loop *L = LI->getLoopFor(phi->getParent());
  if (!L) {
    errs() << "ANDREW: isExtraInductionVariable: Loop is null\n";
    return false;
  }
  if (L && getInductionVariable(L) == phi) {
    errs() << "ANDREW: isExtraInductionVariable: is main IV. PHI: " << *phi << "\n"; 
    return false;
  }

  Type *PhiTy = phi->getType();
  if (!PhiTy->isIntegerTy() && !PhiTy->isFloatingPointTy() &&
      !PhiTy->isPointerTy()) {
    errs() << "ANDREW: isExtraInductionVariable: not int/float/ptr type\n";
    return false;
  }

  const SCEVAddRecExpr *AddRec = nullptr;
  if (SE->isSCEVable(PhiTy))
    AddRec = dyn_cast<SCEVAddRecExpr>(SE->getSCEV(phi));
  if (!AddRec || !AddRec->isAffine()) {
    errs() << "ANDREW: isExtraInductionVariable: not affine SCEV\n";
    return false;
  }

  errs() << "ANDREW: isExtraInductionVariable: found extra IV. PHI: " << *phi << "\n";
  return true;
}

bool CWriter::isExtraIVEquivalentToMainIV(PHINode *phi) {
  if (!phi)
    return false;
  Loop *L = LI->getLoopFor(phi->getParent());
  if (!L)
    return false;
  PHINode *mainIV = getInductionVariable(L);
  if (!mainIV || mainIV == phi)
    return false;
  if (!isExtraInductionVariable(phi))
    return false;
  if (!SE->isSCEVable(phi->getType()) || !SE->isSCEVable(mainIV->getType()))
    return false;

  const SCEVAddRecExpr *mainRec =
      dyn_cast<SCEVAddRecExpr>(SE->getSCEV(mainIV));
  const SCEVAddRecExpr *extraRec =
      dyn_cast<SCEVAddRecExpr>(SE->getSCEV(phi));
  if (!mainRec || !extraRec || mainRec->getLoop() != extraRec->getLoop())
    return false;

  Type *mainTy = mainRec->getType();
  Type *extraTy = extraRec->getType();
  if (!mainTy->isIntegerTy() || !extraTy->isIntegerTy())
    return false;

  // Normalize both recurrences to a common integer width so we can compare
  // loops like i64 canonical IV vs i32 companion IV (e.g. k.4) directly.
  Type *cmpTy = mainTy;
  unsigned mainBits = cast<IntegerType>(mainTy)->getBitWidth();
  unsigned extraBits = cast<IntegerType>(extraTy)->getBitWidth();
  if (extraBits > mainBits)
    cmpTy = extraTy;

  auto normalizeSCEVToCmpTy = [&](const SCEV *Expr) -> const SCEV * {
    Type *exprTy = Expr->getType();
    if (exprTy == cmpTy)
      return Expr;
    if (!exprTy->isIntegerTy())
      return nullptr;

    unsigned exprBits = cast<IntegerType>(exprTy)->getBitWidth();
    unsigned cmpBits = cast<IntegerType>(cmpTy)->getBitWidth();
    if (exprBits < cmpBits)
      return SE->getNoopOrSignExtend(Expr, cmpTy);
    if (exprBits > cmpBits)
      return SE->getTruncateOrNoop(Expr, cmpTy);
    return Expr;
  };

  const SCEV *mainStart = normalizeSCEVToCmpTy(mainRec->getStart());
  const SCEV *extraStart = normalizeSCEVToCmpTy(extraRec->getStart());
  const SCEV *mainStep =
      normalizeSCEVToCmpTy(mainRec->getStepRecurrence(*SE));
  const SCEV *extraStep =
      normalizeSCEVToCmpTy(extraRec->getStepRecurrence(*SE));
  if (!mainStart || !extraStart || !mainStep || !extraStep)
    return false;

  return mainStart == extraStart && mainStep == extraStep;
}

bool CWriter::isIVIncrement(Value *V) {
  if (!V)
    return false;
  Instruction *inst = dyn_cast<Instruction>(V);
  if (!inst)
    return false;

  Loop *L = LI->getLoopFor(inst->getParent());
  if (!L)
    return false;

  PHINode *IV = getInductionVariable(L);
  if (!IV)
    return false;
  for (unsigned i = 0; i < IV->getNumIncomingValues(); ++i) {
    BasicBlock *predBB = IV->getIncomingBlock(i);
    if (LI->getLoopFor(predBB) == L &&
        cast<Instruction>(IV->getIncomingValue(i)) == inst)
      return true;
  }

  return false;
}

bool CWriter::isExtraIVIncrement(Value *V) {
  if (!V)
    return false;
  Instruction *inst = dyn_cast<Instruction>(V);
  if (!inst)
    return false;

  Loop *L = LI->getLoopFor(inst->getParent());
  if (!L)
    return false;

  PHINode *IV = getInductionVariable(L);
  if (!IV || IVMap.find(IV) == IVMap.end())
    return false;
  for (auto relatedIV : IVMap[IV]) {
    for (unsigned i = 0; i < relatedIV->getNumIncomingValues(); ++i) {
      BasicBlock *predBB = relatedIV->getIncomingBlock(i);
      if (LI->getLoopFor(predBB) == L &&
          cast<Instruction>(relatedIV->getIncomingValue(i)) == inst)
        return true;
    }
  }

  return false;
}

raw_ostream &CWriter::printFunctionAttributes(raw_ostream &Out,
                                              AttributeList Attrs) {
  SmallVector<std::string, 5> AttrsToPrint;
  for (const auto &FnAttr : Attrs.getFnAttributes()) {
    if (FnAttr.isEnumAttribute() || FnAttr.isIntAttribute()) {
      switch (FnAttr.getKindAsEnum()) {
      case Attribute::AttrKind::AlwaysInline:
        AttrsToPrint.push_back("always_inline");
        break;
      case Attribute::AttrKind::Cold:
        AttrsToPrint.push_back("cold");
        break;
      case Attribute::AttrKind::Naked:
        AttrsToPrint.push_back("naked");
        break;
      case Attribute::AttrKind::NoDuplicate:
        AttrsToPrint.push_back("noclone");
        break;
      case Attribute::AttrKind::NoInline:
        AttrsToPrint.push_back("noinline");
        break;
      case Attribute::AttrKind::NoUnwind:
        AttrsToPrint.push_back("nothrow");
        break;
      case Attribute::AttrKind::ReadOnly:
        AttrsToPrint.push_back("pure");
        break;
      case Attribute::AttrKind::ReadNone:
        AttrsToPrint.push_back("const");
        break;
      case Attribute::AttrKind::ReturnsTwice:
        AttrsToPrint.push_back("returns_twice");
        break;
      case Attribute::AttrKind::StackProtect:
      case Attribute::AttrKind::StackProtectReq:
      case Attribute::AttrKind::StackProtectStrong:
        AttrsToPrint.push_back("stack_protect");
        break;
      case Attribute::AttrKind::AllocSize: {
        const auto AllocSize = FnAttr.getAllocSizeArgs();
        if (AllocSize.second.hasValue()) {
          AttrsToPrint.push_back(
              "alloc_size(" + std::to_string(AllocSize.first) + "," +
              std::to_string(AllocSize.second.getValue()) + ")");
        } else {
          AttrsToPrint.push_back("alloc_size(" +
                                 std::to_string(AllocSize.first) + ")");
        }
      } break;

      default:
        break;
      }
    }
    if (FnAttr.isStringAttribute()) {
      if (FnAttr.getKindAsString() == "patchable-function" &&
          FnAttr.getValueAsString() == "prologue-short-redirect") {
        AttrsToPrint.push_back("ms_hook_prologue");
      }
    }
  }
  if (!AttrsToPrint.empty()) {
    headerUseAttributeList();
    Out << " __ATTRIBUTELIST__((";
    bool DidPrintAttr = false;
    for (const auto &Attr : AttrsToPrint) {
      if (DidPrintAttr)
        Out << ", ";
      Out << Attr;
      DidPrintAttr = true;
    }
    Out << "))";
  }
  return Out;
}

raw_ostream &CWriter::printFunctionDeclaration(
    raw_ostream &Out, FunctionType *Ty,
    std::pair<AttributeList, CallingConv::ID> PAL) {
  Out << "typedef ";
  printFunctionProto(Out, Ty, PAL, getFunctionName(Ty, PAL), nullptr);
  return Out << ";\n";
}

void CWriter::printDict() {
  nameDictOut.open("dictionary.json");

  nameDictOut << "{\n";
  nameDictOut << "\"globals\": [\n";
  bool isfirst = true;
  for (auto glob: nameDict.GlobalVars) {
    if(!isfirst) nameDictOut << ", \n";
    isfirst = false;
    nameDictOut << "\t\"" << glob << "\"";
  }
  nameDictOut << "\n],\n\"struct_definitions\": {\n";
  isfirst = true;
  for (auto struc: nameDict.StructDefs) {
    if(!isfirst) nameDictOut << ",\n";
    isfirst = false;
    // Emit mapping form: "struct_definitions": { "<struct_name>": [ fields ] , ... }
    nameDictOut << "\"" << struc.first << "\": [\n";
    bool isfirstfield = true;
    for(auto field: struc.second) {
      if(!isfirstfield) nameDictOut << ", \n";
      isfirstfield = false;
      nameDictOut << "\t\"" << field << "\"";
    }
    nameDictOut << "\n]";
  }
  nameDictOut << "\n},\n\"functions\": {\n";
  isfirst = true;
  for(auto func: nameDict.LocalVars) {
    if(!isfirst) nameDictOut << ", \n";
    isfirst = false;
    nameDictOut << "\"" << func.first << "\": [\n";
    bool isfirstvar = true;
    for(auto var: func.second) {
      if(!isfirstvar) nameDictOut << ", \n";
      isfirstvar = false;
      nameDictOut << "\t\"" << var << "\"";
    }
    nameDictOut << "\n]";
  }

  nameDictOut << "\n}";
  nameDictOut << "\n}";

  nameDictOut.close();
}

// Commonly accepted types and names for main()'s return type and arguments.
static const std::initializer_list<std::pair<StringRef, StringRef>> MainArgs = {
    // Standard C return type.
    {"int", ""},
    // Standard C.
    {"int", "argc"},
    // Standard C. The canonical form is `*argv[]`, but `**argv` is equivalent
    // and more convenient here.
    {"char **", "argv"},
    // De-facto UNIX standard (not POSIX!) extra argument `*envp[]`.
    // The C standard mentions this as a "common extension".
    {"char **", "envp"},
};
// Commonly accepted argument counts for the C main() function.
static const std::initializer_list<unsigned> MainArgCounts = {
    0, // Standard C `main(void)`
    2, // Standard C `main(argc, argv)`
    3, // De-facto UNIX standard `main(argc, argv, envp)`
};

// C compilers are pedantic about the exact type of main(), and this is
// usually an error rather than a warning. Since the C backend emits unsigned
// or explicitly-signed types, it would always get the type of main() wrong.
// Therefore, we use this function to detect common cases and special-case them.
bool CWriter::isStandardMain(const FunctionType *FTy) {
  if (std::find(MainArgCounts.begin(), MainArgCounts.end(),
                FTy->getNumParams()) == MainArgCounts.end())
    return false;

  cwriter_assert(FTy->getNumContainedTypes() <= MainArgs.size());

  for (unsigned i = 0; i < FTy->getNumContainedTypes(); i++) {
    const Type *T = FTy->getContainedType(i);
    const StringRef CType = MainArgs.begin()[i].first;

    if (CType.equals("int") && !T->isIntegerTy())
      return false;

    if (CType.equals("char **") &&
        (!T->isPointerTy() || !T->getPointerElementType()->isPointerTy() ||
         !T->getPointerElementType()->getPointerElementType()->isIntegerTy(8)))
      return false;
  }

  return true;
}

// Forward declaration: definition appears near demangling helpers.
static std::string getPreferredFunctionCName(const std::string &SymbolName);

raw_ostream &CWriter::printFunctionProto(
    raw_ostream &Out, FunctionType *FTy,
    std::pair<AttributeList, CallingConv::ID> Attrs, const std::string &Name,
    iterator_range<Function::arg_iterator> *ArgList, int skipArgSteps) {
  bool shouldFixMain = (Name == "main" && isStandardMain(FTy));

  // Keep emitted C identifiers stable/safe. `Name` is already sanitized by
  // GetValueName/CBEMangle; demangled names are for comments only.
  std::string demangledName = demangleFunctionName(Name);
  std::string printedName = getPreferredFunctionCName(Name);

  AttributeList &PAL = Attrs.first;

  if (PAL.hasAttribute(AttributeList::FunctionIndex, Attribute::NoReturn)) {
    headerUseNoReturn();
    Out << "__noreturn ";
  }

  if (Name.find("cudakernel") != std::string::npos)
    Out << "__global__ ";

  bool isStructReturn = false;
  if (shouldFixMain) {
    Out << MainArgs.begin()[0].first;
  } else {
    // Should this function actually return a struct by-value?
    isStructReturn = PAL.hasAttribute(1, Attribute::StructRet) ||
                     PAL.hasAttribute(2, Attribute::StructRet);
    // Get the return type for the function.
    Type *RetTy;
    if (!isStructReturn)
      RetTy = FTy->getReturnType();
    else {
      // If this is a struct-return function, print the struct-return type.
      RetTy = cast<PointerType>(FTy->getParamType(0))->getElementType();
    }
    printTypeName(
        Out, RetTy,
        /*isSigned=*/
        PAL.hasAttribute(AttributeList::ReturnIndex, Attribute::SExt));
  }

  switch (Attrs.second) {
  case CallingConv::C:
    break;
  // Consider the LLVM fast calling convention as the same as the C calling
  // convention for now.
  case CallingConv::Fast:
    break;
  case CallingConv::X86_StdCall:
    Out << " __stdcall";
    break;
  case CallingConv::X86_FastCall:
    Out << " __fastcall";
    break;
  case CallingConv::X86_ThisCall:
    Out << " __thiscall";
    break;
  default:
    DBG_ERRS("Unhandled calling convention " << Attrs.second);
    errorWithMessage("Encountered Unhandled Calling Convention");
    break;
  }
  Out << ' ' << printedName << '(';

  unsigned Idx = 1;
  bool PrintedArg = false;
  FunctionType::param_iterator I = FTy->param_begin(), E = FTy->param_end();
  Function::arg_iterator ArgName =
      ArgList ? ArgList->begin() : Function::arg_iterator();

  // If this is a struct-return function, don't print the hidden
  // struct-return argument.
  if (isStructReturn) {
    cwriter_assert(!shouldFixMain);
    cwriter_assert(I != E && "Invalid struct return function!");
    ++I;
    ++Idx;
    if (ArgList)
      ++ArgName;
  }

  /*
   * OpenMP: if it's an omp function call, skip some args if not needed
   */
  for (int i = 0; i < skipArgSteps; i++) {
    ++I;
    ++Idx;
    if (ArgList)
      ++ArgName;
  }

  std::set<std::string> ReservedArgNames;
  if (ArgList && ArgList->begin() != ArgList->end()) {
    Function *ArgParent = ArgList->begin()->getParent();
    for (auto &Entry : IV2Name) {
      if (Instruction *Inst = dyn_cast<Instruction>(Entry.first)) {
        if (Inst->getFunction() == ArgParent)
          ReservedArgNames.insert(Entry.second);
      } else if (Argument *Arg = dyn_cast<Argument>(Entry.first)) {
        if (Arg->getParent() == ArgParent)
          ReservedArgNames.insert(Entry.second);
      }
    }
  }
  std::set<std::string> UsedArgNames;

  for (; I != E; ++I) {
    Type *ArgTy = *I;
    if (PAL.hasAttribute(Idx, Attribute::ByVal)) {
      cwriter_assert(!shouldFixMain);
      cwriter_assert(ArgTy->isPointerTy());
      ArgTy = cast<PointerType>(ArgTy)->getElementType();
    }
    if (PrintedArg)
      Out << ", ";
    if (shouldFixMain)
      Out << MainArgs.begin()[Idx].first;
    else
      printTypeNameUnaligned(
          Out, ArgTy,
          /*isSigned=*/PAL.hasAttribute(Idx, Attribute::SExt));
    PrintedArg = true;
    if (ArgList) {
      Out << ' ';
      std::string argName;
      if (shouldFixMain)
        argName = MainArgs.begin()[Idx].second;
      else
        argName = GetValueName(ArgName);
      std::string finalArgName =
          makeArgConflictFreeName(argName, ReservedArgNames, UsedArgNames);
      argNameOverrides[ArgName] = finalArgName;
      Out << finalArgName;
      nameDict.LocalVars[demangledName].insert(finalArgName);
      ++ArgName;
    }
    ++Idx;
  }

  if (FTy->isVarArg()) {
    cwriter_assert(!shouldFixMain);
    if (!PrintedArg) {
      Out << "int"; // dummy argument for empty vaarg functs
      if (ArgList)
        Out << " vararg_dummy_arg";
    }
    Out << ", ...";
  } else if (!PrintedArg) {
    Out << "void";
  }
  Out << ")";
  return Out;
}

raw_ostream &CWriter::printArrayDeclaration(raw_ostream &Out, ArrayType *ATy) {
  cwriter_assert(!isEmptyType(ATy));
  // Arrays are wrapped in structs to allow them to have normal
  // value semantics (avoiding the array "decay").
  Out << getArrayName(ATy) << " {\n  ";
  printTypeName(Out, ATy->getElementType());
  Out << " array[" << utostr(ATy->getNumElements()) << "];\n};\n";
  // lump it with struct declarations
  nameDict.StructDefs.insert({getArrayName(ATy).substr(7), {"array"}});
  return Out;
}

raw_ostream &CWriter::printVectorDeclaration(raw_ostream &Out,
                                             VectorType *VTy) {
  cwriter_assert(!isEmptyType(VTy));
  // Vectors are printed like arrays
  Out << getVectorName(VTy, false) << " {\n  ";
  printTypeName(Out, VTy->getElementType());
  Out << " vector[" << utostr(NumberOfElements(VTy))
      << "];\n} __attribute__((aligned(" << TD->getABITypeAlignment(VTy)
      << ")));\n";
  return Out;
}

void CWriter::printConstantArray(ConstantArray *CPA,
                                 enum OperandContext Context) {
  printConstant(cast<Constant>(CPA->getOperand(0)), Context);
  for (unsigned i = 1, e = CPA->getNumOperands(); i != e; ++i) {
    Out << ", ";
    printConstant(cast<Constant>(CPA->getOperand(i)), Context);
  }
}

void CWriter::printConstantVector(ConstantVector *CP,
                                  enum OperandContext Context) {
  printConstant(cast<Constant>(CP->getOperand(0)), Context);
  for (unsigned i = 1, e = CP->getNumOperands(); i != e; ++i) {
    Out << ", ";
    printConstant(cast<Constant>(CP->getOperand(i)), Context);
  }
}

void CWriter::printConstantDataSequential(ConstantDataSequential *CDS,
                                          enum OperandContext Context) {
  printConstant(CDS->getElementAsConstant(0), Context);
  for (unsigned i = 1, e = CDS->getNumElements(); i != e; ++i) {
    Out << ", ";
    printConstant(CDS->getElementAsConstant(i), Context);
  }
}

bool CWriter::printConstantString(Constant *C, enum OperandContext Context) {
  // As a special case, print the array as a string if it is an array of
  // ubytes or an array of sbytes with positive values.
  ConstantDataSequential *CDS = dyn_cast<ConstantDataSequential>(C);
  if (!CDS || !CDS->isString())
    return false;
  if (Context != ContextStatic)
    return false; // TODO

  Out << "{ \"";
  // Keep track of whether the last number was a hexadecimal escape.
  bool LastWasHex = false;

  StringRef Bytes = CDS->getAsString();

  unsigned length = Bytes.size();
  // We can skip the last character only if it is an implied null.
  // Beware: C does not force character (i.e. (u)int8_t here) arrays to have a
  // null terminator, but if the length is not specified it will imply one!
  if (length >= 1 && Bytes[length - 1] == '\0')
    length--;

  for (unsigned i = 0; i < length; ++i) {
    unsigned char C = Bytes[i];

    // Print it out literally if it is a printable character.  The only thing
    // to be careful about is when the last letter output was a hex escape
    // code, in which case we have to be careful not to print out hex digits
    // explicitly (the C compiler thinks it is a continuation of the previous
    // character, sheesh...)
    if (isprint(C) && (!LastWasHex || !isxdigit(C))) {
      LastWasHex = false;
      if (C == '"' || C == '\\')
        Out << "\\" << (char)C;
      else
        Out << (char)C;
    } else {
      LastWasHex = false;
      switch (C) {
      case '\n':
        Out << "\\n";
        break;
      case '\t':
        Out << "\\t";
        break;
      case '\r':
        Out << "\\r";
        break;
      case '\v':
        Out << "\\v";
        break;
      case '\a':
        Out << "\\a";
        break;
      case '\"':
        Out << "\\\"";
        break;
      case '\'':
        Out << "\\\'";
        break;
      default:
        Out << "\\x";
        Out << (char)((C / 16 < 10) ? (C / 16 + '0') : (C / 16 - 10 + 'A'));
        Out << (char)(((C & 15) < 10) ? ((C & 15) + '0')
                                      : ((C & 15) - 10 + 'A'));
        LastWasHex = true;
        break;
      }
    }
  }
  Out << "\" }";
  return true;
}

// isFPCSafeToPrint - Returns true if we may assume that CFP may be written out
// textually as a double (rather than as a reference to a stack-allocated
// variable). We decide this by converting CFP to a string and back into a
// double, and then checking whether the conversion results in a bit-equal
// double to the original value of CFP. This depends on us and the target C
// compiler agreeing on the conversion process (which is pretty likely since we
// only deal in IEEE FP).

// TODO copied from CppBackend, new code should use raw_ostream
static inline std::string ftostr(const APFloat &V) {
  if (&V.getSemantics() != &APFloat::IEEEdouble() &&
      &V.getSemantics() != &APFloat::IEEEsingle()) {
    return "<unknown format in ftostr>"; // error
  }
  SmallVector<char, 32> Buffer;
  V.toString(Buffer);
  return std::string(Buffer.data(), Buffer.size());
}

static bool isFPCSafeToPrint(const ConstantFP *CFP) {
  bool ignored;
  // Do long doubles in hex for now.
  if (CFP->getType() != Type::getFloatTy(CFP->getContext()) &&
      CFP->getType() != Type::getDoubleTy(CFP->getContext()))
    return false;
  APFloat APF = APFloat(CFP->getValueAPF()); // copy
  if (CFP->getType() == Type::getFloatTy(CFP->getContext()))
    APF.convert(APFloat::IEEEdouble(), APFloat::rmNearestTiesToEven, &ignored);
#if HAVE_PRINTF_A && ENABLE_CBE_PRINTF_A
  char Buffer[100];
  sprintf(Buffer, "%a", APF.convertToDouble());
  if (!strncmp(Buffer, "0x", 2) || !strncmp(Buffer, "-0x", 3) ||
      !strncmp(Buffer, "+0x", 3))
    return APF.bitwiseIsEqual(APFloat(atof(Buffer)));
  return false;
#else
  std::string StrVal = ftostr(APF);

  while (StrVal[0] == ' ')
    StrVal.erase(StrVal.begin());

  // Check to make sure that the stringized number is not some string like "Inf"
  // or NaN.  Check that the string matches the "[-+]?[0-9]" regex.
  if ((StrVal[0] >= '0' && StrVal[0] <= '9') ||
      ((StrVal[0] == '-' || StrVal[0] == '+') &&
       (StrVal[1] >= '0' && StrVal[1] <= '9')))
    // Reparse stringized version!
    return APF.bitwiseIsEqual(APFloat(atof(StrVal.c_str())));
  return false;
#endif
}

/// Print out the casting for a cast operation. This does the double casting
/// necessary for conversion to the destination type, if necessary.
/// @brief Print a cast
void CWriter::printCast(unsigned opc, Type *SrcTy, Type *DstTy) {
  errs() << "SUSAN: printing cast from: " << *SrcTy << " to " << *DstTy << "\n";
  // Print the destination type cast
  switch (opc) {
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::IntToPtr:
  case Instruction::Trunc:
  case Instruction::BitCast:
  case Instruction::FPExt:
  case Instruction::FPTrunc: // For these the DstTy sign doesn't matter
    Out << '(';
    printTypeName(Out, DstTy);
    Out << ')';
    break;
  case Instruction::ZExt:
  case Instruction::PtrToInt:
  case Instruction::FPToUI: // For these, make sure we get an unsigned dest
    Out << '(';
    printSimpleType(Out, DstTy, false);
    Out << ')';
    break;
  case Instruction::SExt:
  case Instruction::FPToSI: // For these, make sure we get a signed dest
    Out << '(';
    printSimpleType(Out, DstTy, true);
    Out << ')';
    break;
  default:
    errorWithMessage("Invalid cast opcode");
  }

  // Print the source type cast
  switch (opc) {
  case Instruction::UIToFP:
  case Instruction::ZExt:
    Out << '(';
    printSimpleType(Out, SrcTy, false);
    Out << ')';
    break;
  case Instruction::SIToFP:
  case Instruction::SExt:
    Out << '(';
    printSimpleType(Out, SrcTy, true);
    Out << ')';
    break;
  case Instruction::IntToPtr:
  case Instruction::PtrToInt:
    // Avoid "cast to pointer from integer of different size" warnings
    Out << "(uintptr_t)";
    break;
  case Instruction::Trunc:
  case Instruction::BitCast:
  case Instruction::FPExt:
  case Instruction::FPTrunc:
  case Instruction::FPToSI:
  case Instruction::FPToUI:
    break; // These don't need a source cast.
  default:
    errorWithMessage("Invalid cast opcode");
  }
}

// printConstant - The LLVM Constant to C Constant converter.
void CWriter::printConstant(Constant *CPV, enum OperandContext Context) {
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(CPV)) {
    // TODO: VectorType are valid here, but not supported
    if (!CE->getType()->isIntegerTy() && !CE->getType()->isFloatingPointTy() &&
        !CE->getType()->isPointerTy()) {
      DBG_ERRS("Unsupported constant type " << *CE->getType()
                                            << " in: " << *CE);
      errorWithMessage("Unsupported constant type");
    }
    switch (CE->getOpcode()) {
    case Instruction::Trunc:
    case Instruction::ZExt:
    case Instruction::SExt:
    case Instruction::FPTrunc:
    case Instruction::FPExt:
    case Instruction::UIToFP:
    case Instruction::SIToFP:
    case Instruction::FPToUI:
    case Instruction::FPToSI:
    case Instruction::PtrToInt:
    case Instruction::IntToPtr:
    case Instruction::BitCast:
      Out << "(";
      printCast(CE->getOpcode(), CE->getOperand(0)->getType(), CE->getType());
      if (CE->getOpcode() == Instruction::SExt &&
          CE->getOperand(0)->getType() == Type::getInt1Ty(CPV->getContext())) {
        // Make sure we really sext from bool here by subtracting from 0
        Out << "0-";
      }
      printConstant(CE->getOperand(0), ContextCasted);
      if (CE->getType() == Type::getInt1Ty(CPV->getContext()) &&
          (CE->getOpcode() == Instruction::Trunc ||
           CE->getOpcode() == Instruction::FPToUI ||
           CE->getOpcode() == Instruction::FPToSI ||
           CE->getOpcode() == Instruction::PtrToInt)) {
        // Make sure we really truncate to bool here by anding with 1
        Out << "&1u";
      }
      Out << ')';
      return;

    case Instruction::GetElementPtr: {
      Out << "(";
      // ConstantExpr GEPs used as a load's pointer operand must be printed as
      // memory access expressions; otherwise a [0] index can be folded away to
      // the base pointer symbol, yielding pointer-vs-integer compares.
      //
      // Keep this narrow: CurInstr may temporarily point at some previously
      // inlined load while printing unrelated call arguments, so guard on exact
      // pointer-operand identity.
      bool accessMemory = false;
      if (auto *LI = dyn_cast_or_null<LoadInst>(CurInstr)) {
        accessMemory = (LI->getPointerOperand() == CPV);
      } else if (auto *SI = dyn_cast_or_null<StoreInst>(CurInstr)) {
        accessMemory = (SI->getPointerOperand() == CPV);
      } else if (const auto *GI = dyn_cast_or_null<GetElementPtrInst>(CurInstr)) {
        // When a CE-GEP is used as the base of another GEP that eventually
        // feeds memory, print it in memory-address mode to avoid accidental
        // address/value reclassification.
        if (GI->getPointerOperand() == CPV) {
          auto *GINonConst = const_cast<GetElementPtrInst *>(GI);
          accessMemory = accessGEPMemory.find(GINonConst) != accessGEPMemory.end() ||
                         GEPAccessesMemory(GINonConst);
        }
      }
      printGEPExpressionStruct(CE->getOperand(0), gep_type_begin(CPV),
                               gep_type_end(CPV), accessMemory);
      Out << ")";
      return;
    }
    case Instruction::Select:
      Out << '(';
      printConstant(CE->getOperand(0), ContextCasted);
      Out << '?';
      printConstant(CE->getOperand(1), ContextNormal);
      Out << ':';
      printConstant(CE->getOperand(2), ContextNormal);
      Out << ')';
      return;
    case Instruction::Add:
    case Instruction::FAdd:
    case Instruction::Sub:
    case Instruction::FSub:
    case Instruction::Mul:
    case Instruction::FMul:
    case Instruction::SDiv:
    case Instruction::UDiv:
    case Instruction::FDiv:
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
    case Instruction::And:
    case Instruction::Or:
    case Instruction::Xor:
    case Instruction::ICmp:
    case Instruction::Shl:
    case Instruction::LShr:
    case Instruction::AShr: {
      Out << '(';
      bool NeedsClosingParens = printConstExprCast(CE);
      printConstantWithCast(CE->getOperand(0), CE->getOpcode());
      switch (CE->getOpcode()) {
      case Instruction::Add:
      case Instruction::FAdd:
        Out << " + ";
        break;
      case Instruction::Sub:
      case Instruction::FSub:
        Out << " - ";
        break;
      case Instruction::Mul:
      case Instruction::FMul:
        Out << " * ";
        break;
      case Instruction::URem:
      case Instruction::SRem:
      case Instruction::FRem:
        Out << " % ";
        break;
      case Instruction::UDiv:
      case Instruction::SDiv:
      case Instruction::FDiv:
        Out << " / ";
        break;
      case Instruction::And:
        Out << " & ";
        break;
      case Instruction::Or:
        Out << " | ";
        break;
      case Instruction::Xor:
        Out << " ^ ";
        break;
      case Instruction::Shl:
        Out << " << ";
        break;
      case Instruction::LShr:
      case Instruction::AShr:
        Out << " >> ";
        break;
      case Instruction::ICmp:
        switch (CE->getPredicate()) {
        case ICmpInst::ICMP_EQ:
          Out << " == ";
          break;
        case ICmpInst::ICMP_NE:
          Out << " != ";
          break;
        case ICmpInst::ICMP_SLT:
        case ICmpInst::ICMP_ULT:
          Out << " < ";
          break;
        case ICmpInst::ICMP_SLE:
        case ICmpInst::ICMP_ULE:
          Out << " <= ";
          break;
        case ICmpInst::ICMP_SGT:
        case ICmpInst::ICMP_UGT:
          Out << " > ";
          break;
        case ICmpInst::ICMP_SGE:
        case ICmpInst::ICMP_UGE:
          Out << " >= ";
          break;
        default:
          errorWithMessage("Illegal ICmp predicate");
        }
        break;
      default:
        errorWithMessage("Illegal opcode here!");
      }
      printConstantWithCast(CE->getOperand(1), CE->getOpcode());
      if (NeedsClosingParens)
        Out << "))";
      Out << ')';
      return;
    }
    case Instruction::FCmp: {
      Out << '(';
      bool NeedsClosingParens = printConstExprCast(CE);
      if (CE->getPredicate() == FCmpInst::FCMP_FALSE)
        Out << "0";
      else if (CE->getPredicate() == FCmpInst::FCMP_TRUE)
        Out << "1";
      else {
        const auto Pred = (CmpInst::Predicate)CE->getPredicate();
        headerUseFCmpOp(Pred);
        Out << "llvm_fcmp_" << getCmpPredicateName(Pred) << "(";
        printConstant(CE->getOperand(0), ContextCasted);
        Out << ", ";
        printConstant(CE->getOperand(1), ContextCasted);
        Out << ")";
      }
      if (NeedsClosingParens)
        Out << "))";
      Out << ')';
      return;
    }
    case Instruction::AddrSpaceCast: {
      // Address spaces don't exist in C.
      //
      // Keep the default behavior for general pointer values (especially
      // ConstantExpr GEP addresses like struct fields), because those rely on
      // pointer-style printing in store/load paths.
      //
      // Special-case only global-array bases so nested GEPs index as
      // ce[i][j] instead of (&ce)[i][j].
      Value *ASOp = CE->getOperand(0);
      bool globalArrayBase = false;
      if (auto *GV = dyn_cast<GlobalVariable>(ASOp)) {
        if (GV->hasInitializer())
          globalArrayBase = isa<ArrayType>(GV->getInitializer()->getType());
      }

      if (globalArrayBase) {
        errs() << "ANDREW: lowering AddrSpaceCast global-array base directly: "
               << *ASOp << "\n";
        writeOperandInternal(ASOp, ContextNormal, false);
      } else {
        errs() << "ANDREW: lowering AddrSpaceCast via printConstant: "
               << *ASOp << "\n";
        printConstant(cast<Constant>(ASOp), Context);
      }
      return;
    }
    default:
      DBG_ERRS("CWriter Error: Unhandled constant expression: " << *CE);
      errorWithMessage("unhandled constant expression");
    }
  } else if (isa<UndefValue>(CPV) && CPV->getType()->isSingleValueType()) {
    if (CPV->getType()->isVectorTy()) {
      if (Context == ContextStatic) {
        Out << "{}";
        return;
      }
      VectorType *VT = cast<VectorType>(CPV->getType());
      cwriter_assert(!isEmptyType(VT));
      CtorDeclTypes.insert(VT);
      errs() << "SUSAN: inserting VT" << *VT << "\n";
      Out << "/*undef*/llvm_ctor_";
      printTypeString(Out, VT, false);
      Out << "(";
      Constant *Zero = Constant::getNullValue(VT->getElementType());

      unsigned NumElts = NumberOfElements(VT);
      for (unsigned i = 0; i != NumElts; ++i) {
        if (i)
          Out << ", ";
        printConstant(Zero, ContextCasted);
      }
      Out << ")";

    } else {
      Constant *Zero = Constant::getNullValue(CPV->getType());
      Out << "/*UNDEF*/";
      return printConstant(Zero, Context);
    }
    return;
  }

  if (ConstantInt *CI = dyn_cast<ConstantInt>(CPV)) {
    Type *Ty = CI->getType();
    unsigned ActiveBits = CI->getValue().getMinSignedBits();
    if (Ty == Type::getInt1Ty(CPV->getContext())) {
      Out << (CI->getZExtValue() ? '1' : '0');
    } else if (Context != ContextNormal && Ty->getPrimitiveSizeInBits() <= 64 &&
               ActiveBits < Ty->getPrimitiveSizeInBits()) {
      // if (ActiveBits >= 32)
      //   Out << "INT64_C(";
      Out << CI->getSExtValue(); // most likely a shorter representation
      // if (ActiveBits >= 32)
      //   Out << ")";
    } else if (Ty->getPrimitiveSizeInBits() < 32 && Context == ContextNormal) {
      Out << "((";
      printSimpleType(Out, Ty, false) << ')';
      Out << CI->getSExtValue();
      Out << ')';
    } else if (Ty->getPrimitiveSizeInBits() <= 32) {
      // Out << CI->getZExtValue() << 'u';
      Out << CI->getSExtValue();
    } else if (Ty->getPrimitiveSizeInBits() <= 64) {
      // Out << CI->getZExtValue();
      Out << CI->getSExtValue();
    } else if (Ty->getPrimitiveSizeInBits() <= 128) {
      headerUseInt128();
      const APInt &V = CI->getValue();
      const APInt &Vlo = V.getLoBits(64);
      const APInt &Vhi = V.getHiBits(64);
      Out << (Context == ContextStatic ? "UINT128_C" : "llvm_ctor_u128");
      Out << "(UINT64_C(" << Vhi.getZExtValue() << "), UINT64_C("
          << Vlo.getZExtValue() << "))";
    }
    return;
  }

  switch (CPV->getType()->getTypeID()) {
  case Type::FloatTyID:
  case Type::DoubleTyID:
  case Type::X86_FP80TyID:
  case Type::PPC_FP128TyID:
  case Type::FP128TyID: {
    ConstantFP *FPC = cast<ConstantFP>(CPV);
    auto I = FPConstantMap.find(FPC);
    if (I != FPConstantMap.end()) {
      // Because of FP precision problems we must load from a stack allocated
      // value that holds the value in hex.
      Out << "(*("
          << (FPC->getType() == Type::getFloatTy(CPV->getContext()) ? "float"
              : FPC->getType() == Type::getDoubleTy(CPV->getContext())
                  ? "double"
                  : "long double")
          << "*)&FPConstant" << I->second << ')';
    } else {
      double V;
      if (FPC->getType() == Type::getFloatTy(CPV->getContext()))
        V = FPC->getValueAPF().convertToFloat();
      else if (FPC->getType() == Type::getDoubleTy(CPV->getContext()))
        V = FPC->getValueAPF().convertToDouble();
      else {
        // Long double.  Convert the number to double, discarding precision.
        // This is not awesome, but it at least makes the CBE output somewhat
        // useful.
        APFloat Tmp = FPC->getValueAPF();
        bool LosesInfo;
        Tmp.convert(APFloat::IEEEdouble(), APFloat::rmTowardZero, &LosesInfo);
        V = Tmp.convertToDouble();
      }

      if (std::isnan(V)) {
        // The value is NaN

        // FIXME the actual NaN bits should be emitted.
        // The prefix for a quiet NaN is 0x7FF8. For a signalling NaN,
        // it's 0x7ff4.
        const unsigned long QuietNaN = 0x7ff8UL;
        // const unsigned long SignalNaN = 0x7ff4UL;

        // We need to grab the first part of the FP #
        char Buffer[100];

        uint64_t ll = DoubleToBits(V);
        sprintf(Buffer, "0x%llx", static_cast<long long>(ll));

        std::string Num(&Buffer[0], &Buffer[6]);
        unsigned long Val = strtoul(Num.c_str(), 0, 16);

        headerUseNanInf();
        if (FPC->getType() == Type::getFloatTy(FPC->getContext()))
          Out << "LLVM_NAN" << (Val == QuietNaN ? "" : "S") << "F(\"" << Buffer
              << "\") /*nan*/ ";
        else
          Out << "LLVM_NAN" << (Val == QuietNaN ? "" : "S") << "(\"" << Buffer
              << "\") /*nan*/ ";
      } else if (std::isinf(V)) {
        // The value is Inf
        if (V < 0)
          Out << '-';
        headerUseNanInf();
        Out << "LLVM_INF"
            << (FPC->getType() == Type::getFloatTy(FPC->getContext()) ? "F"
                                                                      : "")
            << " /*inf*/ ";
      } else {
        std::string Num;
#if HAVE_PRINTF_A && ENABLE_CBE_PRINTF_A
        // Print out the constant as a floating point number.
        char Buffer[100];
        sprintf(Buffer, "%a", V);
        Num = Buffer;
#else
        Num = ftostr(FPC->getValueAPF());
#endif
        Out << Num;
      }
    }
    break;
  }

  case Type::ArrayTyID: {
    if (printConstantString(CPV, Context))
      break;
    ArrayType *AT = cast<ArrayType>(CPV->getType());
    cwriter_assert(AT->getNumElements() != 0 && !isEmptyType(AT));
    if (Context != ContextStatic) {
      errs() << "SUSAN: inserting AT" << *AT << "\n";
      CtorDeclTypes.insert(AT);
      Out << "llvm_ctor_";
      printTypeString(Out, AT, false);
      Out << "(";
      Context = ContextCasted;
    } else {
      Out << "{ "; // Arrays are wrapped in struct types.
      // SUSAN: not wrapped in struct any more
    }
    if (ConstantArray *CA = dyn_cast<ConstantArray>(CPV)) {
      printConstantArray(CA, Context);
    } else if (ConstantDataSequential *CDS =
                   dyn_cast<ConstantDataSequential>(CPV)) {
      printConstantDataSequential(CDS, Context);
    } else {
      cwriter_assert(isa<ConstantAggregateZero>(CPV) || isa<UndefValue>(CPV));
      Constant *CZ = Constant::getNullValue(AT->getElementType());
      printConstant(CZ, Context);
      for (unsigned i = 1, e = AT->getNumElements(); i != e; ++i) {
        Out << ", ";
        printConstant(CZ, Context);
      }
    }
    Out << (Context == ContextStatic
                ? " }"
                : ")"); // Arrays are wrapped in struct types.
    break;
  }
#if LLVM_VERSION_MAJOR > 10
  case Type::FixedVectorTyID:
  case Type::ScalableVectorTyID:
#else
  case Type::VectorTyID:
#endif
  {
    VectorType *VT = cast<VectorType>(CPV->getType());
    cwriter_assert(VT->getNumElements() != 0 && !isEmptyType(VT));
    if (Context != ContextStatic) {
      CtorDeclTypes.insert(VT);
      errs() << "SUSAN: inserting VT 2323" << *VT << "\n";
      Out << "llvm_ctor_";
      printTypeString(Out, VT, false);
      Out << "(";
      Context = ContextCasted;
    } else {
      Out << "{ ";
    }
    if (ConstantVector *CV = dyn_cast<ConstantVector>(CPV)) {
      printConstantVector(CV, Context);
    } else if (ConstantDataSequential *CDS =
                   dyn_cast<ConstantDataSequential>(CPV)) {
      printConstantDataSequential(CDS, Context);
    } else {
      cwriter_assert(isa<ConstantAggregateZero>(CPV) || isa<UndefValue>(CPV));
      Constant *CZ = Constant::getNullValue(VT->getElementType());
      printConstant(CZ, Context);

      for (unsigned i = 1, e = NumberOfElements(VT); i != e; ++i) {
        Out << ", ";
        printConstant(CZ, Context);
      }
    }
    Out << (Context == ContextStatic ? " }" : ")");
    break;
  }

  case Type::StructTyID: {
    StructType *ST = cast<StructType>(CPV->getType());
    cwriter_assert(!isEmptyType(ST));
    if (Context != ContextStatic) {
      CtorDeclTypes.insert(ST);
      errs() << "SUSAN: inserting ST" << *ST << "\n";
      Out << "llvm_ctor_";
      printTypeString(Out, ST, false);
      Out << "(";
      Context = ContextCasted;
    } else {
      Out << "{ ";
    }

    if (isa<ConstantAggregateZero>(CPV) || isa<UndefValue>(CPV)) {
      bool printed = false;
      for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i) {
        Type *ElTy = ST->getElementType(i);
        if (isEmptyType(ElTy))
          continue;
        if (printed)
          Out << ", ";
        printConstant(Constant::getNullValue(ElTy), Context);
        printed = true;
      }
      cwriter_assert(printed);
    } else {
      bool printed = false;
      for (unsigned i = 0, e = CPV->getNumOperands(); i != e; ++i) {
        Constant *C = cast<Constant>(CPV->getOperand(i));
        if (isEmptyType(C->getType()))
          continue;
        if (printed)
          Out << ", ";
        printConstant(C, Context);
        printed = true;
      }
      cwriter_assert(printed);
    }
    Out << (Context == ContextStatic ? " }" : ")");
    break;
  }

  case Type::PointerTyID:
    if (isa<ConstantPointerNull>(CPV)) {
      Out << "((";
      printTypeName(Out, CPV->getType()); // sign doesn't matter
      Out << ")0)";
      break;
    } else if (GlobalValue *GV = dyn_cast<GlobalValue>(CPV)) {
      writeOperand(GV);
      break;
    }
    // FALL THROUGH
  default:
    DBG_ERRS("Unknown constant type: " << *CPV);
    errorWithMessage("unknown constant type");
  }
}

// Some constant expressions need to be casted back to the original types
// because their operands were casted to the expected type. This function takes
// care of detecting that case and printing the cast for the ConstantExpr.
bool CWriter::printConstExprCast(ConstantExpr *CE) {
  bool NeedsExplicitCast = false;
  Type *Ty = CE->getOperand(0)->getType();
  bool TypeIsSigned = false;
  switch (CE->getOpcode()) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
    // We need to cast integer arithmetic so that it is always performed
    // as unsigned, to avoid undefined behavior on overflow.
  case Instruction::LShr:
  case Instruction::URem:
  case Instruction::UDiv:
    NeedsExplicitCast = true;
    break;
  case Instruction::AShr:
  case Instruction::SRem:
  case Instruction::SDiv:
    NeedsExplicitCast = true;
    TypeIsSigned = true;
    break;
  case Instruction::SExt:
    Ty = CE->getType();
    NeedsExplicitCast = true;
    TypeIsSigned = true;
    break;
  case Instruction::ZExt:
  case Instruction::Trunc:
  case Instruction::FPTrunc:
  case Instruction::FPExt:
  case Instruction::UIToFP:
  case Instruction::SIToFP:
  case Instruction::FPToUI:
  case Instruction::FPToSI:
  case Instruction::PtrToInt:
  case Instruction::IntToPtr:
  case Instruction::BitCast:
    Ty = CE->getType();
    NeedsExplicitCast = true;
    break;
  default:
    break;
  }
  if (NeedsExplicitCast) {
    Out << "((";
    printTypeName(Out, Ty, TypeIsSigned); // not integer, sign doesn't matter
    Out << ")(";
  }
  return NeedsExplicitCast;
}

//  Print a constant assuming that it is the operand for a given Opcode. The
//  opcodes that care about sign need to cast their operands to the expected
//  type before the operation proceeds. This function does the casting.
void CWriter::printConstantWithCast(Constant *CPV, unsigned Opcode) {

  // Extract the operand's type, we'll need it.
  Type *OpTy = CPV->getType();
  // TODO: VectorType are valid here, but not supported
  if (!OpTy->isIntegerTy() && !OpTy->isFloatingPointTy()) {
    DBG_ERRS("Unsupported 'constant with cast' type " << *OpTy
                                                      << " in: " << *CPV);
    errorWithMessage("Unsupported 'constant with cast' type");
  }

  // Indicate whether to do the cast or not.
  bool shouldCast;
  bool typeIsSigned;
  opcodeNeedsCast(Opcode, shouldCast, typeIsSigned);

  // Write out the casted constant if we should, otherwise just write the
  // operand.
  if (shouldCast) {
    Out << "((";
    printSimpleType(Out, OpTy, typeIsSigned);
    Out << ")";
    printConstant(CPV, ContextCasted);
    Out << ")";
  } else
    printConstant(CPV, ContextCasted);
}

std::string demangleFunctionName(std::string str) {
  std::string FuncName = demangle(str);

  return FuncName.substr(0, FuncName.find("("));
}

static bool isSimpleCIdentifier(const std::string &Name) {
  if (Name.empty())
    return false;
  unsigned char first = static_cast<unsigned char>(Name[0]);
  bool firstOk = ((first >= 'a' && first <= 'z') ||
                  (first >= 'A' && first <= 'Z') || first == '_');
  if (!firstOk)
    return false;
  for (unsigned i = 1; i < Name.size(); ++i) {
    unsigned char ch = static_cast<unsigned char>(Name[i]);
    bool ok = ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
               (ch >= '0' && ch <= '9') || ch == '_');
    if (!ok)
      return false;
  }
  return true;
}

static std::string getPreferredFunctionCName(const std::string &SymbolName) {
  std::string Demangled = demangleFunctionName(SymbolName);
  if (isSimpleCIdentifier(Demangled))
    return Demangled;
  return SymbolName;
}

std::string demangleVariableName(std::string var) {
  SmallVector<std::string, 16> splitedStrs;

  std::string VarName;
  VarName.reserve(var.capacity());
  for (std::string::iterator I = var.begin(), E = var.end(); I != E; ++I) {
    unsigned char ch = *I;
    if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
          (ch >= '0' && ch <= '9') || ch == '_')) {
      char buffer[5];
      sprintf(buffer, "_%x_", ch);
      VarName += buffer;
    } else
      VarName += ch;
  }

  return VarName;
}

static bool isCudaBuiltinStyleArgName(StringRef Name) {
  return Name.contains("threadIdx.") || Name.contains("blockIdx.") ||
         Name.contains("blockDim.") || Name.contains("gridDim.");
}

static bool isShortIVStyleName(StringRef Name) {
  return Name == "i" || Name == "j" || Name == "k" || Name == "m";
}

static std::string makeArgConflictFreeName(const std::string &Base,
                                           const std::set<std::string> &Reserved,
                                           std::set<std::string> &Used) {
  std::string SafeBase = Base.empty() ? std::string("arg") : Base;
  std::string Candidate = SafeBase;
  if (Reserved.find(Candidate) == Reserved.end() &&
      Used.find(Candidate) == Used.end()) {
    Used.insert(Candidate);
    return Candidate;
  }

  Candidate = SafeBase + "_arg";
  if (Reserved.find(Candidate) == Reserved.end() &&
      Used.find(Candidate) == Used.end()) {
    Used.insert(Candidate);
    return Candidate;
  }

  unsigned Suffix = 1;
  while (true) {
    std::string Numbered = Candidate + "_" + utostr(Suffix++);
    if (Reserved.find(Numbered) == Reserved.end() &&
        Used.find(Numbered) == Used.end()) {
      Used.insert(Numbered);
      return Numbered;
    }
  }
}

std::string CWriter::GetValueName(Value *Operand, bool isDeclaration) {
  if (isDeclaration) {
    errs() << "SUSAN: declaring 3252: " << *Operand << "\n";
    cnt_totalVariables++;
  }

  if (Operand->getName() == "xmalloc")
    return "malloc";

  if (!Operand)
    return "";
  if (argNameOverrides.find(Operand) != argNameOverrides.end())
    return argNameOverrides[Operand];
  if (IV2Name.find(Operand) != IV2Name.end()) {
    bool foundSourceName = false;
    for (auto inst2var : IRNaming) {
      if (inst2var.first == Operand) {
        foundSourceName = true;
        break;
      }
    }
    if (!foundSourceName) {
      errs() << "SUSAN: found in IV2Name map " << *Operand << "\n";
      errs() << "name:  " << IV2Name[Operand] << "\n";
      if (isDeclaration) {
        errs() << "SUSAN: reconstructed variable counter increment for iv:"
               << IV2Name[Operand] << "\n";
        cnt_reconstructedVariables++;
      }
      return IV2Name[Operand];
    }
  }
  errs() << "SUSAN: getting value name for: " << *Operand << "\n";
  // SUSAN: variable names associated with phi will be replaced by phi
  if (TruncInst *inst = dyn_cast<TruncInst>(Operand))
    return GetValueName(inst->getOperand(0));

  if (Instruction *inst = dyn_cast<Instruction>(Operand))
    if (deleteAndReplaceInsts.find(inst) != deleteAndReplaceInsts.end()) {
      return GetValueName(deleteAndReplaceInsts[inst]);
    }

  if (inlinedArgNames.find(Operand) != inlinedArgNames.end())
    return inlinedArgNames[Operand];

  // SUSAN: where the vairable names are printed
  for (auto inst2var : IRNaming)
    if (inst2var.first == Operand) {
      errs() << "inst from IRNaming: " << *inst2var.first << "\n";
      errs() << "original name : " << inst2var.second << "\n";
      std::string var = demangleVariableName(inst2var.second);
      errs() << "returning name: " << var << "\n";
      if (isDeclaration) {
        errs() << "SUSAN: declaring with reconstructed name 3286: " << var
               << "\n";
        cnt_reconstructedVariables++;
      }
      return var;
    }
  // for (auto const& [var, insts] : Var2IRs)
  // for (auto &inst : insts)
  // if(inst == operandInst) return var;

  // Resolve potential alias.
  if (GlobalAlias *GA = dyn_cast<GlobalAlias>(Operand)) {
    Operand = GA->getAliasee();
  }

  // use IV name for IVInc
  // Instruction *incInst = dyn_cast<Instruction>(Operand);
  // if(IVInc2IV.find(incInst) != IVInc2IV.end())
  //  return GetValueName(IVInc2IV[incInst]);

  // YEBIN : add FIXME prefix to vars with no metadata
  std::string Name{Operand->getName()};
  bool noMetadataName = Name.empty();
  if (noMetadataName) { // Assign unique names to local temporaries.
    unsigned No = AnonValueNumbers.getOrInsert(Operand);

    Name = utostr(No);
    // Name = "_" + utostr(No);
    if (!TheModule->getNamedValue(Name)) {
      // Short name for the common case where there's no conflicting global.
      return "__FIXME__" + Name;
    }

    Name = "tmp_" + Name;
    // Name = Name + "_tmp";
  }

  // Mangle globals and also append a FIXME to vars
  if (isa<GlobalVariable>(Operand)) {
    if(noMetadataName)
      return "__FIXME_GLOBAL__" + CBEMangle(Name);
    return CBEMangle(Name);
  }

  // Mangle globals with the standard mangler interface for LLC compatibility.
  if (isa<GlobalValue>(Operand)) {
    return CBEMangle(Name);
  }

  std::string VarName;
  VarName.reserve(Name.capacity());

  for (std::string::iterator I = Name.begin(), E = Name.end(); I != E; ++I) {
    unsigned char ch = *I;

    if (!((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') ||
          (ch >= '0' && ch <= '9') || ch == '_')) {
      char buffer[5];
      sprintf(buffer, "_%x_", ch);
      VarName += buffer;
    } else
      VarName += ch;
  }

  // return "_" + VarName;
  return "__FIXME__" + VarName;
}

/// writeInstComputationInline - Emit the computation for the specified
/// instruction inline, with no destination provided.
void CWriter::writeInstComputationInline(Instruction &I, bool startExpression) {
  if (deleteAndReplaceInsts.find(&I) != deleteAndReplaceInsts.end()) {
    if (deleteAndReplaceInsts[&I]->getName() == "stderr") {
      Out << " stderr ";
      return;
    }
    writeOperandInternal(deleteAndReplaceInsts[&I]);
    return;
  }
  gepStart = startExpression;
  // C can't handle non-power-of-two integer types
  unsigned mask = 0;
  Type *Ty = I.getType();
  if (Ty->isIntegerTy()) {
    IntegerType *ITy = static_cast<IntegerType *>(Ty);
#if LLVM_VERSION_MAJOR <= 10
    if (!ITy->isPowerOf2ByteWidth())
#else
    if (!IsPowerOfTwo(ITy->getBitWidth()))
#endif
      mask = ITy->getBitMask();
  }

  // If this is a non-trivial bool computation, make sure to truncate down to
  // a 1 bit value.  This is important because we want "add i1 x, y" to return
  // "0" when x and y are true, not "2" for example.
  // Also truncate odd bit sizes
  // if (mask)
  // Out << "((";

  visit(&I);

  // if (mask)
  // Out << ")&" << mask << ")";
}

void CWriter::writeOperandInternal(Value *Operand, enum OperandContext Context,
                                   bool startExpression) {
  if (PHINode *phi = dyn_cast<PHINode>(Operand)) {
    if (LoadInst *ld = dyn_cast<LoadInst>(phi->getIncomingValue(0)))
      if (ld->getPointerOperand()->getName() == "stderr") {
        Out << "stderr";
        return;
      }
  }

  if (inlinedArgNames.find(Operand) != inlinedArgNames.end()) {
    errs() << "SUSAN: returning inlined name 3339: "
           << inlinedArgNames[Operand];
    errs() << "SUSAN: operand: " << *Operand << "\n";
    bool printDouble = false;
    if (valuesCast2Double.find(Operand) != valuesCast2Double.end()) {
      printDouble = true;
      for (auto [ld, object] : doubleGeps)
        if (object == Operand) {
          printDouble = false;
          break;
        }
    }
    if (printDouble)
      Out << "((double*)";
    Out << inlinedArgNames[Operand];
    if (printDouble)
      Out << ")";
    return;
  }

  Instruction *inst = dyn_cast<Instruction>(Operand);
  if (inst && deleteAndReplaceInsts.find(inst) != deleteAndReplaceInsts.end()) {
    if (deleteAndReplaceInsts[inst]->getName() == "stderr") {
      Out << " stderr ";
      return;
    }
    writeOperandInternal(deleteAndReplaceInsts[inst]);
    return;
  }

  // Keep extra induction variables as distinct symbols in regular loop code.
  // The old remap to main IV (via IVMap) can collapse indices like nza -> k.
  // Restrict this remap to OpenMP-specific paths where it was originally useful.
  if (isExtraInductionVariable(Operand) && IS_OPENMP_FUNCTION) {
    PHINode *phi = dyn_cast<PHINode>(Operand);
    for (auto [iv, relatedIVs] : IVMap)
      if (relatedIVs.find(phi) != relatedIVs.end()) {
        writeOperandInternal(iv);

        Value *offset = nullptr;
        for (auto LP : LoopProfiles) {
          if (LP->IV == iv) {
            if (!LP->lbAlloca || !LP->incr)
              break;
            errs() << "SUSAN: main IV's lb: " << *LP->lb << "\n";
            errs() << "SUSAN: main IV's lballoca: " << *LP->lbAlloca << "\n";
            errs() << "SUSAN: main IV's incr: " << *LP->incr << "\n";

            Value *initVal = nullptr;
            Instruction *incrementInst = nullptr;
            for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
              BasicBlock *predBB = phi->getIncomingBlock(i);
              if (LI->getLoopFor(predBB) != LP->L)
                initVal = phi->getIncomingValue(i);
              else
                incrementInst = dyn_cast<Instruction>(phi->getIncomingValue(i));
            }

            // check if IV steps are the same
            if (incrementInst->getOperand(1) != LP->incr)
              break;
            assert(initVal && "relatedIV doesn't have initVal??\n");

            errs() << "SUSAN: relatedIV's init val: " << *initVal << "\n";
            if (BinaryOperator *binOp = dyn_cast<BinaryOperator>(initVal)) {
              Value *opnd0 = binOp->getOperand(0);
              opnd0 = findOriginalValue(opnd0);
              errs() << "SUSAN: relatedIV original opnd0: " << *opnd0 << "\n";
              if (opnd0 == LP->lbAlloca)
                offset = binOp->getOperand(1);
            }
          }
        }

        if (offset)
          writeOperandInternal(offset);
        return;
      }
  }
  if (isExtraInductionVariable(Operand) && !IS_OPENMP_FUNCTION) {
    PHINode *phi = dyn_cast<PHINode>(Operand);
    if (phi && isExtraIVEquivalentToMainIV(phi)) {
      Loop *L = LI->getLoopFor(phi->getParent());
      PHINode *mainIV = getInductionVariable(L);
      errs() << "ANDREW: remapping equivalent extra IV to main IV: "
             << *Operand << " -> " << *mainIV << "\n";
      if (mainIV->getType() != phi->getType()) {
        Out << "((";
        printSimpleType(Out, phi->getType(), /*isSigned=*/true);
        Out << ")";
        writeOperandInternal(mainIV);
        Out << ")";
      } else {
        writeOperandInternal(mainIV);
      }
      return;
    }
    errs() << "ANDREW: preserving extra IV in non-OpenMP path: " << *Operand
           << "\n";
  }

  if (Instruction *I = dyn_cast<Instruction>(Operand))
    // Should we inline this instruction to build a tree?
    if (isInlinableInst(*I) && !isDirectAlloca(I)) {
      if (isa<LoadInst>(I) && addParenthesis.find(I) != addParenthesis.end())
        Out << '(';
      writeInstComputationInline(*I, startExpression);
      if (isa<LoadInst>(I) && addParenthesis.find(I) != addParenthesis.end())
        Out << ')';
      return;
    }

  Constant *CPV = dyn_cast<Constant>(Operand);

  if (CPV && !isa<GlobalValue>(CPV)) {
    printConstant(CPV, Context);
  } else {
    if (isa<Function>(Operand)) {
      Out << getPreferredFunctionCName(GetValueName(Operand));
    } else
      Out << GetValueName(Operand);
  }
}

void CWriter::writeOperand(Value *Operand, enum OperandContext Context,
                           bool startExpression) {
  errs() << "CBackend: writeOperand 3595: " << *Operand << "\n";
  if (LoadInst *ld = dyn_cast<LoadInst>(Operand))
    if (ld->getPointerOperand()->getName() == "stderr") {
      Out << "stderr";
      return;
    }
  /*if(PHINode *phi = dyn_cast<PHINode>(Operand)){
    if(phis2print.find(phi) != phis2print.end()){
      writeOperand(phis2print[phi]);
      return;
    }
  }*/
  if (inlinedArgNames.find(Operand) != inlinedArgNames.end()) {
    errs() << "SUSAN: returning inlined name 3426: "
           << inlinedArgNames[Operand];
    bool printDouble = false;
    if (valuesCast2Double.find(Operand) != valuesCast2Double.end()) {
      printDouble = true;
      for (auto [ld, object] : doubleGeps)
        if (object == Operand) {
          printDouble = false;
          break;
        }
    }
    if (printDouble)
      Out << "(double*)";
    Out << inlinedArgNames[Operand];
    return;
  }

  auto isLoopIVOrRelated = [this](Value *V) -> bool {
    if (!V)
      return false;
    if (PHINode *phi = dyn_cast<PHINode>(V))
      return isInductionVariable(phi) || isExtraInductionVariable(phi);
    if (Instruction *inst = dyn_cast<Instruction>(V))
      return isIVIncrement(inst) || isExtraIVIncrement(inst);
    return false;
  };

  if (isLoopIVOrRelated(Operand)) {
    errs() << "ANDREW: writeOperand preserving IV-related operand: " << *Operand;
    if (CurLoop)
      errs() << " in loop " << CurLoop->getHeader()->getName();
    errs() << "\n";
  }

  if (!isLoopIVOrRelated(Operand) &&
      InstsToReplaceByPhi.find(Operand) != InstsToReplaceByPhi.end()) {
    // Follow the replacement chain iteratively with cycle detection as a safety net
    std::set<Value*> Visited;
    Value *Current = Operand;
    
    while (InstsToReplaceByPhi.find(Current) != InstsToReplaceByPhi.end()) {
        if (!Visited.insert(Current).second) {
            // Cycle detected - this shouldn't happen now that preprossesPHIs2Print
            // prevents cycles, but keep as safety. Use first visited as representative.
            errs() << "WARNING: Cycle detected in InstsToReplaceByPhi at: " << *Current << "\n";
            break;
        }
        Value *Next = InstsToReplaceByPhi[Current];
        if (isLoopIVOrRelated(Current) || isLoopIVOrRelated(Next)) {
          errs() << "ANDREW: preserving IV-related operand mapping: " << *Current
                 << " -> " << *Next << "\n";
          break;
        }
        Current = Next;
    }
    
    // Current is now the final replacement (or a cycle representative)
    // Update Operand and continue with the rest of writeOperand
    if (!isLoopIVOrRelated(Current))
      Operand = Current;
  }



  Instruction *inst = dyn_cast<Instruction>(Operand);
  if (inst && deleteAndReplaceInsts.find(inst) != deleteAndReplaceInsts.end()) {
    if (deleteAndReplaceInsts[inst]->getName() == "stderr") {
      Out << " stderr ";
      return;
    }
    writeOperand(deleteAndReplaceInsts[inst]);
    return;
  }

  bool isAddressImplicit = isAddressExposed(Operand);
  // Global variables are referenced as their addresses by llvm
  if (isAddressImplicit) {
    // We can't directly declare a zero-sized variable in C, so
    // printTypeNameForAddressableValue uses a single-byte type instead.
    // We fix up the pointer type here.
    if (!isEmptyType(Operand->getType()->getPointerElementType()))
      Out << "(&";
    else
      Out << "((void*)&";
  }

  bool isOmpLoop = false;
  LoopProfile *LP = nullptr;
  for (auto lp : LoopProfiles)
    if (CurLoop && lp->L == CurLoop && lp->isOmpLoop) {
      isOmpLoop = true;
      LP = lp;
      break;
    }

  bool printIVIncrementExpr =
      isIVIncrement(Operand) && !isa<CmpInst>(CurInstr) && !omp_declarePrivate;
  Value *ivBaseToPrint = nullptr;
  Value *ivStepToPrint = nullptr;
  bool negateStep = false;

  if (printIVIncrementExpr) {
    if (Instruction *IncInst = dyn_cast<Instruction>(Operand)) {
      if (BinaryOperator *BO = dyn_cast<BinaryOperator>(IncInst)) {
        Value *Op0 = BO->getOperand(0);
        Value *Op1 = BO->getOperand(1);
        if (BO->getOpcode() == Instruction::Add) {
          if (isLoopIVOrRelated(Op0)) {
            ivBaseToPrint = Op0;
            ivStepToPrint = Op1;
          } else if (isLoopIVOrRelated(Op1)) {
            ivBaseToPrint = Op1;
            ivStepToPrint = Op0;
          }
        } else if (BO->getOpcode() == Instruction::Sub &&
                   isLoopIVOrRelated(Op0)) {
          ivBaseToPrint = Op0;
          ivStepToPrint = Op1;
          negateStep = true;
        }
      }
    }

    if (!ivBaseToPrint && isOmpLoop && LP)
      ivBaseToPrint = LP->IV;
    if (!ivStepToPrint && isOmpLoop && LP)
      ivStepToPrint = LP->incr;
  }

  if (printIVIncrementExpr)
    Out << "(";

  if (printIVIncrementExpr && ivBaseToPrint)
    writeOperandInternal(ivBaseToPrint, Context, startExpression);
  else
    writeOperandInternal(Operand, Context, startExpression);

  if (printIVIncrementExpr) {
    if (ivStepToPrint) {
      if (ConstantInt *StepConst = dyn_cast<ConstantInt>(ivStepToPrint)) {
        int64_t StepVal = StepConst->getSExtValue();
        if (negateStep)
          StepVal = -StepVal;
        if (StepVal >= 0)
          Out << " + " << StepVal;
        else
          Out << " - " << -StepVal;
      } else {
        Out << (negateStep ? " - " : " + ");
        writeOperandInternal(ivStepToPrint, Context, startExpression);
      }
    } else {
      // Conservative fallback for unusual increment forms.
      Out << " + 1";
    }
    Out << ")";
  }

  if (isAddressImplicit)
    Out << ')';

  errs() << "CBackend: writeoperand here 3674? \n";
}

/// writeOperandDeref - Print the result of dereferencing the specified
/// operand with '*'.  This is equivalent to printing '*' then using
/// writeOperand, but avoids excess syntax in some cases.
void CWriter::writeOperandDeref(Value *Operand) {
  if (isAddressExposed(Operand)) {
    // Already something with an address exposed.
    writeOperandInternal(Operand);
  } else {
    Out << "*(";
    writeOperand(Operand);
    Out << ")";
  }
}

// Some instructions need to have their result value casted back to the
// original types because their operands were casted to the expected type.
// This function takes care of detecting that case and printing the cast
// for the Instruction.
bool CWriter::writeInstructionCast(Instruction &I) {
  Type *Ty = I.getOperand(0)->getType();
  switch (I.getOpcode()) {
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
    // We need to cast integer arithmetic so that it is always performed
    // as unsigned, to avoid undefined behavior on overflow.
  case Instruction::LShr:
  case Instruction::URem:
  case Instruction::UDiv:
    Out << "((";
    printSimpleType(Out, Ty, false);
    Out << ")(";
    return true;
  case Instruction::AShr:
  case Instruction::SRem:
  case Instruction::SDiv:
    Out << "((";
    printSimpleType(Out, Ty, true);
    Out << ")(";
    return true;
  default:
    break;
  }
  return false;
}

void CWriter::writePointerCastSource(Value *Operand, PointerCastUseKind UseKind,
                                     enum OperandContext Context,
                                     bool startExpression) {
  Value *BaseObj = Operand ? Operand->stripPointerCasts() : nullptr;
  bool isBareAddressExposedObject =
      BaseObj && isAddressExposed(BaseObj) &&
      !isa<GetElementPtrInst>(Operand) && !isa<GEPOperator>(Operand);

  bool isScalarElementLValue = false;
  if (auto *GEP = dyn_cast<GetElementPtrInst>(Operand)) {
    Type *EltTy = GEP->getResultElementType();
    isScalarElementLValue =
        EltTy && !EltTy->isAggregateType() && !EltTy->isPointerTy();
  } else if (auto *GEPOp = dyn_cast<GEPOperator>(Operand)) {
    Type *EltTy = GEPOp->getResultElementType();
    isScalarElementLValue =
        EltTy && !EltTy->isAggregateType() && !EltTy->isPointerTy();
  }

  if (isBareAddressExposedObject) {
    writeOperand(BaseObj, Context, startExpression);
    return;
  }

  if (UseKind == PointerCastDirectMemoryOperand && isScalarElementLValue) {
    Out << "(&";
    writeOperandInternal(Operand, Context, startExpression);
    Out << ")";
    return;
  }

  writeOperandInternal(Operand, Context, startExpression);
}

// Write the operand with a cast to another type based on the Opcode being used.
// This will be used in cases where an instruction has specific type
// requirements (usually signedness) for its operands.
void CWriter::opcodeNeedsCast(
    unsigned Opcode,
    // Indicate whether to do the cast or not.
    bool &shouldCast,
    // Indicate whether the cast should be to a signed type or not.
    bool &castIsSigned) {

  // Based on the Opcode for which this Operand is being written, determine
  // the new type to which the operand should be casted by setting the value
  // of OpTy. If we change OpTy, also set shouldCast to true.
  switch (Opcode) {
  default:
    // for most instructions, it doesn't matter
    shouldCast = false;
    castIsSigned = false;
    break;
  case Instruction::Add:
  case Instruction::Sub:
  case Instruction::Mul:
    // We need to cast integer arithmetic so that it is always performed
    // as unsigned, to avoid undefined behavior on overflow.
  case Instruction::LShr:
  case Instruction::UDiv:
  case Instruction::URem: // Cast to unsigned first
    shouldCast = true;
    castIsSigned = false;
    break;
  case Instruction::GetElementPtr:
  case Instruction::AShr:
  case Instruction::SDiv:
  case Instruction::SRem: // Cast to signed first
    shouldCast = true;
    castIsSigned = true;
    break;
  }
}

void CWriter::writeOperandWithCast(Value *Operand, unsigned Opcode,
                                   bool startExpression) {
  // Write out the casted operand if we should, otherwise just write the
  // operand.

  // Extract the operand's type, we'll need it.
  bool shouldCast;
  bool castIsSigned;
  opcodeNeedsCast(Opcode, shouldCast, castIsSigned);

  Type *OpTy = Operand->getType();
  if (shouldCast) {
    Out << "((";
    printSimpleType(Out, OpTy, castIsSigned);
    Out << ")";
    writeOperand(Operand, ContextCasted, startExpression);
    Out << ")";
  } else
    writeOperand(Operand, ContextCasted, startExpression);
}

// Write the operand with a cast to another type based on the icmp predicate
// being used.
void CWriter::writeOperandWithCast(Value *Operand, ICmpInst &Cmp) {

  // ANDREW CASTING
  bool shouldCast = needsCast(Operand, Cmp);

  // Write out the casted operand if we should, otherwise just write the
  // operand.
  if (!shouldCast) {
    writeOperand(Operand);
    return;
  }

  // Should this be a signed comparison?  If so, convert to signed.
  bool castIsSigned = Cmp.isSigned();

  // If the operand was a pointer, convert to a large integer type.
  Type *OpTy = Operand->getType();
  if (OpTy->isPointerTy())
    OpTy = TD->getIntPtrType(Operand->getContext());

  Out << "((";
  printSimpleType(Out, OpTy, castIsSigned);
  Out << ")";
  writeOperand(Operand);
  Out << ")";
}
// ANDREW CASTING
bool CWriter::isCSigned(Value *V) const {
  if (isa<ConstantInt>(V))
    return true; // Constants are printed as signed integers in CBE
  if (Instruction *I = dyn_cast<Instruction>(V)) {
     if (isa<SExtInst>(I))
       return true; // Explicit SExt is signed in C
     if (isInlinableInst(*I))
         return false; // Safely assume inlined ops might be unsigned -> force cast
    return signedInsts.find(I) != signedInsts.end();
  }
  // Globals and Arguments are unsigned by default in CBE
  return false;
}

bool CWriter::needsCast(Value *V, ICmpInst &I) const {
  if (!I.isRelational())
    return false;
  if (V->getType()->isPointerTy())
    return true;
  return isCSigned(V) != I.isSigned();
}

static void defineConstantDoubleTy(raw_ostream &Out) {
  Out << "typedef uint64_t ConstantDoubleTy;\n";
}

static void defineConstantFloatTy(raw_ostream &Out) {
  Out << "typedef uint32_t ConstantFloatTy;\n";
}

static void defineConstantFP80Ty(raw_ostream &Out) {
  Out << "typedef struct { uint64_t f1; uint16_t f2; "
         "uint16_t pad[3]; } ConstantFP80Ty;\n";
}

static void defineConstantFP128Ty(raw_ostream &Out) {
  // This is used for both kinds of 128-bit long double; meaning differs.
  Out << "typedef struct { uint64_t f1; uint64_t f2; }"
         " ConstantFP128Ty;\n";
}

static void defineBuiltinAlloca(raw_ostream &Out) {
  // Alloca is hard to get, and we don't want to include stdlib.h here.
  Out << "/* get a declaration for alloca */\n"
      << "#if defined(__CYGWIN__) || defined(__MINGW32__)\n"
      << "#define  alloca(x) __builtin_alloca((x))\n"
      << "#define _alloca(x) __builtin_alloca((x))\n"
      << "#elif defined(__APPLE__)\n"
      << "extern void *__builtin_alloca(unsigned long);\n"
      << "#define alloca(x) __builtin_alloca(x)\n"
      << "#define longjmp _longjmp\n"
      << "#define setjmp _setjmp\n"
      << "#elif defined(__sun__)\n"
      << "#if defined(__sparcv9)\n"
      << "extern void *__builtin_alloca(unsigned long);\n"
      << "#else\n"
      << "extern void *__builtin_alloca(unsigned int);\n"
      << "#endif\n"
      << "#define alloca(x) __builtin_alloca(x)\n"
      << "#elif defined(__FreeBSD__) || defined(__NetBSD__) || "
         "defined(__OpenBSD__) || defined(__DragonFly__) || defined(__arm__)\n"
      << "#define alloca(x) __builtin_alloca(x)\n"
      << "#elif defined(_MSC_VER)\n"
      << "#define alloca(x) _alloca(x)\n"
      << "#else\n"
      << "#include <alloca.h>\n"
      << "#endif\n\n";
}

static void defineExternalWeak(raw_ostream &Out) {
  // On Mac OS X, "external weak" is spelled "__attribute__((weak_import))".
  Out << "#if defined(__GNUC__) && defined(__APPLE_CC__)\n"
      << "#define __EXTERNAL_WEAK__ __attribute__((weak_import))\n"
      << "#elif defined(__GNUC__)\n"
      << "#define __EXTERNAL_WEAK__ __attribute__((weak))\n"
      << "#else\n"
      << "#define __EXTERNAL_WEAK__\n"
      << "#endif\n\n";
}

static void defineAttributeWeak(raw_ostream &Out) {
  // For now, turn off the weak linkage attribute on Mac OS X. (See above.)
  Out << "#if defined(__GNUC__) && defined(__APPLE_CC__)\n"
      << "#define __ATTRIBUTE_WEAK__\n"
      << "#elif defined(__GNUC__)\n"
      << "#define __ATTRIBUTE_WEAK__ __attribute__((weak))\n"
      << "#else\n"
      << "#define __ATTRIBUTE_WEAK__\n"
      << "#endif\n\n";
}

static void defineHidden(raw_ostream &Out) {
  // Add hidden visibility support. FIXME: APPLE_CC?
  Out << "#if defined(__GNUC__)\n"
      << "#define __HIDDEN__ __attribute__((visibility(\"hidden\")))\n"
      << "#endif\n\n";
}

static void defineAttributeList(raw_ostream &Out) {
  // gcc attributes
  Out << "#if defined(__GNUC__)\n"
      << "#define  __ATTRIBUTELIST__(x) __attribute__(x)\n"
      << "#else\n"
      << "#define  __ATTRIBUTELIST__(x)  \n"
      << "#endif\n\n";

  // We output GCC specific attributes to preserve 'linkonce'ness on globals.
  // If we aren't being compiled with GCC, just drop these attributes.
  Out << "#ifdef _MSC_VER  /* Can only support \"linkonce\" vars with GCC */\n"
      << "#define __attribute__(X)\n"
      << "#endif\n\n";
}

static void defineUnalignedLoad(raw_ostream &Out) {
  // Define unaligned-load helper macro
  Out << "#ifdef _MSC_VER\n";
  Out << "#define __UNALIGNED_LOAD__(type, align, op) *((type "
         "__unaligned*)op)\n";
  Out << "#else\n";
  Out << "#define __UNALIGNED_LOAD__(type, align, op) ((struct { type data "
         "__attribute__((packed, aligned(align))); }*)op)->data\n";
  Out << "#endif\n\n";
}

static void defineMsAlign(raw_ostream &Out) {
  Out << "#ifdef _MSC_VER\n";
  Out << "#define __MSALIGN__(X) __declspec(align(X))\n";
  Out << "#else\n";
  Out << "#define __MSALIGN__(X)\n";
  Out << "#endif\n\n";
}

static void defineUnreachable(raw_ostream &Out) {
  Out << "#ifdef _MSC_VER\n";
  Out << "#define __builtin_unreachable() __assume(0)\n";
  Out << "#endif\n";
}

static void defineNoReturn(raw_ostream &Out) {
  Out << "#ifdef _MSC_VER\n";
  Out << "#define __noreturn __declspec(noreturn)\n";
  Out << "#else\n";
  Out << "#define __noreturn __attribute__((noreturn))\n";
  Out << "#endif\n";
}

static void defineForceInline(raw_ostream &Out) {
  Out << "#ifndef _MSC_VER\n";
  Out << "#define __forceinline __attribute__((always_inline)) inline\n";
  Out << "#endif\n\n";
}

static void defineNanInf(raw_ostream &Out) {
  // Define NaN and Inf as GCC builtins if using GCC
  // From the GCC documentation:
  //
  //   double __builtin_nan (const char *str)
  //
  // This is an implementation of the ISO C99 function nan.
  //
  // Since ISO C99 defines this function in terms of strtod, which we do
  // not implement, a description of the parsing is in order. The string is
  // parsed as by strtol; that is, the base is recognized by leading 0 or
  // 0x prefixes. The number parsed is placed in the significand such that
  // the least significant bit of the number is at the least significant
  // bit of the significand. The number is truncated to fit the significand
  // field provided. The significand is forced to be a quiet NaN.
  //
  // This function, if given a string literal, is evaluated early enough
  // that it is considered a compile-time constant.
  //
  //   float __builtin_nanf (const char *str)
  //
  // Similar to __builtin_nan, except the return type is float.
  //
  //   double __builtin_inf (void)
  //
  // Similar to __builtin_huge_val, except a warning is generated if the
  // target floating-point format does not support infinities. This
  // function is suitable for implementing the ISO C99 macro INFINITY.
  //
  //   float __builtin_inff (void)
  //
  // Similar to __builtin_inf, except the return type is float.
  Out << "#ifdef __GNUC__\n"
      << "#define LLVM_NAN(NanStr)   __builtin_nan(NanStr)   /* Double */\n"
      << "#define LLVM_NANF(NanStr)  __builtin_nanf(NanStr)  /* Float */\n"
      //<< "#define LLVM_NANS(NanStr)  __builtin_nans(NanStr)  /* Double */\n"
      //<< "#define LLVM_NANSF(NanStr) __builtin_nansf(NanStr) /* Float */\n"
      << "#define LLVM_INF           __builtin_inf()         /* Double */\n"
      << "#define LLVM_INFF          __builtin_inff()        /* Float */\n"
      << "#define LLVM_PREFETCH(addr,rw,locality) "
         "__builtin_prefetch(addr,rw,locality)\n"
      << "#define __ATTRIBUTE_CTOR__ __attribute__((constructor))\n"
      << "#define __ATTRIBUTE_DTOR__ __attribute__((destructor))\n"
      << "#else\n"
      << "#define LLVM_NAN(NanStr)   ((double)NAN)           /* Double */\n"
      << "#define LLVM_NANF(NanStr)  ((float)NAN))           /* Float */\n"
      //<< "#define LLVM_NANS(NanStr)  ((double)NAN)           /* Double */\n"
      //<< "#define LLVM_NANSF(NanStr) ((single)NAN)           /* Float */\n"
      << "#define LLVM_INF           ((double)INFINITY)      /* Double */\n"
      << "#define LLVM_INFF          ((float)INFINITY)       /* Float */\n"
      << "#define LLVM_PREFETCH(addr,rw,locality)            /* PREFETCH */\n"
      << "#define __ATTRIBUTE_CTOR__ \"__attribute__((constructor)) not "
         "supported on this compiler\"\n"
      << "#define __ATTRIBUTE_DTOR__ \"__attribute__((destructor)) not "
         "supported on this compiler\"\n"
      << "#endif\n\n";
}

static void defineStackSaveRestore(raw_ostream &Out) {
  Out << "#if !defined(__GNUC__) || __GNUC__ < 4 /* Old GCC's, or compilers "
         "not GCC */ \n"
      << "#define __builtin_stack_save() 0   /* not implemented */\n"
      << "#define __builtin_stack_restore(X) /* noop */\n"
      << "#endif\n\n";
}

static void defineInt128(raw_ostream &Out) {
  // Output typedefs for 128-bit integers
  Out << "#if defined(__GNUC__) && defined(__LP64__) /* 128-bit integer types "
         "*/\n"
      << "typedef int __attribute__((mode(TI))) int128_t;\n"
      << "typedef unsigned __attribute__((mode(TI))) uint128_t;\n"
      << "#define UINT128_C(hi, lo) (((uint128_t)(hi) << 64) | "
         "(uint128_t)(lo))\n"
      << "static __forceinline uint128_t llvm_ctor_u128(uint64_t hi, uint64_t "
         "lo) {"
      << " return UINT128_C(hi, lo); }\n"
      << "static __forceinline bool llvm_icmp_eq_u128(uint128_t l, uint128_t "
         "r) {"
      << " return l == r; }\n"
      << "static __forceinline bool llvm_icmp_ne_u128(uint128_t l, uint128_t "
         "r) {"
      << " return l != r; }\n"
      << "static __forceinline bool llvm_icmp_ule_u128(uint128_t l, uint128_t "
         "r) {"
      << " return l <= r; }\n"
      << "static __forceinline bool llvm_icmp_sle_i128(int128_t l, int128_t r) "
         "{"
      << " return l <= r; }\n"
      << "static __forceinline bool llvm_icmp_uge_u128(uint128_t l, uint128_t "
         "r) {"
      << " return l >= r; }\n"
      << "static __forceinline bool llvm_icmp_sge_i128(int128_t l, int128_t r) "
         "{"
      << " return l >= r; }\n"
      << "static __forceinline bool llvm_icmp_ult_u128(uint128_t l, uint128_t "
         "r) {"
      << " return l < r; }\n"
      << "static __forceinline bool llvm_icmp_slt_i128(int128_t l, int128_t r) "
         "{"
      << " return l < r; }\n"
      << "static __forceinline bool llvm_icmp_ugt_u128(uint128_t l, uint128_t "
         "r) {"
      << " return l > r; }\n"
      << "static __forceinline bool llvm_icmp_sgt_i128(int128_t l, int128_t r) "
         "{"
      << " return l > r; }\n"

      << "#else /* manual 128-bit types */\n"
      // TODO: field order should be reversed for big-endian
      << "typedef struct { uint64_t lo; uint64_t hi; } uint128_t;\n"
      << "typedef uint128_t int128_t;\n"
      << "#define UINT128_C(hi, lo) {(lo), (hi)}\n" // only use in Static
                                                    // context
      << "static __forceinline uint128_t llvm_ctor_u128(uint64_t hi, uint64_t "
         "lo) {"
      << " uint128_t r; r.lo = lo; r.hi = hi; return r; }\n"
      << "static __forceinline bool llvm_icmp_eq_u128(uint128_t l, uint128_t "
         "r) {"
      << " return l.hi == r.hi && l.lo == r.lo; }\n"
      << "static __forceinline bool llvm_icmp_ne_u128(uint128_t l, uint128_t "
         "r) {"
      << " return l.hi != r.hi || l.lo != r.lo; }\n"
      << "static __forceinline bool llvm_icmp_ule_u128(uint128_t l, uint128_t "
         "r) {"
      << " return l.hi < r.hi ? 1 : (l.hi == r.hi ? l.lo <= l.lo : 0); }\n"
      << "static __forceinline bool llvm_icmp_sle_i128(int128_t l, int128_t r) "
         "{"
      << " return (int64_t)l.hi < (int64_t)r.hi ? 1 : (l.hi == r.hi ? "
         "(int64_t)l.lo <= (int64_t)l.lo : 0); }\n"
      << "static __forceinline bool llvm_icmp_uge_u128(uint128_t l, uint128_t "
         "r) {"
      << " return l.hi > r.hi ? 1 : (l.hi == r.hi ? l.lo >= l.hi : 0); }\n"
      << "static __forceinline bool llvm_icmp_sge_i128(int128_t l, int128_t r) "
         "{"
      << " return (int64_t)l.hi > (int64_t)r.hi ? 1 : (l.hi == r.hi ? "
         "(int64_t)l.lo >= (int64_t)l.lo : 0); }\n"
      << "static __forceinline bool llvm_icmp_ult_u128(uint128_t l, uint128_t "
         "r) {"
      << " return l.hi < r.hi ? 1 : (l.hi == r.hi ? l.lo < l.hi : 0); }\n"
      << "static __forceinline bool llvm_icmp_slt_i128(int128_t l, int128_t r) "
         "{"
      << " return (int64_t)l.hi < (int64_t)r.hi ? 1 : (l.hi == r.hi ? "
         "(int64_t)l.lo < (int64_t)l.lo : 0); }\n"
      << "static __forceinline bool llvm_icmp_ugt_u128(uint128_t l, uint128_t "
         "r) {"
      << " return l.hi > r.hi ? 1 : (l.hi == r.hi ? l.lo > l.hi : 0); }\n"
      << "static __forceinline bool llvm_icmp_sgt_i128(int128_t l, int128_t r) "
         "{"
      << " return (int64_t)l.hi > (int64_t)r.hi ? 1 : (l.hi == r.hi ? "
         "(int64_t)l.lo > (int64_t)l.lo : 0); }\n"
      << "#define __emulate_i128\n"
      << "#endif\n\n";
}

static void defineThreadFence(raw_ostream &Out) {
  Out << "#ifdef _MSC_VER\n"
      << "#define __atomic_thread_fence(x) __faststorefence\n"
      << "#endif\n\n";
}

/// FindStaticTors - Given a static ctor/dtor list, unpack its contents into
/// the StaticTors set.
static void FindStaticTors(GlobalVariable *GV,
                           std::set<Function *> &StaticTors) {
  ConstantArray *InitList = dyn_cast<ConstantArray>(GV->getInitializer());
  if (!InitList)
    return;

  for (unsigned i = 0, e = InitList->getNumOperands(); i != e; ++i)
    if (ConstantStruct *CS =
            dyn_cast<ConstantStruct>(InitList->getOperand(i))) {
      if (CS->getNumOperands() != 2)
        return; // Not array of 2-element structs.

      if (CS->getOperand(1)->isNullValue())
        return; // Found a null terminator, exit printing.
      Constant *FP = CS->getOperand(1);
      if (ConstantExpr *CE = dyn_cast<ConstantExpr>(FP))
        if (CE->isCast())
          FP = CE->getOperand(0);
      if (Function *F = dyn_cast<Function>(FP))
        StaticTors.insert(F);
    }
}

enum SpecialGlobalClass {
  NotSpecial = 0,
  GlobalCtors,
  GlobalDtors,
  NotPrinted
};

/// getGlobalVariableClass - If this is a global that is specially recognized
/// by LLVM, return a code that indicates how we should handle it.
static SpecialGlobalClass getGlobalVariableClass(GlobalVariable *GV) {
  // If this is a global ctors/dtors list, handle it now.
  if (GV->hasAppendingLinkage() && GV->use_empty()) {
    if (GV->getName() == "llvm.global_ctors")
      return GlobalCtors;
    else if (GV->getName() == "llvm.global_dtors")
      return GlobalDtors;
  }

  // Otherwise, if it is other metadata, don't print it.  This catches things
  // like debug information.
  if (StringRef(GV->getSection()) == "llvm.metadata")
    return NotPrinted;

  return NotSpecial;
}

// PrintEscapedString - Print each character of the specified string, escaping
// it if it is not printable or if it is an escape char.
static void PrintEscapedString(const char *Str, unsigned Length,
                               raw_ostream &Out) {
  for (unsigned i = 0; i != Length; ++i) {
    unsigned char C = Str[i];
    if (isprint(C) && C != '\\' && C != '"')
      Out << C;
    else if (C == '\\')
      Out << "\\\\";
    else if (C == '\"')
      Out << "\\\"";
    else if (C == '\t')
      Out << "\\t";
    else
      Out << "\\x" << hexdigit(C >> 4) << hexdigit(C & 0x0F);
  }
}

// PrintEscapedString - Print each character of the specified string, escaping
// it if it is not printable or if it is an escape char.
static void PrintEscapedString(const std::string &Str, raw_ostream &Out) {
  PrintEscapedString(Str.c_str(), Str.size(), Out);
}

// //Hailong Jiang
// //To tell if the loop is parallelized
// void isDoAllLoop (){

//   isOmpLoop = true;
// }

// generateCompilerSpecificCode - This is where we add conditional compilation
// directives to cater to specific compilers as need be.
void CWriter::generateCompilerSpecificCode(raw_ostream &Out,
                                           const DataLayout *) const {
  if (headerIncConstantDoubleTy())
    defineConstantDoubleTy(Out);
  if (headerIncConstantFloatTy())
    defineConstantFloatTy(Out);
  if (headerIncConstantFP80Ty())
    defineConstantFP80Ty(Out);
  if (headerIncConstantFP128Ty())
    defineConstantFP128Ty(Out);
  if (headerIncBuiltinAlloca())
    defineBuiltinAlloca(Out);
  if (headerIncUnreachable())
    defineUnreachable(Out);
  if (headerIncNoReturn())
    defineNoReturn(Out);
  if (headerIncForceInline())
    defineForceInline(Out);
  if (headerIncExternalWeak())
    defineExternalWeak(Out);
  if (headerIncAttributeWeak())
    defineAttributeWeak(Out);
  if (headerIncHidden())
    defineHidden(Out);
  if (headerIncAttributeList())
    defineAttributeList(Out);
  if (headerIncUnalignedLoad())
    defineUnalignedLoad(Out);
  if (headerIncMsAlign())
    defineMsAlign(Out);
  if (headerIncNanInf())
    defineNanInf(Out);
  if (headerIncInt128())
    defineInt128(Out);
  if (headerIncThreadFence())
    defineThreadFence(Out);
  if (headerIncStackSaveRestore())
    defineStackSaveRestore(Out);
}

bool CWriter::doInitialization(Module &M) {
  TheModule = &M;

  TD = new DataLayout(&M);
  IL = new IntrinsicLowering(*TD);
#if LLVM_VERSION_MAJOR < 9
  IL->AddPrototypes(M);
#endif

#if 0
  std::string Triple = TheModule->getTargetTriple();
  if (Triple.empty())
    Triple = llvm::sys::getDefaultTargetTriple();

  std::string E;
  if (const Target *Match = TargetRegistry::lookupTarget(Triple, E))
    TAsm = Match->createMCAsmInfo(Triple);
#endif
  TAsm = new CBEMCAsmInfo();
  MRI = new MCRegisterInfo();
#if LLVM_VERSION_MAJOR > 12
  TCtx = new MCContext(llvm::Triple(TheModule->getTargetTriple()), TAsm, MRI,
                       nullptr);
#else
  TCtx = new MCContext(TAsm, MRI, nullptr);
#endif
  return false;
}

bool CWriter::doFinalization(Module &M) {
  // Output all code to the file
  std::string methods = Out.str();
  _Out.clear();
  generateHeader(M);
  std::string header = OutHeaders.str() + Out.str();
  printDict();
  _Out.clear();
  _OutHeaders.clear();
  FileOut << header << methods;

  // Free memory...

  delete IL;
  IL = nullptr;

  delete TD;
  TD = nullptr;

  delete TCtx;
  TCtx = nullptr;

  delete TAsm;
  TAsm = nullptr;

  delete MRI;
  MRI = nullptr;

  delete MOFI;
  MOFI = nullptr;

  FPConstantMap.clear();
  ByValParams.clear();
  AnonValueNumbers.clear();
  UnnamedStructIDs.clear();
  UnnamedFunctionIDs.clear();
  TypedefDeclTypes.clear();
  SelectDeclTypes.clear();
  CmpDeclTypes.clear();
  CastOpDeclTypes.clear();
  InlineOpDeclTypes.clear();
  CtorDeclTypes.clear();
  prototypesToGen.clear();

  return true; // may have lowered an IntrinsicCall
}

void CWriter::findOMPFunctions(Module &M) {
  /*
   * OpenMP: search for openmp functions
   */
  ompFuncs.clear();
  std::set<std::string> ompThreadQueryFuncs;
  for (Module::iterator FI = M.begin(), FE = M.end(); FI != FE; ++FI) {
    Function *F = &*FI;
    for (inst_iterator I = inst_begin(F), E = inst_end(F); I != E; ++I) {
      CallInst *callInst = dyn_cast<CallInst>(&*I);
      if (!callInst)
        continue;
      if (Function *F = callInst->getCalledFunction()) {
        if (F->getName() == "__kmpc_fork_call") {
          ConstantExpr *utaskCast =
              dyn_cast<ConstantExpr>(callInst->getArgOperand(2));
          Function *utask;
          if (utaskCast && utaskCast->isCast())
            utask = dyn_cast<Function>(utaskCast->getOperand(0));
          else
            utask = dyn_cast<Function>(callInst->getArgOperand(2));

          errs() << "SUSAN: adding utask" << *utask << "\n";
          ompFuncs[callInst] = utask;
        } else if (F->getName() == "omp_get_thread_num" ||
                   F->getName() == "omp_get_num_threads") {
          ompThreadQueryFuncs.insert(callInst->getFunction()->getName().str());
        }
      }
    }
  }

  if (ompFuncs.empty() && !ompThreadQueryFuncs.empty()) {
    errs() << "CBackend warning: found OpenMP thread-query calls without "
              "canonical __kmpc_fork_call in module. Parallel regions may have "
              "been flattened before C backend.\n";
    errs() << "CBackend warning: affected function(s): ";
    bool first = true;
    for (const auto &funcName : ompThreadQueryFuncs) {
      if (!first)
        errs() << ", ";
      errs() << funcName;
      first = false;
    }
    errs() << "\n";
  }

  // delete argInput in a struct
  for (auto [callInst, utask] : ompFuncs) {
    int numArgs = std::distance(utask->arg_begin(), utask->arg_end()) - 2;
    for (auto idx = 3; idx < numArgs + 3; ++idx) {
      Value *argInput = callInst->getArgOperand(idx);
      // Value *arg = utask->getArg(idx-1);
      // Hailong Jiang 03/23/2023
      Value *arg = utask->arg_begin() + (idx - 1);
      PointerType *ty = dyn_cast<PointerType>(arg->getType());
      if (ty)
        type2declare[argInput] = ty->getPointerElementType();

      errs() << "SUSAN: argInput 4158: " << *argInput << "\n";
      errs() << "SUSAN: arg 4158: " << *arg << "\n";
      PointerType *ptrTy = dyn_cast<PointerType>(argInput->getType());
      if (ptrTy && isa<StructType>(ptrTy->getPointerElementType())) {
        for (auto U : argInput->users()) {
          GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(U);
          if (!gep)
            continue;
          ConstantInt *constint = dyn_cast<ConstantInt>(gep->getOperand(2));

          for (auto storeU : gep->users()) {
            if (StoreInst *store = dyn_cast<StoreInst>(storeU)) {
              errs() << "SUSAN: found store for struct 9066: " << *store
                     << "\n";
              if (isa<AllocaInst>(store->getOperand(0)))
                for (auto user : store->getOperand(0)->users())
                  if (StoreInst *storeAlloca = dyn_cast<StoreInst>(user))
                    if (storeAlloca->getPointerOperand() ==
                        store->getOperand(0))
                      deadInsts.insert(storeAlloca);
              deadInsts.insert(store);
            } else if (CastInst *cast = dyn_cast<CastInst>(storeU)) {
              for (auto storeU : cast->users()) {
                StoreInst *store = dyn_cast<StoreInst>(storeU);
                if (!store)
                  continue;
                errs() << "SUSAN: found store for struct 9095: " << *store
                       << "\n";
                deadInsts.insert(store);
              }
            }
          }
        }
      } else {
        if (auto alloca = isDirectAlloca(argInput))
          for (auto user : alloca->users())
            if (StoreInst *store = dyn_cast<StoreInst>(user))
              if (store->getPointerOperand() == alloca) {
                errs() << "YEBIN: FOUND DEAD STORE " << *store << "\n";
                deadInsts.insert(store);
              }
      }
    }
  }
}

void CWriter::generateHeader(Module &M) {
  // Keep track of which functions are static ctors/dtors so they can have
  // an attribute added to their prototypes.
  std::set<Function *> StaticCtors, StaticDtors;
  for (Module::global_iterator I = M.global_begin(), E = M.global_end(); I != E;
       ++I) {
    switch (getGlobalVariableClass(&*I)) {
    default:
      break;
    case GlobalCtors:
      FindStaticTors(&*I, StaticCtors);
      break;
    case GlobalDtors:
      FindStaticTors(&*I, StaticDtors);
      break;
    }
  }

  // Include required standard headers
  OutHeaders << "/* Provide Declarations */\n";
  if (headerIncStdarg())
    OutHeaders << "#include <stdarg.h>\n";
  if (headerIncSetjmp())
    OutHeaders << "#include <setjmp.h>\n";
  if (headerIncLimits())
    OutHeaders << "#include <limits.h>\n";
  // Support for integers with explicit sizes. This one isn't conditional
  // because virtually all CBE output will use it.
  OutHeaders << "#include <stdint.h>\n"; // Sized integer support
  OutHeaders << "#include <stdio.h>\n";  //
  OutHeaders << "#include <stdlib.h>\n";
  OutHeaders << "#include <string.h>\n";
  OutHeaders << "#include <math.h>\n";
  // Provide a definition for `bool' if not compiling with a C++ compiler.
  OutHeaders << "#ifndef __cplusplus\ntypedef unsigned char bool;\n#endif\n";
  OutHeaders << "\n";

  Out << "\n\n/* Global Declarations */\n";

  // First output all the declarations for the program, because C requires
  // Functions & globals to be declared before they are used.
  if (!M.getModuleInlineAsm().empty()) {
    Out << "\n/* Module asm statements */\n"
        << "__asm__ (";

    // Split the string into lines, to make it easier to read the .ll file.
    std::string Asm = M.getModuleInlineAsm();
    size_t CurPos = 0;
    size_t NewLine = Asm.find_first_of('\n', CurPos);
    while (NewLine != std::string::npos) {
      // We found a newline, print the portion of the asm string from the
      // last newline up to this newline.
      Out << "\"";
      PrintEscapedString(
          std::string(Asm.begin() + CurPos, Asm.begin() + NewLine), Out);
      Out << "\\n\"\n";
      CurPos = NewLine + 1;
      NewLine = Asm.find_first_of('\n', CurPos);
    }
    Out << "\"";
    PrintEscapedString(std::string(Asm.begin() + CurPos, Asm.end()), Out);
    Out << "\");\n"
        << "/* End Module asm statements */\n";
  }

  // collect any remaining types
  raw_null_ostream NullOut;
  for (Module::global_iterator I = M.global_begin(), E = M.global_end(); I != E;
       ++I) {
    // Ignore special globals, such as debug info.
    if (getGlobalVariableClass(&*I))
      continue;
    printTypeName(NullOut, I->getType()->getElementType(), false);
  }

  // collect function types that are arguments in the standard library calls
  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {
    Function *func = &*I;
    std::pair<AttributeList, CallingConv::ID> Attrs =
        std::make_pair(func->getAttributes(), func->getCallingConv());
    AttributeList &PAL = Attrs.first;
    FunctionType *FTy = func->getFunctionType();
    bool isStructReturn = false;
    isStructReturn = PAL.hasAttribute(1, Attribute::StructRet) ||
                     PAL.hasAttribute(2, Attribute::StructRet);

    // Get the return type for the function.
    Type *RetTy;
    if (!isStructReturn)
      RetTy = FTy->getReturnType();
    else {
      // If this is a struct-return function, print the struct-return type.
      RetTy = cast<PointerType>(FTy->getParamType(0))->getElementType();
    }

    // add return type to functionIDs if it's function type;
    if (RetTy->getTypeID() == Type::FunctionTyID) {
      FunctionType *FTy = cast<FunctionType>(RetTy);
      UnnamedFunctionIDs.getOrInsert(std::make_pair(FTy, Attrs));
    }

    // Get the argument types
    FunctionType::param_iterator II = FTy->param_begin(), EE = FTy->param_end();
    for (; II != EE; ++II) {
      Type *ArgTy = *II;
      // add argument type to functionIDs if it's function type;
      if (ArgTy->getTypeID() == Type::FunctionTyID) {
        FunctionType *FTy = cast<FunctionType>(ArgTy);
        UnnamedFunctionIDs.getOrInsert(std::make_pair(FTy, Attrs));
      } else if (ArgTy->getTypeID() == Type::PointerTyID) {
        Type *ElTy = ArgTy->getPointerElementType();
        if (FunctionType *FTy = dyn_cast<FunctionType>(ElTy)) {
          std::pair<AttributeList, CallingConv::ID> PAL_generic =
              std::make_pair(AttributeList(), CallingConv::C);
          UnnamedFunctionIDs.getOrInsert(std::make_pair(FTy, PAL_generic));
        }
      }
    }
  }

  // Intrinsic helper emission can request llvmBitCastUnion after this point,
  // so keep the typedef available in generated C unconditionally.
  headerUseBitCastUnion();
  printModuleTypes(Out);

  // Global variable declarations...
  if (!M.global_empty()) {
    Out << "\n/* External Global Variable Declarations */\n";
    for (Module::global_iterator I = M.global_begin(), E = M.global_end();
         I != E; ++I) {
      if (!I->isDeclaration())
        continue;

      if (I->getName() == "stderr")
        continue;
      if (I->hasDLLImportStorageClass())
        Out << "__declspec(dllimport) ";
      else if (I->hasDLLExportStorageClass())
        Out << "__declspec(dllexport) ";

      if (I->hasExternalLinkage() || I->hasExternalWeakLinkage() ||
          I->hasCommonLinkage())
        Out << "extern ";
      else
        continue; // Internal Global

      // Thread Local Storage
      if (I->isThreadLocal())
        Out << "__thread ";

      Type *ElTy = I->getType()->getElementType();
      unsigned Alignment = I->getAlignment();
      bool IsOveraligned =
          Alignment && Alignment > TD->getABITypeAlignment(ElTy);
      if (IsOveraligned) {
        headerUseMsAlign();
        Out << "__MSALIGN__(" << Alignment << ") ";
      }
      printTypeNameForAddressableValue(Out, ElTy, false);
      Out << ' ' << GetValueName(&*I);
      if (IsOveraligned)
        Out << " __attribute__((aligned(" << Alignment << ")))";

      if (I->hasExternalWeakLinkage()) {
        headerUseExternalWeak();
        Out << " __EXTERNAL_WEAK__";
      }
      Out << ";\n";
    }
  }

  // Function declarations
  Out << "\n/* Function Declarations */\n";

  // Store the intrinsics which will be declared/defined below.
  SmallVector<Function *, 16> intrinsicsToDefine;

  findOMPFunctions(M);

  // size_t Function_Order_ID = 0;

  for (Module::iterator I = M.begin(), E = M.end(); I != E; ++I) {

    errs() << "ANDREW: function name: " << (&*I)->getName() << "\n";
    /*
     * OpenMP: skip declaring kmpc functions
     */
    if ((&*I)->getName().contains("__kmpc"))
      continue;
    if ((&*I)->getName().contains("free"))
      continue;
    if ((&*I)->getName().contains("strtol"))
      continue;
    if ((&*I)->getName().contains("fprintf"))
      continue;
    if ((&*I)->getName().contains("fscanf"))
      continue;
    if ((&*I)->getName().contains("atoll"))
      continue;
    if ((&*I)->getName().contains("calloc"))
      continue;
    if ((&*I)->getName().contains("puts"))
      continue;
    // Hailong Jiang 04/04/2023
    if ((&*I)->getName().contains("printf"))
      continue;
    if ((&*I)->getName().contains("fputc"))
      continue;
    if ((&*I)->getName().contains("malloc"))
      continue;
    if ((&*I)->getName().contains("posix_memalign"))
      continue;
    if ((&*I)->getName().contains("fwrite"))
      continue;
    if ((&*I)->getName().contains("exit"))
      continue;
    if ((&*I)->getName().contains("atoi"))
      continue;

    // ANDREW: MORE PRINCIPLED FIX NEEDS TO BE DONE HERE
    if ((&*I)->getName() == "rand")
      continue;
    if ((&*I)->getName().contains("fopen"))
      continue;
    if ((&*I)->getName().contains("fgetc"))
      continue;
    if ((&*I)->getName().contains("fclose"))
      continue;
    if ((&*I)->getName().contains("memcpy"))
      continue;
    // Do not print any cuda functions, they should not be in the code
    // Check for specific CUDA runtime API functions
    if ((&*I)->getName().contains("cudaBindTextureToArray"))
      continue;
    if ((&*I)->getName().contains("cudaBindTexture"))
      continue;
    if ((&*I)->getName().contains("cudaConfigureCall"))
      continue;
    if ((&*I)->getName().contains("cudaCreateChannelDesc"))
      continue;
    if ((&*I)->getName().contains("cudaDeviceSynchronize"))
      continue;
    if ((&*I)->getName().contains("cudaEventElapsedTime"))
      continue;
    if ((&*I)->getName().contains("cudaEventRecord"))
      continue;
    if ((&*I)->getName().contains("cudaEventSynchronize"))
      continue;
    if ((&*I)->getName().contains("cudaEventCreate"))
      continue;
    if ((&*I)->getName().contains("cudaFreeArray"))
      continue;
    if ((&*I)->getName().contains("cudaFreeHost"))
      continue;
    if ((&*I)->getName().contains("cudaFree"))
      continue;
    if ((&*I)->getName().contains("cudaFuncSetCacheConfig"))
      continue;
    if ((&*I)->getName().contains("cudaGetDeviceProperties"))
      continue;
    if ((&*I)->getName().contains("cudaGetDeviceCount"))
      continue;
    if ((&*I)->getName().contains("cudaGetDevice"))
      continue;
    if ((&*I)->getName().contains("cudaGetErrorString"))
      continue;
    if ((&*I)->getName().contains("cudaGetLastError"))
      continue;
    if ((&*I)->getName().contains("cudaLaunch"))
      continue;
    if ((&*I)->getName().contains("cudaMallocPitch"))
      continue;
    if ((&*I)->getName().contains("cudaMallocHost"))
      continue;
    if ((&*I)->getName().contains("cudaMallocArray"))
      continue;
    if ((&*I)->getName().contains("cudaMalloc"))
      continue;
    if ((&*I)->getName().contains("cudaMemcpyToSymbol"))
      continue;
    if ((&*I)->getName().contains("cudaMemcpyToArray"))
      continue;
    if ((&*I)->getName().contains("cudaMemcpyAsync"))
      continue;
    if ((&*I)->getName().contains("cudaMemcpy2D"))
      continue;
    if ((&*I)->getName().contains("cudaMemcpy"))
      continue;
    if ((&*I)->getName().contains("cudaMemGetInfo"))
      continue;
    if ((&*I)->getName().contains("cudaMemsetAsync"))
      continue;
    if ((&*I)->getName().contains("cudaMemset"))
      continue;
    if ((&*I)->getName().contains("cudaSetDevice"))
      continue;
    if ((&*I)->getName().contains("cudaSetupArgument"))
      continue;
    if ((&*I)->getName().contains("cudaStreamCreate"))
      continue;
    if ((&*I)->getName().contains("cudaStreamDestroy"))
      continue;
    if ((&*I)->getName().contains("cudaStreamSynchronize"))
      continue;
    if ((&*I)->getName().contains("cudaStreamWaitEvent"))
      continue;
    if ((&*I)->getName().contains("cudaThreadExit"))
      continue;
    if ((&*I)->getName().contains("cudaThreadSynchronize"))
      continue;
    if ((&*I)->getName().contains("cudaUnbindTexture"))
      continue;
    if ((&*I)->getName().contains("strcat"))
      continue;
    if ((&*I)->getName().contains("strcpy"))
      continue;
    if ((&*I)->getName().contains("memset"))
      continue;
    // if((&*I)->getName().contains("xmalloc")) continue;
    //  Don't print declarations for intrinsic functions.
    //  Store the used intrinsics, which need to be explicitly defined.
    if (I->isIntrinsic()) {
      switch (I->getIntrinsicID()) {
      default:
        continue;
      case Intrinsic::uadd_with_overflow:
      case Intrinsic::sadd_with_overflow:
      case Intrinsic::usub_with_overflow:
      case Intrinsic::ssub_with_overflow:
      case Intrinsic::umul_with_overflow:
      case Intrinsic::smul_with_overflow:
      case Intrinsic::bswap:
      case Intrinsic::ceil:
      case Intrinsic::ctlz:
      case Intrinsic::ctpop:
      case Intrinsic::cttz:
      case Intrinsic::fabs:
      case Intrinsic::floor:
      case Intrinsic::fma:
      case Intrinsic::fmuladd:
      case Intrinsic::pow:
      case Intrinsic::powi:
      case Intrinsic::rint:
      case Intrinsic::sqrt:
      case Intrinsic::trunc:
        intrinsicsToDefine.push_back(&*I);
        continue;

      // ANDREW: ptx specific
      case Intrinsic::nvvm_mul24_i:
      case Intrinsic::nvvm_fma_rn_d:
      case Intrinsic::nvvm_add_rn_d:
      case Intrinsic::nvvm_mul_rn_d:
      case Intrinsic::nvvm_lohi_i2d:
      case Intrinsic::nvvm_d2i_hi:
      case Intrinsic::nvvm_d2i_lo:
        errs() << "ANDREW: Intrinsic: ptx helper hit\n";
        intrinsicsToDefine.push_back(&*I);
        continue;
      }
    }

    // Skip a few functions that have already been defined in headers
    if ((headerIncSetjmp() &&
         (I->getName() == "setjmp" || I->getName() == "longjmp" ||
          I->getName() == "_setjmp" || I->getName() == "siglongjmp" ||
          I->getName() == "sigsetjmp")) ||
        (headerIncMath() &&
         (I->getName() == "pow" || I->getName() == "powf" ||
          I->getName() == "sqrt" || I->getName() == "sqrtf" ||
          I->getName() == "trunc" || I->getName() == "truncf" ||
          I->getName() == "rint" || I->getName() == "rintf" ||
          I->getName() == "floor" || I->getName() == "floorf" ||
          I->getName() == "ceil" || I->getName() == "ceilf")) ||
        I->getName() == "alloca" || I->getName() == "_alloca" ||
        I->getName() == "_chkstk" || I->getName() == "__chkstk" ||
        I->getName() == "___chkstk_ms")
      continue;

    if (I->hasDLLImportStorageClass())
      Out << "__declspec(dllimport) ";
    else if (I->hasDLLExportStorageClass())
      Out << "__declspec(dllexport) ";

    // if (I->hasLocalLinkage())
    //   Out << "static ";
    if (I->hasExternalWeakLinkage())
      Out << "extern ";

    /*
     * OpenMP: declare outlined functions
     */
    bool printedOmpDec = false;
    std::set<Function *> printedFunction;
    for (auto [call, utask] : ompFuncs)
      if (&*I == utask &&
          printedFunction.find(utask) == printedFunction.end()) {
        printedFunction.insert(utask);
        printFunctionProto(Out, &*I, 2);
        printedOmpDec = true;
      }

    if (!printedOmpDec) {
    //   Out << "__FIXME__FUNCTION_ORDER_ID__" << Function_Order_ID << '\n';
    //   ++Function_Order_ID;
       printFunctionProto(Out, &*I);
    }

    printFunctionAttributes(Out, I->getAttributes());
    if (I->hasWeakLinkage() || I->hasLinkOnceLinkage()) {
      headerUseAttributeWeak();
      Out << " __ATTRIBUTE_WEAK__";
    }
    if (I->hasExternalWeakLinkage()) {
      headerUseExternalWeak();
      Out << " __EXTERNAL_WEAK__";
    }
    if (StaticCtors.count(&*I))
      Out << " __ATTRIBUTE_CTOR__";
    if (StaticDtors.count(&*I))
      Out << " __ATTRIBUTE_DTOR__";
    if (I->hasHiddenVisibility()) {
      headerUseHidden();
      Out << " __HIDDEN__";
    }

    if (I->hasName() && I->getName()[0] == 1)
      Out << " __asm__ (\"" << I->getName().substr(1) << "\")";

    Out << ";\n";
  }

  // Output the global variable definitions and contents...
  if (!M.global_empty()) {
    Out << "\n\n/* Global Variable Definitions and Initialization */\n";

    // SUSAN: globals can be out of order in llvm IR, for example, the following
    // is legal:
    // @tree.sdown = internal global i8* getelementptr inbounds ([4 x i8], [4 x
    // i8]* @.str, i32 0, i32 0), align 8, !dbg !0
    // @.str = private unnamed_addr constant [4 x i8] c"  |\00", align 1
    // Therefore code can't be produced in llvm order
    //
    // for (Module::global_iterator I = M.global_begin(), E = M.global_end();
    //     I != E; ++I) {
    //  GlobalVariable* inst = &*I;
    //  errs() << "SUSAN: global variable: " << *inst << "\n";
    //  declareOneGlobalVariable(&*I);
    //}
    //
    std::set<GlobalVariable *> declared;
    std::queue<GlobalVariable *> workingList;
    for (Module::global_iterator I = M.global_begin(), E = M.global_end();
         I != E; ++I) {
      GlobalVariable *glob = &*I;
      workingList.push(glob);
    }

    while (!workingList.empty()) {
      bool isReady2Declare = true;
      GlobalVariable *currGlob = workingList.front();
      errs() << "SUSAN: currGlob: " << *currGlob << "\n";
      workingList.pop();

      if (currGlob->hasInitializer()) {
        Constant *initializer = currGlob->getInitializer();
        if (ConstantExpr *initializerExpr = dyn_cast<ConstantExpr>(initializer))
          if (GEPOperator *initializerOp =
                  dyn_cast<GEPOperator>(initializerExpr)) {
            Value *v = initializerOp->getPointerOperand();
            if (GlobalVariable *useGlob = dyn_cast<GlobalVariable>(v))
              if (declared.find(useGlob) == declared.end()) {
                workingList.push(currGlob);
                isReady2Declare = false;
              }
          }

        if (ConstantArray *initArr = dyn_cast<ConstantArray>(initializer)) {
          for (Value *Element : initArr->operands())
            if (GEPOperator *initializerOp = dyn_cast<GEPOperator>(Element)) {
              Value *v = initializerOp->getPointerOperand();
              if (GlobalVariable *globElement = dyn_cast<GlobalVariable>(v))
                if (declared.find(globElement) == declared.end()) {
                  workingList.push(currGlob);
                  isReady2Declare = false;
                  break;
                }
            }
        }

        if (ConstantStruct *initStruct =
                dyn_cast<ConstantStruct>(initializer)) {
          for (Value *Element : initStruct->operands())
            if (GEPOperator *initializerOp = dyn_cast<GEPOperator>(Element)) {
              Value *v = initializerOp->getPointerOperand();
              if (GlobalVariable *globElement = dyn_cast<GlobalVariable>(v))
                if (declared.find(globElement) == declared.end()) {
                  workingList.push(currGlob);
                  isReady2Declare = false;
                  break;
                }
            }
        }
      }

      if (isReady2Declare) {
        declareOneGlobalVariable(currGlob);
        declared.insert(currGlob);
      }
    }
  }

  // Alias declarations...
  if (!M.alias_empty()) {
    Out << "\n/* External Alias Declarations */\n";
    for (Module::alias_iterator I = M.alias_begin(), E = M.alias_end(); I != E;
         ++I) {
      cwriter_assert(!I->isDeclaration() &&
                     !isEmptyType(I->getType()->getPointerElementType()));
      // if (I->hasLocalLinkage())
      //   continue; // Internal Global

      if (I->hasDLLImportStorageClass())
        Out << "__declspec(dllimport) ";
      else if (I->hasDLLExportStorageClass())
        Out << "__declspec(dllexport) ";

      // Thread Local Storage
      if (I->isThreadLocal())
        Out << "__thread ";

      Type *ElTy = I->getType()->getElementType();
      unsigned Alignment = I->getBaseObject()->getAlignment();
      bool IsOveraligned =
          Alignment && Alignment > TD->getABITypeAlignment(ElTy);
      if (IsOveraligned) {
        headerUseMsAlign();
        Out << "__MSALIGN__(" << Alignment << ") ";
      }
      // GetValueName would resolve the alias, which is not what we want,
      // so use getName directly instead (assuming that the Alias has a name...)
      printTypeName(Out, ElTy, false) << " *" << I->getName();
      if (IsOveraligned)
        Out << " __attribute__((aligned(" << Alignment << ")))";

      if (I->hasExternalWeakLinkage()) {
        headerUseExternalWeak();
        Out << " __EXTERNAL_WEAK__";
      }
      Out << " = ";
      writeOperand(I->getAliasee(), ContextStatic);
      Out << ";\n";
    }
  }

  Out << "\n\n/* LLVM Intrinsic Builtin Function Bodies */\n";

  // Loop over all select operations
  if (!SelectDeclTypes.empty())
    headerUseForceInline();
  for (std::set<Type *>::iterator it = SelectDeclTypes.begin(),
                                  end = SelectDeclTypes.end();
       it != end; ++it) {
    // static __forceinline Rty llvm_select_u8x4(<bool x 4> condition, <u8 x 4>
    // iftrue, <u8 x 4> ifnot) {
    //   Rty r = {
    //     condition[0] ? iftrue[0] : ifnot[0],
    //     condition[1] ? iftrue[1] : ifnot[1],
    //     condition[2] ? iftrue[2] : ifnot[2],
    //     condition[3] ? iftrue[3] : ifnot[3]
    //   };
    //   return r;
    // }
    Out << "static __forceinline ";
    printTypeNameUnaligned(Out, *it, false);
    Out << " llvm_select_";
    printTypeString(Out, *it, false);
    Out << "(";
    if (isa<VectorType>(*it))
#if LLVM_VERSION_MAJOR >= 12
      printTypeNameUnaligned(
          Out,
          VectorType::get(Type::getInt1Ty((*it)->getContext()),
                          cast<VectorType>(*it)->getElementCount()),
          false);
#else
      printTypeNameUnaligned(
          Out,
          VectorType::get(Type::getInt1Ty((*it)->getContext()),
                          cast<VectorType>(*it)->getNumElements()),
          false);
#endif
    else
      Out << "bool";
    Out << " condition, ";
    printTypeNameUnaligned(Out, *it, false);
    Out << " iftrue, ";
    printTypeNameUnaligned(Out, *it, false);
    Out << " ifnot) {\n  ";
    printTypeNameUnaligned(Out, *it, false);
    Out << " r;\n";
    if (isa<VectorType>(*it)) {
      unsigned n, l = NumberOfElements(cast<VectorType>(*it));
      for (n = 0; n < l; n++) {
        Out << "  r.vector[" << n << "] = condition.vector[" << n
            << "] ? iftrue.vector[" << n << "] : ifnot.vector[" << n << "];\n";
      }
    } else {
      Out << "  r = condition ? iftrue : ifnot;\n";
    }
    Out << "  return r;\n}\n";
  }

  // Loop over all compare operations
  if (!CmpDeclTypes.empty())
    headerUseForceInline();
  for (std::set<std::pair<CmpInst::Predicate, VectorType *>>::iterator
           it = CmpDeclTypes.begin(),
           end = CmpDeclTypes.end();
       it != end; ++it) {
    // static __forceinline <bool x 4> llvm_icmp_ge_u8x4(<u8 x 4> l, <u8 x 4> r)
    // {
    //   Rty c = {
    //     l[0] >= r[0],
    //     l[1] >= r[1],
    //     l[2] >= r[2],
    //     l[3] >= r[3],
    //   };
    //   return c;
    // }
    unsigned n, l = NumberOfElements((*it).second);
    VectorType *RTy =
#if LLVM_VERSION_MAJOR >= 12
        VectorType::get(Type::getInt1Ty((*it).second->getContext()), l,
                        (*it).second->getElementCount().isScalar());
#else
        VectorType::get(Type::getInt1Ty((*it).second->getContext()), l);
#endif
    bool isSigned = CmpInst::isSigned((*it).first);
    Out << "static __forceinline ";
    printTypeName(Out, RTy, isSigned);
    const auto Pred = (*it).first;
    if (CmpInst::isFPPredicate(Pred)) {
      headerUseFCmpOp(Pred);
      Out << " llvm_fcmp_";
    } else
      Out << " llvm_icmp_";
    Out << getCmpPredicateName(Pred) << "_";
    printTypeString(Out, (*it).second, isSigned);
    Out << "(";
    printTypeNameUnaligned(Out, (*it).second, isSigned);
    Out << " l, ";
    printTypeNameUnaligned(Out, (*it).second, isSigned);
    Out << " r) {\n  ";
    printTypeName(Out, RTy, isSigned);
    Out << " c;\n";
    for (n = 0; n < l; n++) {
      Out << "  c.vector[" << n << "] = ";
      if (CmpInst::isFPPredicate((*it).first)) {
        Out << "llvm_fcmp_" << getCmpPredicateName((*it).first) << "(l.vector["
            << n << "], r.vector[" << n << "]);\n";
      } else {
        Out << "l.vector[" << n << "]";
        switch ((*it).first) {
        case CmpInst::ICMP_EQ:
          Out << " == ";
          break;
        case CmpInst::ICMP_NE:
          Out << " != ";
          break;
        case CmpInst::ICMP_ULE:
        case CmpInst::ICMP_SLE:
          Out << " <= ";
          break;
        case CmpInst::ICMP_UGE:
        case CmpInst::ICMP_SGE:
          Out << " >= ";
          break;
        case CmpInst::ICMP_ULT:
        case CmpInst::ICMP_SLT:
          Out << " < ";
          break;
        case CmpInst::ICMP_UGT:
        case CmpInst::ICMP_SGT:
          Out << " > ";
          break;
        default:
          DBG_ERRS("Invalid icmp predicate!" << (*it).first);
          errorWithMessage("invalid icmp predicate");
        }
        Out << "r.vector[" << n << "];\n";
      }
    }
    Out << "  return c;\n}\n";
  }

  // Loop over all (vector) cast operations
  if (!CastOpDeclTypes.empty())
    headerUseForceInline();
  for (std::set<
           std::pair<CastInst::CastOps, std::pair<Type *, Type *>>>::iterator
           it = CastOpDeclTypes.begin(),
           end = CastOpDeclTypes.end();
       it != end; ++it) {
    // static __forceinline <u32 x 4> llvm_ZExt_u8x4_u32x4(<u8 x 4> in) { //
    // Src->isVector == Dst->isVector
    //   Rty out = {
    //     in[0],
    //     in[1],
    //     in[2],
    //     in[3]
    //   };
    //   return out;
    // }
    // static __forceinline u32 llvm_BitCast_u8x4_u32(<u8 x 4> in) { //
    // Src->bitsSize == Dst->bitsSize
    //   union {
    //     <u8 x 4> in;
    //     u32 out;
    //   } cast;
    //   cast.in = in;
    //   return cast.out;
    // }
    CastInst::CastOps opcode = (*it).first;
    Type *SrcTy = (*it).second.first;
    Type *DstTy = (*it).second.second;
    bool SrcSigned, DstSigned;
    switch (opcode) {
    default:
      SrcSigned = false;
      DstSigned = false;
      break;
    case Instruction::SIToFP:
      SrcSigned = true;
      DstSigned = false;
      break;
    case Instruction::FPToSI:
      SrcSigned = false;
      DstSigned = true;
      break;
    case Instruction::SExt:
      SrcSigned = true;
      DstSigned = true;
      break;
    }

    Out << "static __forceinline ";
    printTypeName(Out, DstTy, DstSigned);
    Out << " llvm_" << Instruction::getOpcodeName(opcode) << "_";
    printTypeString(Out, SrcTy, false);
    Out << "_";
    printTypeString(Out, DstTy, false);
    Out << "(";
    printTypeNameUnaligned(Out, SrcTy, SrcSigned);
    Out << " in) {\n";
    if (opcode == Instruction::BitCast) {
      Out << "  union {\n    ";
      printTypeName(Out, SrcTy, SrcSigned);
      Out << " in;\n    ";
      printTypeName(Out, DstTy, DstSigned);
      Out << " out;\n  } cast;\n";
      Out << "  cast.in = in;\n  return cast.out;\n}\n";
    } else if (isa<VectorType>(DstTy)) {
      Out << "  ";
      printTypeName(Out, DstTy, DstSigned);
      Out << " out;\n";
      unsigned n, l = NumberOfElements(cast<VectorType>(DstTy));
      cwriter_assert(cast<VectorType>(SrcTy)->getNumElements() == l);
      for (n = 0; n < l; n++) {
        Out << "  out.vector[" << n << "] = in.vector[" << n << "];\n";
      }
      Out << "  return out;\n}\n";
    } else {
      Out << "#ifndef __emulate_i128\n";
      // easy case first: compiler supports i128 natively
      Out << "  return in;\n";
      Out << "#else\n";
      Out << "  ";
      printTypeName(Out, DstTy, DstSigned);
      Out << " out;\n";
      Out << "  LLVM";
      switch (opcode) {
      case Instruction::UIToFP:
        Out << "UItoFP";
        break;
      case Instruction::SIToFP:
        Out << "SItoFP";
        break;
      case Instruction::Trunc:
        Out << "Trunc";
        break;
      case Instruction::FPExt:
        Out << "FPExt";
        break;
      case Instruction::FPTrunc:
        Out << "FPTrunc";
        break;
      case Instruction::ZExt:
        Out << "ZExt";
        break;
      case Instruction::FPToUI:
        Out << "FPtoUI";
        break;
      case Instruction::SExt:
        Out << "SExt";
        break;
      case Instruction::FPToSI:
        Out << "FPtoSI";
        break;
      default:
        DBG_ERRS("Invalid cast opcode: " << opcode);
        errorWithMessage("Invalid cast opcode for i128");
      }
      Out << "(" << SrcTy->getPrimitiveSizeInBits() << ", &in, "
          << DstTy->getPrimitiveSizeInBits() << ", &out);\n";
      Out << "  return out;\n";
      Out << "#endif\n";
      Out << "}\n";
    }
  }

  // Loop over all simple vector operations
  if (!InlineOpDeclTypes.empty())
    headerUseForceInline();
  for (std::set<std::pair<unsigned, Type *>>::iterator
           it = InlineOpDeclTypes.begin(),
           end = InlineOpDeclTypes.end();
       it != end; ++it) {
    // static __forceinline <u32 x 4> llvm_BinOp_u32x4(<u32 x 4> a, <u32 x 4> b)
    // {
    //   Rty r = {
    //      a[0] OP b[0],
    //      a[1] OP b[1],
    //      a[2] OP b[2],
    //      a[3] OP b[3],
    //   };
    //   return r;
    // }
    unsigned opcode = (*it).first;
    Type *OpTy = (*it).second;
    Type *ElemTy =
        isa<VectorType>(OpTy) ? cast<VectorType>(OpTy)->getElementType() : OpTy;
    bool shouldCast;
    bool isSigned;
    opcodeNeedsCast(opcode, shouldCast, isSigned);

    Out << "static __forceinline ";
    printTypeName(Out, OpTy);
    if (opcode == BinaryNeg) {
      Out << " llvm_neg_";
      printTypeString(Out, OpTy, false);
      Out << "(";
      printTypeNameUnaligned(Out, OpTy, isSigned);
      Out << " a) {\n  ";
    } else if (opcode == BinaryNot) {
      Out << " llvm_not_";
      printTypeString(Out, OpTy, false);
      Out << "(";
      printTypeNameUnaligned(Out, OpTy, isSigned);
      Out << " a) {\n  ";
    } else {
      Out << " llvm_" << Instruction::getOpcodeName(opcode) << "_";
      printTypeString(Out, OpTy, false);
      Out << "(";
      printTypeNameUnaligned(Out, OpTy, isSigned);
      Out << " a, ";
      printTypeNameUnaligned(Out, OpTy, isSigned);
      Out << " b) {\n  ";
    }

    printTypeName(Out, OpTy);
    // C can't handle non-power-of-two integer types
    unsigned mask = 0;
    if (ElemTy->isIntegerTy()) {
      IntegerType *ITy = static_cast<IntegerType *>(ElemTy);
#if LLVM_VERSION_MAJOR <= 10
      if (!ITy->isPowerOf2ByteWidth())
#else
      if (!IsPowerOfTwo(ITy->getBitWidth()))
#endif
        mask = ITy->getBitMask();
    }

    if (isa<VectorType>(OpTy)) {
      Out << " r;\n";
      unsigned n, l = NumberOfElements(cast<VectorType>(OpTy));
      for (n = 0; n < l; n++) {
        Out << "  r.vector[" << n << "] = ";
        if (mask)
          Out << "(";
        if (opcode == BinaryNeg) {
          Out << "-a.vector[" << n << "]";
        } else if (opcode == BinaryNot) {
          Out << "~a.vector[" << n << "]";
        } else if (opcode == Instruction::FRem) {
          // Output a call to fmod/fmodf instead of emitting a%b
          if (ElemTy->isFloatTy())
            Out << "fmodf(a.vector[" << n << "], b.vector[" << n << "])";
          else if (ElemTy->isDoubleTy())
            Out << "fmod(a.vector[" << n << "], b.vector[" << n << "])";
          else // all 3 flavors of long double
            Out << "fmodl(a.vector[" << n << "], b.vector[" << n << "])";
        } else {
          Out << "a.vector[" << n << "]";
          switch (opcode) {
          case Instruction::Add:
          case Instruction::FAdd:
            Out << " + ";
            break;
          case Instruction::Sub:
          case Instruction::FSub:
            Out << " - ";
            break;
          case Instruction::Mul:
          case Instruction::FMul:
            Out << " * ";
            break;
          case Instruction::URem:
          case Instruction::SRem:
          case Instruction::FRem:
            Out << " % ";
            break;
          case Instruction::UDiv:
          case Instruction::SDiv:
          case Instruction::FDiv:
            Out << " / ";
            break;
          case Instruction::And:
            Out << " & ";
            break;
          case Instruction::Or:
            Out << " | ";
            break;
          case Instruction::Xor:
            Out << " ^ ";
            break;
          case Instruction::Shl:
            Out << " << ";
            break;
          case Instruction::LShr:
          case Instruction::AShr:
            Out << " >> ";
            break;
          default:
            DBG_ERRS("Invalid operator type ! " << opcode);
            errorWithMessage("invalid operator type");
          }
          Out << "b.vector[" << n << "]";
        }
        if (mask)
          Out << ") & " << mask;
        Out << ";\n";
      }

    } else if (OpTy->getPrimitiveSizeInBits() > 64) {
      Out << " r;\n";
      Out << "#ifndef __emulate_i128\n";
      // easy case first: compiler supports i128 natively
      Out << "  r = ";
      if (opcode == BinaryNeg) {
        Out << "-a;\n";
      } else if (opcode == BinaryNot) {
        Out << "~a;\n";
      } else {
        Out << "a";
        switch (opcode) {
        case Instruction::Add:
        case Instruction::FAdd:
          Out << " + ";
          break;
        case Instruction::Sub:
        case Instruction::FSub:
          Out << " - ";
          break;
        case Instruction::Mul:
        case Instruction::FMul:
          Out << " * ";
          break;
        case Instruction::URem:
        case Instruction::SRem:
          Out << " % ";
          break;
        case Instruction::UDiv:
        case Instruction::SDiv:
        case Instruction::FDiv:
          Out << " / ";
          break;
        case Instruction::And:
          Out << " & ";
          break;
        case Instruction::Or:
          Out << " | ";
          break;
        case Instruction::Xor:
          Out << " ^ ";
          break;
        case Instruction::Shl:
          Out << " << ";
          break;
        case Instruction::LShr:
        case Instruction::AShr:
          Out << " >> ";
          break;
        default:
          DBG_ERRS("Invalid operator type !" << opcode);
          errorWithMessage("invalid operator type");
        }
        Out << "b;\n";
      }
      Out << "#else\n";
      // emulated twos-complement i128 math
      if (opcode == BinaryNeg) {
        Out << "  r.hi = ~a.hi;\n";
        Out << "  r.lo = ~a.lo + 1;\n";
        Out << "  if (r.lo == 0) r.hi += 1;\n"; // overflow: carry the one
      } else if (opcode == BinaryNot) {
        Out << "  r.hi = ~a.hi;\n";
        Out << "  r.lo = ~a.lo;\n";
      } else if (opcode == Instruction::And) {
        Out << "  r.hi = a.hi & b.hi;\n";
        Out << "  r.lo = a.lo & b.lo;\n";
      } else if (opcode == Instruction::Or) {
        Out << "  r.hi = a.hi | b.hi;\n";
        Out << "  r.lo = a.lo | b.lo;\n";
      } else if (opcode == Instruction::Xor) {
        Out << "  r.hi = a.hi ^ b.hi;\n";
        Out << "  r.lo = a.lo ^ b.lo;\n";
      } else if (opcode ==
                 Instruction::Shl) { // reminder: undef behavior if b >= 128
        Out << "  if (b.lo >= 64) {\n";
        Out << "    r.hi = (a.lo << (b.lo - 64));\n";
        Out << "    r.lo = 0;\n";
        Out << "  } else if (b.lo == 0) {\n";
        Out << "    r.hi = a.hi;\n";
        Out << "    r.lo = a.lo;\n";
        Out << "  } else {\n";
        Out << "    r.hi = (a.hi << b.lo) | (a.lo >> (64 - b.lo));\n";
        Out << "    r.lo = a.lo << b.lo;\n";
        Out << "  }\n";
      } else {
        // everything that hasn't been manually implemented above
        Out << "  LLVM";
        switch (opcode) {
        // case BinaryNeg: Out << "Neg"; break;
        // case BinaryNot: Out << "FlipAllBits"; break;
        case Instruction::Add:
          Out << "Add";
          break;
        case Instruction::FAdd:
          Out << "FAdd";
          break;
        case Instruction::Sub:
          Out << "Sub";
          break;
        case Instruction::FSub:
          Out << "FSub";
          break;
        case Instruction::Mul:
          Out << "Mul";
          break;
        case Instruction::FMul:
          Out << "FMul";
          break;
        case Instruction::URem:
          Out << "URem";
          break;
        case Instruction::SRem:
          Out << "SRem";
          break;
        case Instruction::UDiv:
          Out << "UDiv";
          break;
        case Instruction::SDiv:
          Out << "SDiv";
          break;
        case Instruction::FDiv:
          Out << "FDiv";
          break;
        // case Instruction::And:  Out << "And"; break;
        // case Instruction::Or:   Out << "Or"; break;
        // case Instruction::Xor:  Out << "Xor"; break;
        // case Instruction::Shl: Out << "Shl"; break;
        case Instruction::LShr:
          Out << "LShr";
          break;
        case Instruction::AShr:
          Out << "AShr";
          break;
        default:
          DBG_ERRS("Invalid operator type !" << opcode);
          errorWithMessage("invalid operator type");
        }
        Out << "(16, &a, &b, &r);\n";
      }
      Out << "#endif\n";

    } else {
      Out << " r = ";
      if (mask)
        Out << "(";
      if (opcode == BinaryNeg) {
        Out << "-a";
      } else if (opcode == BinaryNot) {
        Out << "~a";
      } else if (opcode == Instruction::FRem) {
        // Output a call to fmod/fmodf instead of emitting a%b
        if (ElemTy->isFloatTy())
          Out << "fmodf(a, b)";
        else if (ElemTy->isDoubleTy())
          Out << "fmod(a, b)";
        else // all 3 flavors of long double
          Out << "fmodl(a, b)";
      } else {
        Out << "a";
        switch (opcode) {
        case Instruction::Add:
        case Instruction::FAdd:
          Out << " + ";
          break;
        case Instruction::Sub:
        case Instruction::FSub:
          Out << " - ";
          break;
        case Instruction::Mul:
        case Instruction::FMul:
          Out << " * ";
          break;
        case Instruction::URem:
        case Instruction::SRem:
        case Instruction::FRem:
          Out << " % ";
          break;
        case Instruction::UDiv:
        case Instruction::SDiv:
        case Instruction::FDiv:
          Out << " / ";
          break;
        case Instruction::And:
          Out << " & ";
          break;
        case Instruction::Or:
          Out << " | ";
          break;
        case Instruction::Xor:
          Out << " ^ ";
          break;
        case Instruction::Shl:
          Out << " << ";
          break;
        case Instruction::LShr:
        case Instruction::AShr:
          Out << " >> ";
          break;
        default:
          DBG_ERRS("Invalid operator type !" << opcode);
          errorWithMessage("invalid operator type");
        }
        Out << "b";
        if (mask)
          Out << ") & " << mask;
      }
      Out << ";\n";
    }
    Out << "  return r;\n}\n";
  }

  if (!CtorDeclTypes.empty())
    headerUseForceInline();
  // Loop over all inline constructors
  for (std::set<Type *>::iterator it = CtorDeclTypes.begin(),
                                  end = CtorDeclTypes.end();
       it != end; ++it) {
    // static __forceinline <u32 x 4> llvm_ctor_u32x4(u32 x1, u32 x2, u32 x3,
    // u32 x4) {
    //   Rty r = {
    //     x1, x2, x3, x4
    //   };
    //   return r;
    // }
    Out << "static __forceinline ";
    printTypeName(Out, *it);
    Out << " llvm_ctor_";
    printTypeString(Out, *it, false);
    Out << "(";
    StructType *STy = dyn_cast<StructType>(*it);
    ArrayType *ATy = dyn_cast<ArrayType>(*it);
    VectorType *VTy = dyn_cast<VectorType>(*it);
    // errs() << "SUSAN: STy: " << *STy << "\n";
    // errs() << "SUSAN: ATy: " << *ATy << "\n";
    // errs() << "SUSAN: VTy: " << *VTy << "\n";
    unsigned e = (STy ? STy->getNumElements()
                      : (ATy ? ATy->getNumElements() : NumberOfElements(VTy)));
    bool printed = false;
    for (unsigned i = 0; i != e; ++i) {
      Type *ElTy = STy ? STy->getElementType(i) : nullptr;
      if (isEmptyType(ElTy))
        Out << " /* ";
      else if (printed)
        Out << ", ";
      printTypeNameUnaligned(Out, ElTy);
      Out << " x" << i;
      if (isEmptyType(ElTy))
        Out << " */";
      else
        printed = true;
    }
    Out << ") {\n  ";
    printTypeName(Out, *it);
    Out << " r;";
    for (unsigned i = 0; i != e; ++i) {
      Type *ElTy = STy ? STy->getElementType(i) : nullptr;
      if (isEmptyType(ElTy))
        continue;
      if (STy)
        Out << "\n  r." << getFieldName(STy) << i << " = x" << i << ";";
      else if (ATy)
        Out << "\n  r.array[" << i << "] = x" << i << ";";
      else if (VTy)
        Out << "\n  r.vector[" << i << "] = x" << i << ";";
      else
        cwriter_assert(0);
    }
    Out << "\n  return r;\n}\n";
  }

  // Emit definitions of the intrinsics.
  if (!intrinsicsToDefine.empty())
    headerUseForceInline();
  for (SmallVector<Function *, 16>::iterator I = intrinsicsToDefine.begin(),
                                             E = intrinsicsToDefine.end();
       I != E; ++I) {
    printIntrinsicDefinition(**I, Out);
  }

  if (!M.empty())
    Out << "\n\n/* Function Bodies */\n";

  if (!FCmpOps.empty())
    headerUseForceInline();

  generateCompilerSpecificCode(OutHeaders, TD);

  // Loop over all fcmp compare operations. We do that after
  // generateCompilerSpecificCode because we need __forceinline!
  if (FCmpOps.erase(FCmpInst::FCMP_ORD)) {
    defineFCmpOp(OutHeaders, FCmpInst::FCMP_ORD);
  }
  if (FCmpOps.erase(FCmpInst::FCMP_UNO)) {
    defineFCmpOp(OutHeaders, FCmpInst::FCMP_UNO);
  }
  for (auto Pred : FCmpOps) {
    defineFCmpOp(OutHeaders, Pred);
  }
  FCmpOps.clear();
}

void CWriter::declareOneGlobalVariable(GlobalVariable *I) {
  if (I->isDeclaration())
    return;

  // Ignore special globals, such as debug info.
  if (getGlobalVariableClass(&*I))
    return;

  if (I->hasDLLImportStorageClass())
    Out << "__declspec(dllimport) ";
  else if (I->hasDLLExportStorageClass())
    Out << "__declspec(dllexport) ";

  // if (I->hasLocalLinkage())
  //   Out << "static ";

  // Thread Local Storage
  if (I->isThreadLocal())
    Out << "__thread ";

  Type *ElTy = I->getType()->getElementType();
  // unsigned Alignment = I->getAlignment();
  // bool IsOveraligned = Alignment && Alignment >
  // TD->getABITypeAlignment(ElTy); if (IsOveraligned) {
  //   headerUseMsAlign();
  //   Out << "__MSALIGN__(" << Alignment << ") ";
  // }
  // printTypeNameForAddressableValue(Out, ElTy, false);
  // Out << ' ' << GetValueName(I);
  // if (IsOveraligned)
  //   Out << " __attribute__((aligned(" << Alignment << ")))";

  // if (I->hasLinkOnceLinkage())
  //   Out << " __attribute__((common))";
  // else if (I->hasWeakLinkage()) {
  //   headerUseAttributeWeak();
  //   Out << " __ATTRIBUTE_WEAK__";
  // } else if (I->hasCommonLinkage()) {
  //   headerUseAttributeWeak();
  //   Out << " __ATTRIBUTE_WEAK__";
  // }

  // if (I->hasHiddenVisibility()) {
  //   headerUseHidden();
  //   Out << " __HIDDEN__";
  // }

  printTypeNameForAddressableValue(Out, ElTy, false);
  Out << ' ' << GetValueName(I);
  nameDict.GlobalVars.insert(GetValueName(I));
  ArrayType *ArrTy = dyn_cast<ArrayType>(ElTy);
  while (ArrTy) {
    Out << "[" << ArrTy->getNumElements() << "]";
    ArrTy = dyn_cast<ArrayType>(ArrTy->getElementType());
  }
  // If the initializer is not null, emit the initializer.  If it is null,
  // we try to avoid emitting large amounts of zeros.  The problem with
  // this, however, occurs when the variable has weak linkage.  In this
  // case, the assembler will complain about the variable being both weak
  // and common, so we disable this optimization.
  // FIXME common linkage should avoid this problem.
  if (!I->getInitializer()->isNullValue()) {
    Out << " = ";
    writeOperand(I->getInitializer(), ContextStatic);
  } else if (I->hasWeakLinkage()) {
    // We have to specify an initializer, but it doesn't have to be
    // complete.  If the value is an aggregate, print out { 0 }, and let
    // the compiler figure out the rest of the zeros.
    Out << " = ";
    if (I->getInitializer()->getType()->isStructTy() ||
        I->getInitializer()->getType()->isVectorTy()) {
      Out << "{ 0 }";
    } else if (I->getInitializer()->getType()->isArrayTy()) {
      // As with structs and vectors, but with an extra set of braces
      // because arrays are wrapped in structs.
      Out << "{ { 0 } }";
    } else {
      // Just print it out normally.
      writeOperand(I->getInitializer(), ContextStatic);
    }
  }
  Out << ";\n";
}

void CWriter::markLoopIrregularExits(Function &F) {
  std::list<Loop *> loops(LI->begin(), LI->end());
  while (!loops.empty()) {
    Loop *L = loops.front();
    loops.pop_front();

    SmallVector<BasicBlock *, 8> ExitBlocks, ExitingBlocks;
    SmallVector<std::pair<BasicBlock *, BasicBlock *>, 8> ExitEdges;
    L->getExitBlocks(ExitBlocks);
    L->getExitingBlocks(ExitingBlocks);
    L->getExitEdges(ExitEdges);
    for (auto edge : ExitEdges) {
      BasicBlock *exitingBB = edge.first;
      if (exitingBB != L->getHeader()) {
        irregularLoopExits.insert(edge);
      }
    }

    loops.insert(loops.end(), L->getSubLoops().begin(), L->getSubLoops().end());
  }
}

Instruction *CWriter::headerIsExiting(Loop *L, bool &negateCondition,
                                      BranchInst *brInst) {
  errs() << "SUSAN: trying to get exit for loop: " << *L << "\n";
  if (!brInst) {
    BasicBlock *header = L->getHeader();
    Instruction *term = header->getTerminator();
    brInst = dyn_cast<BranchInst>(term);
    assert(brInst && brInst->isConditional() &&
           "exit condition is not a conditional branch inst?");
  }
  SmallVector<BasicBlock *, 1> ExitingBlocks;
  L->getExitingBlocks(ExitingBlocks);
  for (SmallVector<BasicBlock *, 1>::iterator i = ExitingBlocks.begin(),
                                              e = ExitingBlocks.end();
       i != e; ++i) {
    BasicBlock *exit = *i;
    errs() << "SUSAN exitBB: " << *exit << "\n";
    if (exit == L->getHeader()) {
      Value *cond = brInst->getCondition();
      if (isa<CmpInst>(cond) || isa<UnaryInstruction>(cond) ||
          isa<BinaryOperator>(cond) || isa<CallInst>(cond)) {
        if (isa<CallInst>(cond))
          loopCondCalls.insert(dyn_cast<CallInst>(cond));
        BasicBlock *succ0 = brInst->getSuccessor(0);
        if (LI->getLoopFor(succ0) != L)
          negateCondition = true;
        return cast<Instruction>(cond);
      }
      /*else if(isa<CmpInst>(opnd1) || isa<UnaryInstruction>(opnd1) ||
      isa<BinaryOperator>(opnd1) || isa<CallInst>(opnd1)){
        if(isa<CallInst>(opnd1))
          loopCondCalls.insert(dyn_cast<CallInst>(opnd1));
        negateCondition = true;
        return cast<Instruction>(opnd1);
      }*/
      else
        return nullptr;
    }
  }
  return nullptr;
}

bool isPureBranchBB(BasicBlock *BB) {
  Instruction *term = BB->getTerminator();
  std::queue<Instruction *> toVisit;
  toVisit.push(term);

  std::set<Instruction *> BrRelated;
  while (!toVisit.empty()) {
    Instruction *currInst = toVisit.front();
    BrRelated.insert(currInst);
    toVisit.pop();

    for (Use &U : currInst->operands()) {
      if (Instruction *inst = dyn_cast<Instruction>(U.get())) {
        if (inst->getParent() == BB) {
          toVisit.push(inst);
        }
      }
    }
  }

  for (auto &inst : *BB) {
    if (BrRelated.find(&inst) == BrRelated.end())
      return false;
  }

  return true;
}

void CWriter::markGotoBranches(Function &F) {
  for (auto &BB : F) {
    BranchInst *br = dyn_cast<BranchInst>(BB.getTerminator());

    // if it's unconditional, for right now, mark it NOT as goto
    // need an analysis later to identify unconditional goto statements
    if (!br || !br->isConditional())
      continue;

    // if it's in loop header or latch then it's not goto
    Loop *L = LI->getLoopFor(&BB);
    if (L && (&BB == L->getHeader() || &BB == L->getLoopLatch()))
      continue;

    // if it's loop predecessor it's not goto
    bool isPredecessor = false;
    for (auto succ = succ_begin(br); succ != succ_end(br); ++succ) {
      BasicBlock *succBB = *succ;
      Loop *L = LI->getLoopFor(succBB);
      if (L && &BB == L->getLoopPredecessor()) {
        isPredecessor = true;
        break;
      }
    }
    if (isPredecessor)
      continue;

    bool isIfReturn = isExitingFunction(br->getSuccessor(0)) ||
                      isExitingFunction(br->getSuccessor(1));
    if (isIfReturn)
      continue;

    // if the branch is not a region entry and not branching to consecutive
    // block, then it's a goto
    Region *R = RI->getRegionFor(&BB);
    if (&BB != R->getEntry()) {
      for (auto succ = succ_begin(br); succ != succ_end(br); ++succ) {
        BasicBlock *succBB = *succ;
        if (std::next(Function::iterator(&BB)) != Function::iterator(succBB)) {
          // excepion: if all instructions within this bb do is to calculate for
          // the branch, then it can still be translated as an if else block
          if (!isPureBranchBB(&BB)) {
            errs() << "found goto branch:" << *br << "\n";
            errs() << "from BB " << BB << "\n";
            // gotoBranches.insert(br);
          }
        }
      }
    }

    // if we found goto branches, mark its successors as need to print labels
    for (auto br : gotoBranches) {
      for (auto succ = succ_begin(br); succ != succ_end(br); ++succ) {
        BasicBlock *succBB = *succ;
        // printLabels.insert(succBB);
      }
    }
  }
}

// If branch criterias:
// 1. conditional branch
// 2. branch is not from weird loop exits
void CWriter::markIfBranches(Function &F, std::set<BasicBlock *> *visitedBBs) {
  for (auto &BB : F) {
    // for splitted nodes
    // if(visitedBBs->find(&BB) != visitedBBs->end())
    //   continue;
    // visitedBBs->insert(&BB);
    Instruction *term = BB.getTerminator();
    BranchInst *br = dyn_cast<BranchInst>(term);
    if (br && gotoBranches.find(br) != gotoBranches.end())
      continue;

    if (br && br->isConditional()) {
      Loop *L = LI->getLoopFor(&BB);
      bool negateCondition = false;
      if (L && L->getHeader() == &BB && headerIsExiting(L, negateCondition))
        continue;

      // for loop latches
      if (L && L->getLoopLatch() == &BB)
        continue;

      ifBranches.push_back(br);
    }
  }
}

// Split the nodes that have two or more predecessors marked by if statement
void CWriter::NodeSplitting(Function &F) {

  //  std::map<BasicBlock*, int> numOfMarkedPredecessors;
  //
  //  for(auto &BB : F){
  //    // find the successors of a marked basic block
  //    for(auto &inst : BB){
  //      if(ifBranches.find(dyn_cast<BranchInst>(&inst)) != ifBranches.end()){
  //        for (auto succ = succ_begin(&inst);
  //           succ != succ_end(&inst); ++succ){
  //          BasicBlock *succBB = *succ;
  //          if(numOfMarkedPredecessors.find(succBB) ==
  //              numOfMarkedPredecessors.end())
  //            numOfMarkedPredecessors[succBB] = 1;
  //          else
  //            numOfMarkedPredecessors[succBB] ++;
  //        }
  //        break;
  //      }
  //    }
  //  }
  //
  //  for(auto const & [BB, numOfPred] : numOfMarkedPredecessors){
  //    if(numOfPred > 1){
  //      //errs() << "SUSAN: found a node to split:" << *BB << "\n";
  //      std::set<BasicBlock*> copysOfBB;
  //      std::vector<BasicBlock*> preds;
  //
  //      // Does node splitting with the following steps:
  //      // 1. copy the basic block n-1 times, n is the num of predecessor
  //      // 2. for each copied block, update the def use chain
  //      // 3. each copy gets one unique predecessor
  //      for(pred_iterator i=pred_begin(BB), e=pred_end(BB); i!=e; ++i){
  //        preds.push_back(*i);
  //      }
  //
  //
  //      ValueToValueMapTy VMap;
  //      for(long unsigned int i=1; i<preds.size(); i++){
  //        BasicBlock *pred = preds[i];
  //        //clone n-1 BBs for splitting
  //        BasicBlock *copyBB = CloneBasicBlock(BB, VMap,
  //        Twine(".")+Twine("splitted")+Twine(i));
  //
  //        //modify each instruction in copyBB to follow its own def-use chain
  //        for(auto &I : *copyBB){
  //          for (Use &U : I.operands()){
  //            Value *useVal = U.get();
  //            if(VMap.find(useVal)!=VMap.end()){
  //              U.set(VMap[useVal]);
  //            }
  //          }
  //        }
  //
  //        F.getBasicBlockList().push_back(copyBB);
  //
  //        //modify the CFG according to node splitting algorithm
  //        Instruction *term = pred->getTerminator();
  //        for(unsigned int i_succ = 0; i_succ<term->getNumSuccessors();
  //        ++i_succ){
  //          BasicBlock *succBB = term->getSuccessor(i_succ);
  //          if(succBB == BB){
  //            term->replaceSuccessorWith(BB, copyBB);
  //          }
  //        }
  //        copysOfBB.insert(copyBB);
  //        splittedBBs.insert(copyBB);
  //      }
  //
  //      for(auto &copyBB : copysOfBB){
  //        //modify the successor's phi node to include the copied block
  //        //errs() << "SUSAN: copyBB is " << *copyBB << "\n";
  //        Instruction *term = copyBB->getTerminator();
  //        for(unsigned int i_succ = 0; i_succ<term->getNumSuccessors();
  //        ++i_succ){
  //          BasicBlock *succBB = term->getSuccessor(i_succ);
  //          for (BasicBlock::iterator I = succBB->begin(); isa<PHINode>(I);
  //          ++I) {
  //            PHINode *phi = cast<PHINode>(I);
  //            //errs() << "SUSAN: PHINode: " << *phi << "\n";
  //            Value* originalVal = phi->getIncomingValueForBlock(BB);
  //            //errs() << "originalVal:" << *originalVal << "\n";
  //            if(isa<Instruction> (originalVal))
  //              phi->addIncoming(VMap[originalVal],copyBB);
  //            else if(isa<Constant> (originalVal))
  //              phi->addIncoming(originalVal, copyBB);
  //            else
  //              assert(0 && "PHI value is not Constant or Instruction,
  //              check!\n");
  //          }
  //        }
  //      }
  //      copysOfBB.insert(BB);
  //
  //      //In the future there might be a need to modify the phi nodes
  //      /*for(auto & bbToAdjust : copysOfBB){
  //        for (BasicBlock::iterator I = bbToAdjust->begin(); isa<PHINode>(I);
  //        ++I) {
  //          PHINode *PN = cast<PHINode>(I);
  //          errs() << "SUSAN: PHINode: " << *PN << "\n";
  //        }
  //      }*/
  //
  //    }
  //  }
}

/// Output all floating point constants that cannot be printed accurately...
void CWriter::printFloatingPointConstants(Function &F) {
  // Scan the module for floating point constants.  If any FP constant is used
  // in the function, we want to redirect it here so that we do not depend on
  // the precision of the printed form, unless the printed form preserves
  // precision.
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I)
    for (Instruction::op_iterator I_Op = I->op_begin(), E_Op = I->op_end();
         I_Op != E_Op; ++I_Op)
      if (const Constant *C = dyn_cast<Constant>(I_Op))
        printFloatingPointConstants(C);
  Out << '\n';
}

void CWriter::printFloatingPointConstants(const Constant *C) {
  // If this is a constant expression, recursively check for constant fp values.
  if (const ConstantExpr *CE = dyn_cast<ConstantExpr>(C)) {
    for (unsigned i = 0, e = CE->getNumOperands(); i != e; ++i)
      printFloatingPointConstants(CE->getOperand(i));
    return;
  }

  // Otherwise, check for a FP constant that we need to print.
  const ConstantFP *FPC = dyn_cast<ConstantFP>(C);
  if (FPC == nullptr ||
      // Do not put in FPConstantMap if safe.
      isFPCSafeToPrint(FPC) ||
      // Already printed this constant?
      FPConstantMap.has(FPC))
    return;

  unsigned Counter = FPConstantMap.getOrInsert(FPC);

  if (FPC->getType() == Type::getDoubleTy(FPC->getContext())) {
    double Val = FPC->getValueAPF().convertToDouble();
    uint64_t i = FPC->getValueAPF().bitcastToAPInt().getZExtValue();
    headerUseConstantDoubleTy();
    Out << "static const ConstantDoubleTy FPConstant" << Counter << " = 0x"
        << utohexstr(i) << "ULL;    /* " << Val << " */\n";
  } else if (FPC->getType() == Type::getFloatTy(FPC->getContext())) {
    float Val = FPC->getValueAPF().convertToFloat();
    uint32_t i = (uint32_t)FPC->getValueAPF().bitcastToAPInt().getZExtValue();
    headerUseConstantFloatTy();
    Out << "static const ConstantFloatTy FPConstant" << Counter << " = 0x"
        << utohexstr(i) << "U;    /* " << Val << " */\n";
  } else if (FPC->getType() == Type::getX86_FP80Ty(FPC->getContext())) {
    // api needed to prevent premature destruction
    const APInt api = FPC->getValueAPF().bitcastToAPInt();
    const uint64_t *p = api.getRawData();
    headerUseConstantFP80Ty();
    Out << "static const ConstantFP80Ty FPConstant" << Counter << " = { 0x"
        << utohexstr(p[0]) << "ULL, 0x" << utohexstr((uint16_t)p[1])
        << ",{0,0,0}"
        << "}; /* Long double constant */\n";
  } else if (FPC->getType() == Type::getPPC_FP128Ty(FPC->getContext()) ||
             FPC->getType() == Type::getFP128Ty(FPC->getContext())) {
    const APInt api = FPC->getValueAPF().bitcastToAPInt();
    const uint64_t *p = api.getRawData();
    headerUseConstantFP128Ty();
    Out << "static const ConstantFP128Ty FPConstant" << Counter << " = { 0x"
        << utohexstr(p[0]) << ", 0x" << utohexstr(p[1])
        << "}; /* Long double constant */\n";

  } else {
    errorWithMessage("Unknown float type!");
  }
}

static void defineBitCastUnion(raw_ostream &Out) {
  Out << "/* Helper union for bitcasts */\n";
  Out << "typedef union {\n";
  Out << "  uint32_t Int32;\n";
  Out << "  uint64_t Int64;\n";
  Out << "  float Float;\n";
  Out << "  double Double;\n";
  Out << "} llvmBitCastUnion;\n";
}

/// printSymbolTable - Run through symbol table looking for type names.  If a
/// type name is found, emit its declaration...
void CWriter::printModuleTypes(raw_ostream &Out) {
  if (headerIncBitCastUnion()) {
    defineBitCastUnion(Out);
  }

  // Keep track of which types have been printed so far.
  std::set<Type *> TypesPrinted;

  // Loop over all structures then push them into the stack so they are
  // printed in the correct order.
  Out << "\n/* Types Declarations */\n";

  // forward-declare all structs here first

  {
    std::set<Type *> TypesPrinted;
    for (auto it = TypedefDeclTypes.begin(), end = TypedefDeclTypes.end();
         it != end; ++it) {
      forwardDeclareStructs(Out, *it, TypesPrinted);
    }
  }

  Out << "\n/* Function definitions */\n";

  struct FunctionDefinition {
    FunctionType *FT;
    std::vector<FunctionType *> Dependencies;
    std::string NameToPrint;
  };

  std::vector<FunctionDefinition> FunctionTypeDefinitions;
  // Copy Function Types into indexable container
  for (auto &I : UnnamedFunctionIDs) {
    const auto &F = I.first;
    FunctionType *FT = F.first;
    std::vector<FunctionType *> FDeps;
    for (const auto P : F.first->params()) {
      // Handle arbitrarily deep pointer indirection
      Type *PP = P;
      while (PP->isPointerTy())
        PP = PP->getPointerElementType();
      if (auto *PPF = dyn_cast<FunctionType>(PP))
        FDeps.push_back(PPF);
    }
    std::string DeclString;
    raw_string_ostream TmpOut(DeclString);
    printFunctionDeclaration(TmpOut, F.first, F.second);
    TmpOut.flush();
    FunctionTypeDefinitions.emplace_back(
        FunctionDefinition{FT, FDeps, DeclString});
  }

  // Sort function types
  TopologicalSorter Sorter(FunctionTypeDefinitions.size());
  DenseMap<FunctionType *, int> TopologicalSortMap;
  // Add Vertices
  for (unsigned I = 0; I < FunctionTypeDefinitions.size(); I++) {
    TopologicalSortMap[FunctionTypeDefinitions[I].FT] = I;
  }
  // Add Edges
  for (unsigned I = 0; I < FunctionTypeDefinitions.size(); I++) {
    const auto &Dependencies = FunctionTypeDefinitions[I].Dependencies;
    for (unsigned J = 0; J < Dependencies.size(); J++) {
      Sorter.addEdge(I, TopologicalSortMap[Dependencies[J]]);
    }
  }
  Optional<std::vector<int>> TopologicalSortResult = Sorter.sort();
  if (!TopologicalSortResult.hasValue()) {
    errorWithMessage("Cyclic dependencies in function definitions");
  }
  for (const auto I : TopologicalSortResult.getValue()) {
    Out << FunctionTypeDefinitions[I].NameToPrint << "\n";
  }

  // We may have collected some intrinsic prototypes to emit.
  // Emit them now, before the function that uses them is emitted
  for (auto &F : prototypesToGen) {
    Out << '\n';
    printFunctionProto(Out, F);
    Out << ";\n";
  }

  Out << "\n/* Types Definitions */\n";

  for (auto it = TypedefDeclTypes.begin(), end = TypedefDeclTypes.end();
       it != end; ++it) {
    printContainedTypes(Out, *it, TypesPrinted);
  }
}

void CWriter::forwardDeclareStructs(raw_ostream &Out, Type *Ty,
                                    std::set<Type *> &TypesPrinted) {
  if (!TypesPrinted.insert(Ty).second)
    return;
  if (isEmptyType(Ty))
    return;

  for (auto I = Ty->subtype_begin(); I != Ty->subtype_end(); ++I) {
    forwardDeclareStructs(Out, *I, TypesPrinted);
  }

  if (StructType *ST = dyn_cast<StructType>(Ty)) {
    // The IO struct is in stdio.h, should not be exposed
    if (getStructName(ST) == "FILE")
      return;
    Out << getStructName(ST) << ";\n";
  } else if (auto *FT = dyn_cast<FunctionType>(Ty)) {
    // Ensure function types which are only directly used by struct types will
    // get declared.
    (void)getFunctionName(FT);
  }
}

// Push the struct onto the stack and recursively push all structs
// this one depends on.
void CWriter::printContainedTypes(raw_ostream &Out, Type *Ty,
                                  std::set<Type *> &TypesPrinted) {
  // Check to see if we have already printed this struct.
  if (!TypesPrinted.insert(Ty).second)
    return;
  // Skip empty structs
  if (isEmptyType(Ty)) {
    errs() << "YEBIN: " << Ty->getStructName() << " is Empty\n";
    return;
  }

  // Print all contained types first.
  for (Type::subtype_iterator I = Ty->subtype_begin(), E = Ty->subtype_end();
       I != E; ++I)
    printContainedTypes(Out, *I, TypesPrinted);

  if (StructType *ST = dyn_cast<StructType>(Ty)) {
    // Print structure type out.
    printStructDeclaration(Out, ST);
  } else if (ArrayType *AT = dyn_cast<ArrayType>(Ty)) {
    // Print array type out.
    printArrayDeclaration(Out, AT);
  } else if (VectorType *VT = dyn_cast<VectorType>(Ty)) {
    // Print vector type out.
    printVectorDeclaration(Out, VT);
  }
}

static inline bool isFPIntBitCast(Instruction &I) {
  if (!isa<BitCastInst>(I))
    return false;
  Type *SrcTy = I.getOperand(0)->getType();
  Type *DstTy = I.getType();
  return (SrcTy->isFloatingPointTy() && DstTy->isIntegerTy()) ||
         (DstTy->isFloatingPointTy() && SrcTy->isIntegerTy());
}

bool CWriter::isNotDuplicatedDeclaration(Instruction *I, bool isPhi) {
  // If there is no mapping between IR and variable, then there's no duplication
  // if(IR2VarName.find(I) == IR2VarName.end()) return true;
  for (auto inst2var : IRNaming) {
    auto inst = inst2var.first;
    auto var = inst2var.second;
    if (I == inst) {
      auto Vars2Emit = isPhi ? &phiVars : &allVars;
      for (auto &var2emit : *Vars2Emit) {
        if (var2emit == var) {
          Vars2Emit->erase(var2emit);
          return true;
        }
      }
      return false;
    }
  }
  return true;
}

bool CWriter::canDeclareLocalLate(Instruction &I) {
  // PHI values are merged across predecessor/sibling blocks. Declaring them
  // inside one predecessor scope can make sibling-path uses out-of-scope.
  // Keep PHI declarations at function scope.
  if (isa<PHINode>(&I))
    return false;

  // Values that feed an LCSSA/merge PHI in another block also need function
  // scope. Otherwise the region printer can emit the carrier into an inner
  // loop/branch scope and later reuse the coalesced name after that scope has
  // ended.
  auto feedsExternalPhi = [&](Instruction *Root) {
    SmallVector<Value *, 8> Worklist;
    SmallPtrSet<Value *, 16> Visited;
    Worklist.push_back(Root);

    while (!Worklist.empty()) {
      Value *V = Worklist.pop_back_val();
      if (!Visited.insert(V).second)
        continue;

      for (User *U : V->users()) {
        auto *PN = dyn_cast<PHINode>(U);
        if (!PN)
          continue;
        if (PN->getParent() != Root->getParent())
          return true;
        Worklist.push_back(PN);
      }
    }
    return false;
  };
  if (feedsExternalPhi(&I))
    return false;

  // If this instruction's chosen variable name is shared by instructions in
  // other basic blocks (from PHI/name coalescing), declaring it late inside
  // one branch can make the name out-of-scope in sibling branches.
  // This check must run BEFORE the toDeclareLocals/DeclareLocalsLate
  // early-returns, because loop instructions are bulk-added to
  // toDeclareLocals and would otherwise bypass this safety check.
  for (auto inst2var : IRNaming) {
    if (inst2var.first != &I)
      continue;
    const std::string &name = inst2var.second;
    for (auto other : IRNaming) {
      if (other.first == &I)
        continue;
      if (other.second != name)
        continue;
      if (Instruction *otherInst = dyn_cast<Instruction>(other.first))
        if (otherInst->getParent() != I.getParent())
          return false;
    }
    break;
  }

  if (toDeclareLocals.find(&I) != toDeclareLocals.end()) {
    // Loop instructions are collected here in bulk. Some of them still need a
    // wider scope (e.g. values used after the loop body/scope). Keep late
    // declaration only when the value does not escape its defining block.
    if (I.isUsedOutsideOfBlock(I.getParent()))
      return false;
    return true;
  }

  if (!DeclareLocalsLate) {
    return false;
  }

  // When a late declaration ends up inside a deeper scope than one of its uses,
  // the C compiler will reject it. That doesn't happen if we restrict to a
  // single block.
  if (I.isUsedOutsideOfBlock(I.getParent())) {
    return false;
  }

  return true;
}

void CWriter::findSignedInsts(Instruction *inst, Instruction *signedInst) {
  if (CallInst *call = dyn_cast<CallInst>(inst)) {
    if (call->getType()->isIntegerTy())
      signedInsts.insert(signedInst);
  }

  for (User *U : inst->users()) {
    if (CmpInst *cmp = dyn_cast<CmpInst>(U)) {
      switch (cmp->getPredicate()) {
      case CmpInst::ICMP_SLE:
      case CmpInst::ICMP_SGE:
      case CmpInst::ICMP_SLT:
      case CmpInst::ICMP_SGT:
        signedInsts.insert(signedInst);
        break;
      default:
        break;
      }
    } else if (SExtInst *sextInst = dyn_cast<SExtInst>(U)) {

      if (inst->hasOneUse() || (sextInst->hasOneUse() &&
                                isa<GetElementPtrInst>(*sextInst->user_back())))
        declareAsCastedType[sextInst] = inst;

      findSignedInsts(cast<Instruction>(sextInst), inst);
    } else if (GetElementPtrInst *gepInst = dyn_cast<GetElementPtrInst>(U)) {
      if (inst != dyn_cast<Instruction>(gepInst->getPointerOperand()))
        signedInsts.insert(signedInst);
    }
  }
}

static bool reachesSignedICmpUser(Value *Root) {
  if (!Root)
    return false;

  SmallVector<Value *, 32> Worklist;
  SmallPtrSet<Value *, 32> Visited;
  Worklist.push_back(Root);

  while (!Worklist.empty()) {
    Value *V = Worklist.pop_back_val();
    if (!Visited.insert(V).second)
      continue;

    for (User *U : V->users()) {
      if (auto *IC = dyn_cast<ICmpInst>(U)) {
        if (IC->isSigned())
          return true;
        continue;
      }

      Instruction *UI = dyn_cast<Instruction>(U);
      if (!UI)
        continue;

      // Propagate through integer-computing glue so PHI declarations inherit
      // signedness when the compare is on a derived value (e.g. iv.next).
      bool Forward = isa<PHINode>(UI) || isa<CastInst>(UI) ||
                     isa<SelectInst>(UI) || isa<BinaryOperator>(UI);
      if (!Forward)
        continue;
      if (!UI->getType()->isIntegerTy() && !isa<PHINode>(UI))
        continue;
      Worklist.push_back(UI);
    }
  }
  return false;
}

/// void CWriter::insertDeclaredInsts(Instruction* I){
///   // all the insts associated with this variable counts as declared
///   // Shouldn't need this code here but in case there exists empty phi
///   std::set<StringRef> declareVars;
///   for(auto inst2var : IRNaming)
///     if(inst2var.first == I)
///       declareVars.insert(inst2var.second);
///
///   for(auto var : declareVars)
///     for(auto inst2var : IRNaming)
///       if(inst2var.second == var)
///         declaredInsts.insert(inst2var.first);
///
///   if(declareVars.empty())
///     declaredInsts.insert(I);
/// }

void CWriter::DeclareLocalVariable(Instruction *I, bool &PrintedVar,
                                   bool &isDeclared,
                                   std::set<std::string> &declaredLocals) {
  if ((isInductionVariable(I) || isIVIncrement(I)) &&
      I->getType()->isIntegerTy()) {
    bool keepDecl = false;
    // Keep loop IV decl suppression for canonical for-loops, where the IV is
    // declared in the for() header. For while/do-while style loops, the PHI IV
    // is materialized via assignments (e.g. "nzv = 0; while (...)"), so it must
    // be predeclared at function scope.
    if (PHINode *PN = dyn_cast<PHINode>(I))
      if (Loop *L = LI->getLoopFor(PN->getParent()))
        keepDecl = (getLoopType(L) != forLoop);
    if (!keepDecl)
      return;
  }

  if (AllocaInst *AI = isDirectAlloca(I)) {
    auto varName = GetValueName(AI);
    if (declaredLocals.find(varName) != declaredLocals.end())
      return;
    varName = GetValueName(AI, true);
    declaredLocals.insert(varName);
    errs() << "SUSAN: declaring varName 5264: " << varName << "\n";

    Out << "  ";

    bool printedType = false;
    for (auto [sextInst, inst] : declareAsCastedType)
      if (inst == I) {
        errs() << "SUSAN: printing type at 5874: " << *(sextInst->getType())
               << "\n";
        printTypeNameForAddressableValue(Out, sextInst->getType(), true);
        printedType = true;
        break;
      }

    auto type2print = AI->getAllocatedType();
    // if(allocaTypeChange.find(AI) != allocaTypeChange.end())
    //   type2print = allocaTypeChange[AI];

    if (!printedType) {
      errs() << "SUSAN: printing type at 5885: " << *(type2print) << "\n";
      if (signedInsts.find(I) != signedInsts.end())
        printTypeNameForAddressableValue(Out, type2print, true);
      else
        printTypeNameForAddressableValue(Out, type2print, false);
    }

    Out << ' ' << varName;

    ArrayType *ArrTy = dyn_cast<ArrayType>(AI->getAllocatedType());
    while (ArrTy) {
      Out << "[" << ArrTy->getNumElements() << "]";
      ArrTy = dyn_cast<ArrayType>(ArrTy->getElementType());
    }

    Out << ";    /* Address-exposed local */\n";
    PrintedVar = true;
    isDeclared = true;
    nameDict.LocalVars[demangleFunctionName(I->getFunction()->getName())].insert(varName);
  } else if (PHINode *PN = dyn_cast<PHINode>(I)) {
    if (isEmptyType(PN->getType()) || PN->user_empty()) {
      errs() << "CBE_DEBUG: skipping PHI predecl "
             << PN->getFunction()->getName() << "::" << PN->getParent()->getName()
             << " phi=" << *PN << " reason="
             << (isEmptyType(PN->getType()) ? "empty-type" : "no-users") << "\n";
      return;
    }
    auto varName = GetValueName(PN);
    if (declaredLocals.find(varName) != declaredLocals.end() ||
        omp_declaredLocals.find(varName) != omp_declaredLocals.end()) {
      errs() << "CBE_DEBUG: PHI predecl already exists "
             << PN->getFunction()->getName() << "::" << PN->getParent()->getName()
             << " var=" << varName << "\n";
      return;
    }

    errs() << "CBE_DEBUG: PHI predecl emit "
           << PN->getFunction()->getName() << "::" << PN->getParent()->getName()
           << " var=" << varName << " phi=" << *PN << "\n";
    Out << "  ";
    // Do not force PHI predecls to unsigned. If a PHI participates in signed
    // comparisons, printing it as unsigned changes control-flow semantics
    // (e.g. kk >= k in sparse()) and can lead to out-of-bounds accesses.
    // Keep this targeted to signed-compare uses instead of all loop-carried
    // integers, because some recurrence variables are intentionally unsigned
    // (e.g. lg in ilog2/ilog2_device).
    bool forceSignedPhi =
        PN->getType()->isIntegerTy() && reachesSignedICmpUser(PN);
    bool useSigned = forceSignedPhi || (signedInsts.find(PN) != signedInsts.end());
    printTypeName(Out, PN->getType(), useSigned) << ' ' << varName << ";\n";
    declaredLocals.insert(varName);
    nameDict.LocalVars[demangleFunctionName(PN->getFunction()->getName())]
        .insert(varName);
    PrintedVar = true;
    isDeclared = true;
    return;
  } else if (!isEmptyType(I->getType()) && !isInlinableInst(*I)) {
    // Let emitOpenMPAtomicForInst own declaration/type for atomic lowering.
    if (isAtomicAddLoweringCandidate(I))
      return;

    errs() << "YEBIN: WE ARE HERE FOR " << *I << "\n";

    ///*
    // * OpenMP: skip some declarations related to OpenMP calls
    // */
    // if(CallInst* CI = dyn_cast<CallInst>(&*I))
    //  if(Function *ompCall = CI->getCalledFunction())
    //    if(ompCall->getName().contains("__kmpc_master")
    //        || ompCall->getName().contains("__kmpc_end_master"))
    //      return;
    ///*
    // * OpenMP end
    // */

    auto varName = GetValueName(I);
    errs() << "SUSAN: declaring varName 5298: " << varName << "\n";
    if (declaredLocals.find(varName) != declaredLocals.end())
      return;
    errs() << "SUSAN: declared locals:\n";
    for (auto local : declaredLocals)
      errs() << local << "\n";
    bool shouldDeclare =
        !canDeclareLocalLate(*I) && !(&*I)->user_empty() &&
        (isa<PHINode>(I) || isNotDuplicatedDeclaration(I, false));
    if (shouldDeclare) {
      if (declaredLocals.find(varName) != declaredLocals.end())
        return;
      declaredLocals.insert(varName);

      errs() << "SUSAN: inst at 5950: " << *I << "\n";
      errs() << "SUSAN: declaring " << *I << "\n";
      Out << "  ";

      bool printedType = false;
      for (auto [sextInst, inst] : declareAsCastedType)
        if (inst == I) {
          errs() << "SUSAN: printing type at 5930: " << *(sextInst->getType())
                 << "\n";
          printTypeName(Out, sextInst->getType(), true) << ' ' << varName;
          printedType = true;
          break;
        }

      if (!printedType) {
        errs() << "SUSAN: printing type at 5937: " << *(I->getType()) << "\n";
        if (signedInsts.find(I) != signedInsts.end())
          printTypeName(Out, I->getType(), true) << ' ' << varName;
        else
          printTypeName(Out, I->getType(), false) << ' ' << varName;
      }

      Out << ";\n";

      // insertDeclaredInsts(I);
      nameDict.LocalVars[demangleFunctionName(I->getFunction()->getName())]
          .insert(varName);
    }

    PrintedVar = true;
    isDeclared = true;
  }
  // We need a temporary for the BitCast to use so it can pluck a value out
  // of a union to do the BitCast. This is separate from the need for a
  // variable to hold the result of the BitCast.
  if (isFPIntBitCast(*I)) {
    headerUseBitCastUnion();
    Out << "  llvmBitCastUnion " << GetValueName(I) << "__BITCAST_TEMPORARY;\n";
    PrintedVar = true;
  }
}

void CWriter::printFunction(Function &F, bool inlineF) {
  argNameOverrides.clear();

  // SUSAN: collect function argument reference depths
  for (auto arg = F.arg_begin(); arg != F.arg_end(); ++arg) {
    Type *argTy = arg->getType();
    if (isa<ArrayType>(argTy) || isa<PointerType>(argTy) ||
        isa<StructType>(argTy)) {
      findVariableDepth(argTy, cast<Value>(arg), 0);
    }
  }

  /// isStructReturn - Should this function actually return a struct by-value?
  bool isStructReturn = F.hasStructRetAttr();

  if (!inlineF) {
    cwriter_assert(!F.isDeclaration());
    if (F.hasDLLImportStorageClass())
      Out << "__declspec(dllimport) ";
    if (F.hasDLLExportStorageClass())
      Out << "__declspec(dllexport) ";
    // if (F.hasLocalLinkage())
    //   Out << "static ";
  }

  std::string Name = GetValueName(&F);

  FunctionType *FTy = F.getFunctionType();

  if (Name == "main") {
    if (!isStandardMain(FTy)) {
      // Implementations are free to support non-standard signatures for main(),
      // so it would be unreasonable to make it an outright error.
      errs() << "CBackend warning: main() has an unrecognized signature. The "
                "types emitted will not be fixed to match the C standard.\n";
    }
  }

  iterator_range<Function::arg_iterator> args = F.args();

  // SUSAN: build a variable - IR table
  std::map<Value *, std::set<std::string>> IR2vars;
  std::map<std::string, std::set<Value *>> Var2IRs;
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    if (CallInst *CI = dyn_cast<CallInst>(&*I)) {
      if (Function *F = CI->getCalledFunction()) {
        if (F->getIntrinsicID() == Intrinsic::dbg_value ||
            F->getIntrinsicID() == Intrinsic::dbg_declare) {
          errs() << *CI << "\n";
          errs() << *CI->getParent() << "\n";
          Metadata *valMeta =
              cast<MetadataAsValue>(CI->getOperand(0))->getMetadata();
          Metadata *varMeta =
              cast<MetadataAsValue>(CI->getOperand(1))->getMetadata();
          DIExpression *expr = dyn_cast<DIExpression>(
              cast<MetadataAsValue>(CI->getOperand(2))->getMetadata());
          if (expr && expr->getNumElements() > 0)
            continue;
          DILocalVariable *var = dyn_cast<DILocalVariable>(varMeta);
          assert(var && "SUSAN: 2nd argument of llvm.dbg.value is not "
                        "DILocalVariable?\n");
          std::string varName = var->getName().str();
          if (isa<ValueAsMetadata>(valMeta)) {
            Value *valV = cast<ValueAsMetadata>(valMeta)->getValue();
            // if(AllocaInst *alloca = dyn_cast<AllocaInst>(valV))
            //   noneSkipAllocaInsts.insert(alloca);
            if (Argument *arg = dyn_cast<Argument>(valV)) {
              if (isShortIVStyleName(varName))
                continue;
              if (isCudaBuiltinStyleArgName(arg->getName()))
                continue;
              bool conflictsWithCurrentFunctionIVName = false;
              for (auto &Entry : IV2Name) {
                if (Entry.second != varName)
                  continue;
                if (Instruction *Inst = dyn_cast<Instruction>(Entry.first)) {
                  if (Inst->getFunction() == arg->getParent()) {
                    conflictsWithCurrentFunctionIVName = true;
                    break;
                  }
                } else if (Argument *OtherArg = dyn_cast<Argument>(Entry.first)) {
                  if (OtherArg->getParent() == arg->getParent()) {
                    conflictsWithCurrentFunctionIVName = true;
                    break;
                  }
                }
              }
              if (conflictsWithCurrentFunctionIVName)
                continue;
              errs() << "SUSAN: found argument 6346: " << *valV << "\n";
              if (Var2IRs.find(varName) == Var2IRs.end())
                Var2IRs[varName] = std::set<Value *>();
              Var2IRs[varName].insert(arg);
              errs() << "CBackend: varname: " << varName << "\n";
              errs() << *CI << "\n";
              allVars.insert(varName);

              // build IR -> Vars table
              if (IR2vars.find(arg) == IR2vars.end())
                IR2vars[arg] = std::set<std::string>();
              IR2vars[arg].insert(varName);

              // try: build just IRNaming
              IRNaming.insert(std::make_pair(arg, varName));
            } else if (Instruction *valInst = dyn_cast<Instruction>(valV)) {
              if (isa<TruncInst>(valInst) || isa<BitCastInst>(valInst))
                valInst = dyn_cast<Instruction>(valInst->getOperand(0));
              if (!valInst)
                continue;

              // Avoid assigning the same debug name to multiple canonical
              // for-loop PHI IVs (nested loops often reuse source names like
              // "n"), which can later produce self-initialized headers:
              //   for (n = n; ...)
              if (PHINode *phiVal = dyn_cast<PHINode>(valInst)) {
                bool isMainIV = isInductionVariable(phiVal);
                bool isExtraIV = isExtraInductionVariable(phiVal);
                if ((isMainIV || isExtraIV) && phiVal->getType()->isIntegerTy()) {
                  if (Loop *phiLoop = LI->getLoopFor(phiVal->getParent())) {
                    if (LoopProfile *phiLP = findLoopProfile(phiLoop)) {
                      if (phiLP->isForLoop) {
                        // Keep lower-bound carrier PHIs (extra IVs) on stable
                        // synthetic names; binding them to source names like
                        // "n" is order-dependent and can recreate `for (n = n; ...)`.
                        if (isExtraIV && !isMainIV)
                          continue;

                        auto existing = Var2IRs.find(varName);
                        auto phiUsesPhi = [](PHINode *A, PHINode *B) -> bool {
                          for (unsigned i = 0; i < A->getNumIncomingValues(); ++i)
                            if (A->getIncomingValue(i) == B)
                              return true;
                          return false;
                        };
                        bool hasRelatedPhiNameConflict = false;
                        if (existing != Var2IRs.end()) {
                          for (Value *V : existing->second) {
                            PHINode *otherPhi = dyn_cast<PHINode>(V);
                            if (!otherPhi || otherPhi == phiVal)
                              continue;

                            // Only block conflicts that can create problematic
                            // loop-header coupling (e.g. main-IV <-> extra-IV),
                            // not unrelated PHIs that legitimately reuse names
                            // in separate loop regions/functions.
                            bool sameHeader = (otherPhi->getParent() == phiVal->getParent());
                            bool directPhiDependency =
                                phiUsesPhi(phiVal, otherPhi) || phiUsesPhi(otherPhi, phiVal);
                            if (sameHeader || directPhiDependency) {
                              hasRelatedPhiNameConflict = true;
                              break;
                            }
                          }
                        }
                        if (hasRelatedPhiNameConflict)
                          continue;
                      }
                    }
                  }
                }
              }

              if (Var2IRs.find(varName) == Var2IRs.end())
                Var2IRs[varName] = std::set<Value *>();
              Var2IRs[varName].insert(valInst);

              allVars.insert(varName);
              if (isa<PHINode>(valInst)) {
                phiVars.insert(varName);
              }

              // build IR -> Vars table
              if (IR2vars.find(valInst) == IR2vars.end())
                IR2vars[valInst] = std::set<std::string>();
              IR2vars[valInst].insert(varName);

              // try: build just IRNaming
              IRNaming.insert(std::make_pair(valInst, varName));
            }
          } else {
            errs() << "In function " << I->getFunction()->getName() << "\n";
            continue;
            //assert(0 && "SUSAN: 1st argument is not a Value?\n");
          }
        }
      }
    }
  }
  // ANDREW: Propagate variable names to PHI nodes (specifically LCSSA)
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    if (PHINode *phi = dyn_cast<PHINode>(&*I)) {
      // Check if PHI already has a name
      if (IR2vars.find(phi) != IR2vars.end() && !IR2vars[phi].empty())
        continue;

      // Keep canonical for-loop IV/extra-IV PHIs out of late name propagation.
      // Propagating debug names here can collapse IV and lower-bound carrier
      // names and produce self-initialized headers like "for (n = n; ...)".
      if ((isInductionVariable(phi) || isExtraInductionVariable(phi)) &&
          phi->getType()->isIntegerTy()) {
        if (Loop *phiLoop = LI->getLoopFor(phi->getParent())) {
          if (LoopProfile *phiLP = findLoopProfile(phiLoop)) {
            if (phiLP->isForLoop)
              continue;
          }
        }
      }

      std::string candidate = "";
      bool consistent = true;
      bool distinct = false;

      for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i) {
        Value *inc = phi->getIncomingValue(i);
        Value *phiOp = inc;
        if (isa<TruncInst>(phiOp) || isa<BitCastInst>(phiOp))
            phiOp = cast<Instruction>(phiOp)->getOperand(0);

        if (IR2vars.find(phiOp) != IR2vars.end() && !IR2vars[phiOp].empty()) {
          std::string incName = *IR2vars[phiOp].begin(); // Pick first name
          if (!distinct) {
            candidate = incName;
            distinct = true;
          } else if (candidate != incName) {
            consistent = false;
            break;
          }
        }
      }

      if (distinct && consistent) {
        // Avoid collapsing multiple PHI nodes to the same debug name.
        // This can produce self-initializing loop headers such as:
        //   for (int64_t n = n; ...)
        bool conflictsWithOtherPhi = false;
        auto existing = Var2IRs.find(candidate);
        if (existing != Var2IRs.end()) {
          for (auto *V : existing->second) {
            if (V != phi && isa<PHINode>(V)) {
              conflictsWithOtherPhi = true;
              break;
            }
          }
        }
        if (conflictsWithOtherPhi) {
          errs() << "SUSAN: Skipping PHI name propagation due to PHI name conflict: "
                 << candidate << " for " << *phi << "\n";
          continue;
        }

        errs() << "SUSAN: Propagating " << candidate << " to PHI " << *phi
               << "\n";
        IRNaming.insert(std::make_pair(phi, candidate));
        IR2vars[phi].insert(candidate);
        Var2IRs[candidate].insert(phi);
        allVars.insert(candidate);
        phiVars.insert(candidate);
      }
    }
  }

  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    if (PHINode *phi = dyn_cast<PHINode>(&*I)) {
      auto name = GetValueName(phi);
      auto phiNames = IR2vars.find(phi);
      if (phiNames != IR2vars.end() && !phiNames->second.empty())
        name = *phiNames->second.begin();

      errs() << "SUSAN: phi related name: " << name << "\n";
      for (unsigned i = 0; i < phi->getNumIncomingValues(); ++i)
        if (Instruction *incomingInst =
                dyn_cast<Instruction>(phi->getIncomingValue(i))) {
          // Do not let a PHI in one loop rename a PHI from a different loop.
          // This preserves distinct loop-carried state names across nesting
          // boundaries (e.g. outer iv37 vs inner iv39) and avoids collapsing
          // required PHI edge copies into apparent self-assignments.
          if (auto *incomingPhi = dyn_cast<PHINode>(incomingInst)) {
            Loop *phiLoop = LI->getLoopFor(phi->getParent());
            Loop *incomingLoop = LI->getLoopFor(incomingPhi->getParent());
            if (phiLoop && incomingLoop && phiLoop != incomingLoop)
              continue;
          }

          auto isCanonicalForLoopIntIVLike = [&](Instruction *I) -> bool {
            PHINode *PN = dyn_cast<PHINode>(I);
            if (!PN || !PN->getType()->isIntegerTy())
              return false;
            if (!(isInductionVariable(PN) || isExtraInductionVariable(PN)))
              return false;
            Loop *L = LI->getLoopFor(PN->getParent());
            if (!L)
              return false;
            LoopProfile *LP = findLoopProfile(L);
            return LP && LP->isForLoop;
          };

          // Do not let non-IV PHI names overwrite loop IV names.
          // Example failure mode: plane IV gets renamed to juppermax
          // because cond57 includes %plane.0 as one incoming edge.
          if (isInductionVariable(incomingInst) || isIVIncrement(incomingInst) ||
              isExtraInductionVariable(incomingInst) ||
              isExtraIVIncrement(incomingInst)) {
            if (!(isInductionVariable(phi) || isExtraInductionVariable(phi)))
              continue;
          }

          // Keep canonical for-loop IV and extra-IV PHIs independently named.
          // If the main IV's source name is propagated to the incoming extra-IV
          // carrier, we can recreate headers like: for (n = n; ...).
          if (isCanonicalForLoopIntIVLike(phi) &&
              isCanonicalForLoopIntIVLike(incomingInst))
            continue;

          IRNaming.insert(std::make_pair(incomingInst, name));
          IR2vars[incomingInst].insert(name);
          Var2IRs[name].insert(incomingInst);
        }
    }
  }

  errs() << "=========================" << F.getName() << ": IR NAMING "
            "BEFORE=====================\n";
  for (auto inst2var : IRNaming) {
    errs() << *inst2var.first << " -> " << inst2var.second << "\n";
  }

  std::map<Instruction *, std::map<std::string, Instruction *>> MRVar2ValMap;
  std::set<std::string> vars2record;
  for (auto &var2vals : Var2IRs)
    vars2record.insert(var2vals.first);

  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    Instruction *currInst = &*I;
    std::map<std::string, Instruction *> var2val;
    for (auto &var2record : vars2record)
      var2val[var2record] = nullptr;
    MRVar2ValMap[currInst] = var2val;
  }

  // build the MRVar2ValMap
  std::map<Instruction *, std::map<std::string, Instruction *>>
      prevMRVar2ValMap, currMRVar2ValMap;
  do {
    prevMRVar2ValMap = MRVar2ValMap;
    for (auto &BB : F) {
      std::map<std::string, Instruction *> prev_var2val;
      for (auto &I : BB) {
        Instruction *currInst = &I;
        auto curr_var2val = MRVar2ValMap[currInst];

        const std::set<std::string> *vars2gen = nullptr;
        auto currNames = IR2vars.find(currInst);
        if (currNames != IR2vars.end())
          vars2gen = &currNames->second;

        std::map<std::string, Instruction *> merged_var2val;
        std::map<std::string, Instruction *> prev_pred_var2val;
        if (prev_var2val.empty()) {
          for (pred_iterator PI = pred_begin(&BB), E = pred_end(&BB); PI != E;
               ++PI) {
            BasicBlock *pred = *PI;
            Instruction *term = pred->getTerminator();
            std::map<std::string, Instruction *> term_var2val =
                MRVar2ValMap[term];
            for (auto &[var, val] : term_var2val)
              if (val)
                curr_var2val[var] = val;

            if (!prev_pred_var2val.empty()) {
              for (auto &[var, val] : term_var2val)
                if (prev_pred_var2val[var] != val && val &&
                    prev_pred_var2val[var])
                  curr_var2val[var] = nullptr;
            }
            prev_pred_var2val = term_var2val;
          }
        } else {
          for (auto &var : vars2record)
            curr_var2val[var] = prev_var2val[var];
        }

        for (auto &var : vars2record)
          if (vars2gen && vars2gen->find(var) != vars2gen->end())
            curr_var2val[var] = currInst;

        MRVar2ValMap[currInst] = curr_var2val;
        prev_var2val = curr_var2val;
      }
    }
    currMRVar2ValMap = MRVar2ValMap;
  } while (!changeMapValue(prevMRVar2ValMap, currMRVar2ValMap, F));

  // test the table
  for (auto &[inst, var2val] : MRVar2ValMap) {
    errs() << "SUSAN: inst:" << *inst << "\n";
    for (auto &[var, val] : var2val) {
      if (val)
        errs() << var << ":" << *val << "\n";
    }
  }

  // find the contradicting cases and delete them in Var2IRs table
  // Contradicting: if at any instruction I1, it uses I2, but I2 has two
  // coressponding variable v1 and v2, however only v2 has value I2 at this
  // point, then the v1 -> I2 mapping needs to be deleted
  std::vector<std::pair<Value *, std::string>> instVarPair2Delete;
  for (auto &[inst, var2val] : MRVar2ValMap) {
    if (isa<PHINode>(inst))
      continue;
    for (unsigned i = 0, e = inst->getNumOperands(); i != e; ++i)
      if (Instruction *operand = dyn_cast<Instruction>(inst->getOperand(i))) {
        // Keep canonical/extra IV names stable through the contradiction pass.
        // If we erase/rebind these names, for-loop headers can end up mixing
        // fallback IV names (e.g. j) with source names (e.g. plane).
        if (isInductionVariable(operand) || isIVIncrement(operand) ||
            isExtraInductionVariable(operand) || isExtraIVIncrement(operand))
          continue;
        for (auto &[var, valAtOperand] : var2val)
          if (operand == valAtOperand) {
            std::set<std::string> operandVars;
            auto existingVars = IR2vars.find(operand);
            if (existingVars != IR2vars.end())
              operandVars = existingVars->second;

            for (const auto &var2erase : operandVars) {
              if (var2erase != var) {
                Var2IRs[var2erase].erase(operand);
                errs() << "SUSAN: at inst " << *inst << "\n";
                errs() << "SUSAN: removinginst2var at 6152: "
                       << *operand << " -> " << var2erase << "\n";
                instVarPair2Delete.push_back(std::make_pair(operand, var2erase));
              }
            }

            IR2vars[operand].clear();
            IR2vars[operand].insert(var);
            IRNaming.insert(std::make_pair(operand, var));
            Var2IRs[var].insert(operand);
          }
      }
  }

  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    Instruction *inst = &*I;
    auto MRVar2Vals = MRVar2ValMap[inst];
    if (MRVar2Vals.empty())
      continue;

    for (unsigned i = 0, e = inst->getNumOperands(); i != e; ++i)
      if (Instruction *operand = dyn_cast<Instruction>(inst->getOperand(i))) {
        auto operandNames = IR2vars.find(operand);
        if (operandNames == IR2vars.end())
          continue;

        for (const auto &var : operandNames->second) {
          if (MRVar2Vals.find(var) != MRVar2Vals.end() &&
              operand != MRVar2Vals[var] && MRVar2Vals[var] != inst) {
            // Note: if it's IV, we know how to handle it and doesn't need to
            // be deleted Note: if one of them is alloca, it should be fine
            if (operand && MRVar2Vals[var] &&
                (isa<AllocaInst>(operand) ||
                 isa<AllocaInst>(MRVar2Vals[var]))) {
              // if(isa<AllocaInst>(operand) && MRVar2Vals[var])
              //   allocaTypeChange[operand] = MRVar2Vals[var]->getType();
              // else if(isa<AllocaInst>(MRVar2Vals[var]) && operand)
              //   allocaTypeChange[MRVar2Vals[var]] = operand->getType();

              errs() << "SUSAN: inst at 6227: " << *inst << "\n";
              // declaredLocals.insert(GetValueName(operand));
              continue;
            }
            // if(!isInductionVariable(operand) && !isIVIncrement(operand)){
            //   for(auto pair : IRNaming)
            //     //if(pair.first == operand && pair.second == var){
            //     //  errs() << "SUSAN: removinginst2var at 6180: " <<
            //     *operand << " -> " << var << "\n";
            //       instVarPair2Delete.push_back(pair);
            //     }
            // }

            Instruction *shadowingInst = MRVar2Vals[var];
            if (!isInductionVariable(shadowingInst) &&
                !isIVIncrement(shadowingInst)) {
              auto shadowingNames = IR2vars.find(shadowingInst);
              if (shadowingNames != IR2vars.end() &&
                  shadowingNames->second.find(var) !=
                      shadowingNames->second.end()) {
                errs() << "SUSAN: removinginst2var at 6188: " << *operand
                       << " -> " << var << "\n";
                instVarPair2Delete.push_back(
                    std::make_pair(shadowingInst, var));
              }
            }
          }
        }
      }
  }

  for (auto deletePair : instVarPair2Delete) {
    IR2vars[deletePair.first].erase(deletePair.second);
    Var2IRs[deletePair.second].erase(deletePair.first);
    IRNaming.erase(deletePair);
  }

  // Keep declaration bookkeeping in sync with final IRNaming.
  // Some names (e.g. PHI-incoming values coalesced to a PHI-related name)
  // are introduced via IRNaming propagation passes without updating allVars.
  // If allVars misses those names, predeclaration can be skipped and the first
  // assignment site ends up emitting a branch-local declaration, which breaks
  // sibling-branch uses.
  for (auto inst2var : IRNaming) {
    allVars.insert(inst2var.second);
    if (isa<PHINode>(inst2var.first))
      phiVars.insert(inst2var.second);
  }

  errs() << "=========================" << F.getName() << ": IR NAMING=====================\n";
  for (auto inst2var : IRNaming) {
    errs() << *inst2var.first << " -> " << inst2var.second << "\n";
  }

  static size_t Function_Order_ID = 0;
  /*
   * OpenMP: remove first two args from outline
   */
  if (!inlineF) {
    if (F.getName() != "main") {
      Out << "// FUNCTION ORDER ID " << Function_Order_ID << " START\n";
      Out << "// INSERT COMMENT FUNCTION: " << demangleFunctionName(F.getName())
          << "\n";
    } else 
      Out << "// MAIN START\n";
    if (IS_OPENMP_FUNCTION)
      printFunctionProto(Out, FTy,
                         std::make_pair(F.getAttributes(), F.getCallingConv()),
                         Name, &args, 2);
    else
      printFunctionProto(Out, FTy,
                         std::make_pair(F.getAttributes(), F.getCallingConv()),
                         Name, &args);

    Out << " {\n";
  }

  // If this is a struct return function, handle the result with magic.
  if (isStructReturn) {
    Type *StructTy =
        cast<PointerType>(F.arg_begin()->getType())->getElementType();
    Out << "  ";
    printTypeName(Out, StructTy, false)
        << " StructReturn;  /* Struct return temporary */\n";

    Out << "  ";
    printTypeName(Out, F.arg_begin()->getType(), false);
    Out << GetValueName(F.arg_begin()) << " = &StructReturn;\n";
  }

  // Function arguments are already declared in the prototype; mark them so
  // reconstructed debug names do not get emitted as duplicate locals.
  int skipArgSteps = IS_OPENMP_FUNCTION ? 2 : 0;
  bool skipStructRetArg = isStructReturn;
  for (auto arg = F.arg_begin(); arg != F.arg_end(); ++arg) {
    if (skipStructRetArg) {
      skipStructRetArg = false;
      continue;
    }
    if (skipArgSteps > 0) {
      --skipArgSteps;
      continue;
    }
    std::string argName = GetValueName(arg);
    declaredLocals.insert(argName);
    omp_declaredLocals.insert(argName);
  }

  bool PrintedVar = false;

  /*
   * Naturalness: avoid cast by checking the use of the variable before it's
   * declared
   */
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    Instruction *inst = &*I;
    findSignedInsts(inst, inst);
  }
  // if signedInsts have corresponding variable, then that variable is signed
  std::set<std::string> signedVars;
  for (auto signedInst : signedInsts) {
    auto signedNames = IR2vars.find(signedInst);
    if (signedNames != IR2vars.end())
      for (const auto &Name : signedNames->second)
        signedVars.insert(Name);
  }

  for (auto signedVar : signedVars) {
    auto valuesForName = Var2IRs.find(signedVar);
    if (valuesForName != Var2IRs.end())
      for (auto *V : valuesForName->second)
        if (Instruction *inst = dyn_cast<Instruction>(V))
          signedInsts.insert(inst);
  }

  // print local variable information for the function
  bool isDeclared = false;
  if (!IS_OPENMP_FUNCTION) {
    for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
      if (!canDeclareLocalLate(*I)) {
        DeclareLocalVariable(&*I, PrintedVar, isDeclared, declaredLocals);
      }
    }
  }

  // YEBIN: run analysis for kernels after var renaming
  // FIXME: is this the right place?
  if (F.getMetadata("tulip.cuda.kernel.caller")) {
    runAnalysisOnKernelCaller(F);

    Out << "unsigned threadsPerBlock = 256;\n";
    // FIXME: add checks for dims and add prints
    for (auto call : KernelCallDims) {
      Out << "dim3 block" << call.second.first << "("
          << call.second.second.first << ", 1, 1);\n";
      Out << "dim3 grid" << call.second.first << "((";
      writeOperand(call.second.second.second);
      Out << "+block" << call.second.first << ".x-1)/block" << call.second.first
          << ".x, 1, 1);\n\n";
    }
  }

  if (PrintedVar)
    Out << '\n';

  /*
   * OpenMP:
   * Record Liveins
   * Only prints the loop
   */
  errs() << "SUSAN: am I here 6807??\n";
  TopRegion->printRegionDAG();
  //    std::queue<BasicBlock*> toVisit;
  //    std::set<BasicBlock*> visited;
  //    toVisit.push(&F.getEntryBlock());
  //    visited.insert(&F.getEntryBlock());
  //    errs() << "SUSAN: adding entry block: " << F.getEntryBlock() << "\n";
  //    while(!toVisit.empty()){
  //	    BasicBlock *currBB = toVisit.front();
  //	    toVisit.pop();
  //      if (Loop *L = LI->getLoopFor(currBB)) {
  //        if (L->getHeader() == currBB
  //            && L->getParentLoop() == nullptr
  //            && times2bePrinted[currBB]) {
  //          errs() << "SUSAN: printing loop " << currBB->getName() << " at
  //          5538\n"; if(NATURAL_CONTROL_FLOW)
  //            printLoopNew(L);
  //          else
  //            printLoop(L);
  //        }
  //      } else {
  //        errs() << "SUSAN: printing bb:" << currBB->getName() << "\n";
  //        printBasicBlock(currBB);
  //        times2bePrinted[currBB]--;
  //      }
  //
  //      CBERegion *R = findRegionOfBlock(currBB);
  //      if(BranchInst *br = dyn_cast<BranchInst>(currBB->getTerminator())){
  //          errs() << "SUSAN: br:" << *br << "\n";
  //          BasicBlock *succ0 = br->getSuccessor(0);
  //          BasicBlock *succOne = nullptr;
  //          if(br->isConditional()){
  //            succOne = br->getSuccessor(1);
  //          }
  //          if(succOne){
  //            Loop* L = LI->getLoopFor(succOne);
  //            if(!(L && L->getLoopLatch() == succOne))
  //              if(R && !nodeBelongsToRegion(succ0, R)) continue;
  //          }
  //          if(!succOne && R && !nodeBelongsToRegion(succ0, R)) continue;
  //          errs() << "print succ0 :" << *succ0 << "\n";
  //		      if(visited.find(succ0)==visited.end()){
  //            toVisit.push(succ0);
  //            visited.insert(succ0);
  //          }
  //
  //          if(!br->isConditional()) continue;
  //          BasicBlock *succ1 = br->getSuccessor(1);
  //          if(R && !nodeBelongsToRegion(succ1, R, true)) continue;
  //		      if(visited.find(succ1)==visited.end()){
  //            toVisit.push(succ1);
  //            visited.insert(succ1);
  //          }
  //      } else {
  //	      for (auto succ = succ_begin(currBB); succ != succ_end(currBB);
  //++succ){ 		      BasicBlock *succBB = *succ;
  //          if(R && !nodeBelongsToRegion(succBB, R)) continue;
  //		      if(visited.find(succBB)==visited.end()){
  //            toVisit.push(succBB);
  //            visited.insert(succBB);
  //          }
  //        }
  //      }
  //    }
  //

  if (!inlineF) {
    Out << "}\n";
    if (F.getName() != "main") {
      Out << "// FUNCTION ORDER ID " << Function_Order_ID << " END\n\n";
      ++Function_Order_ID;
    } else 
    Out << "// MAIN END\n\n";
  }
}

void CWriter::printCmpOperator(ICmpInst *icmp, bool negateCondition) {
  switch (icmp->getPredicate()) {
  case ICmpInst::ICMP_EQ:
    if (negateCondition)
      Out << "!=";
    else
      Out << " == ";
    break;
  case ICmpInst::ICMP_NE:
    if (negateCondition)
      Out << "==";
    else
      Out << " != ";
    break;
  case ICmpInst::ICMP_ULE:
  case ICmpInst::ICMP_SLE:
    if (negateCondition)
      Out << ">";
    else
      Out << " <= ";
    break;
  case ICmpInst::ICMP_UGE:
  case ICmpInst::ICMP_SGE:
    if (negateCondition)
      Out << "<";
    else
      Out << " >= ";
    break;
  case ICmpInst::ICMP_ULT:
  case ICmpInst::ICMP_SLT:
    if (negateCondition)
      Out << ">=";
    else
      Out << " < ";
    break;
  case ICmpInst::ICMP_UGT:
  case ICmpInst::ICMP_SGT:
    if (negateCondition)
      Out << "<=";
    else
      Out << " > ";
    break;
  default:
    DBG_ERRS("Invalid icmp predicate!" << *icmp);
    errorWithMessage("invalid icmp predicate");
  }
}

void CWriter::printInstruction(Instruction *I, bool printSemiColon) {
  if (I->getMetadata("tulip.target.end.of.map"))
    return;
  if (I->getMetadata("tulip.target.start.of.map"))
    return;
  if (CallInst *CI = dyn_cast<CallInst>(I)) {
    if (CI->getCalledFunction()->getName() == "cudaMemcpy") {
      return;
    }
  }
  if (omp_SkipVals.find(I) != omp_SkipVals.end())
    return;
  if (deadInsts.find(I) != deadInsts.end())
    return;
  Out << "  ";
  if (!(&*I)->user_empty() && !isEmptyType(I->getType()) && !isInlineAsm(*I)) {
    auto varName = GetValueName(&*I, true);
    if (canDeclareLocalLate(*I) && !isIVIncrement(I)) {
      errs() << "SUSAN: printing type name for " << varName << " at 6805\n";
      printTypeName(Out, I->getType(), false) << ' ';
      declaredLocals.insert(varName);
    }
    Out << GetValueName(&*I) << " = ";
  }
  writeInstComputationInline(*I);

  if (printSemiColon)
    Out << ";\n";
}
void CWriter::keepIVUnrelatedInsts(
    BasicBlock *skipBB, Instruction *condInst,
    std::set<Instruction *> &InstsKeptFromSkipBlock) {
  for (auto &I : *skipBB) {
    if (isSkipableInst(&I))
      continue;
    if (isa<BranchInst>(&I) || isIVIncrement(cast<Value>(&I)) || &I == condInst)
      continue;
    bool skipIVRelated = false;
    for (User *U : I.users())
      if (isIVIncrement(U) || isa<BranchInst>(U) || U == condInst) {
        skipIVRelated = true;
        break;
      }
    if (skipIVRelated)
      continue;
    InstsKeptFromSkipBlock.insert(&I);
  }
}

BasicBlock *findDoWhileExitingLatchBlock(Loop *L) {
  /*SmallVector< BasicBlock*, 1> ExitingBlocks;
  SmallVector< BasicBlock*, 1> ExitBlocks;
  L->getExitingBlocks(ExitingBlocks);
  L->getExitBlocks(ExitBlocks);

  for(SmallVector<BasicBlock*,1>::iterator i=ExitingBlocks.begin(),
  e=ExitingBlocks.end(); i!=e; ++i){ BasicBlock *exit = *i;
    if(L->isLoopLatch(exit))
     return exit;
  }*/

  // Assuming loops are all rotated
  BasicBlock *latch = L->getLoopLatch();
  errs() << "SUSAN: latch " << *latch << "\n";
  BranchInst *br = dyn_cast<BranchInst>(latch->getTerminator());
  assert(br && "latch doesn't end with branch inst??\n");
  if (!br->isConditional())
    return latch->getSinglePredecessor();
  return latch;
  // return nullptr;
}

Instruction *CWriter::findCondInst(Loop *L, bool &negateCondition) {
  negateCondition = false;
  auto pickCondFromBranch = [&](BranchInst *Br) -> Instruction * {
    if (!Br || !Br->isConditional())
      return nullptr;

    BasicBlock *Succ0 = Br->getSuccessor(0);
    BasicBlock *Succ1 = Br->getSuccessor(1);
    bool Succ0InLoop = L->contains(Succ0);
    bool Succ1InLoop = L->contains(Succ1);

    // Prefer branches that actually encode loop continuation/exit.
    if (Succ0InLoop == Succ1InLoop)
      return nullptr;

    Value *Cond = Br->getCondition();
    if (!(isa<CmpInst>(Cond) || isa<UnaryInstruction>(Cond) ||
          isa<BinaryOperator>(Cond) || isa<CallInst>(Cond)))
      return nullptr;

    if (isa<CallInst>(Cond))
      loopCondCalls.insert(cast<CallInst>(Cond));

    negateCondition = !Succ0InLoop;
    return cast<Instruction>(Cond);
  };

  auto *Header = L->getHeader();
  auto *Latch = L->getLoopLatch();
  auto *HeaderBr = dyn_cast<BranchInst>(Header ? Header->getTerminator() : nullptr);
  auto *LatchBr =
      dyn_cast<BranchInst>(Latch ? Latch->getTerminator() : nullptr);

  // Rotated/do-while-like loops typically keep the canonical loop guard at latch.
  if (Instruction *Cond = pickCondFromBranch(LatchBr))
    return Cond;

  if (Instruction *Cond = pickCondFromBranch(HeaderBr))
    return Cond;

  // Legacy fallback: preserve previous behavior for unusual loop shapes.
  if (HeaderBr && HeaderBr->isConditional()) {
    Value *Cond = HeaderBr->getCondition();
    if (isa<CmpInst>(Cond) || isa<UnaryInstruction>(Cond) ||
        isa<BinaryOperator>(Cond) || isa<CallInst>(Cond)) {
      if (isa<CallInst>(Cond))
        loopCondCalls.insert(cast<CallInst>(Cond));
      BasicBlock *Succ0 = HeaderBr->getSuccessor(0);
      if (!L->contains(Succ0))
        negateCondition = true;
      return cast<Instruction>(Cond);
    }
  }

  if (LatchBr && LatchBr->isConditional()) {
    Value *Cond = LatchBr->getCondition();
    if (isa<CmpInst>(Cond) || isa<UnaryInstruction>(Cond) ||
        isa<BinaryOperator>(Cond) || isa<CallInst>(Cond)) {
      if (isa<CallInst>(Cond))
        loopCondCalls.insert(cast<CallInst>(Cond));
      BasicBlock *Succ0 = LatchBr->getSuccessor(0);
      if (!L->contains(Succ0))
        negateCondition = true;
      return cast<Instruction>(Cond);
    }
  }

  return nullptr;
}

LoopProfile *CWriter::findLoopProfile(Loop *L) {
  for (auto LP : LoopProfiles)
    if (LP->L == L) {
      errs() << "SUSAN: found LP for L:" << *L << "\n";
      if (LP->isOmpLoop)
        errs() << "isomp\n";
      return LP;
    }
  return nullptr;
}

void CWriter::findCondRelatedInsts(BasicBlock *skipBlock,
                                   std::set<Value *> &condRelatedInsts) {

  if (!skipBlock)
    return;

  Instruction *term = skipBlock->getTerminator();
  std::queue<Instruction *> toVisit;
  std::set<Instruction *> visited;

  toVisit.push(term);
  visited.insert(term);

  while (!toVisit.empty()) {
    Instruction *currInst = toVisit.front();
    toVisit.pop();

    for (Value *opnd : currInst->operands()) {
      Instruction *usedInst = dyn_cast<Instruction>(opnd);
      if (usedInst && usedInst->getParent() == skipBlock &&
          visited.find(usedInst) == visited.end()) {
        toVisit.push(usedInst);
        visited.insert(usedInst);
        condRelatedInsts.insert(usedInst);
      }
    }
  }
}

// header can be skipped if there's no insts with side effect
bool CWriter::canSkipHeader(BasicBlock *header) {
  Value *cmp = nullptr;
  BranchInst *term = dyn_cast<BranchInst>(header->getTerminator());
  if (term && term->isConditional())
    cmp = term->getCondition();

  for (BasicBlock::iterator I = header->begin();
       cast<Instruction>(I) != cmp && I != header->end() && !isa<BranchInst>(I);
       ++I) {
    Instruction *inst = &*I;

    if (isSkipableInst(inst))
      continue;

    bool relatedToControl = false;
    for (User *U : inst->users())
      if (U == cmp || U == term) {
        relatedToControl = true;
        break;
      }
    if (relatedToControl)
      continue;

    return false;
  }

  return true;
}

void CWriter::initializeLoopPHIs(Loop *L, bool skipMainIV) {
  errs() << "CBackend: initializeLoopPHIs loop_header="
         << L->getHeader()->getName()
         << " skip_main_iv=" << (skipMainIV ? "1" : "0") << "\n";
  BasicBlock *Header = L->getHeader();
  for (BasicBlock::iterator I = Header->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    if (deadInsts.find(PN) != deadInsts.end())
      continue;
    if (skipMainIV && isInductionVariable(cast<Value>(PN)) &&
        PN->getType()->isIntegerTy()) {
      errs() << "CBackend: skip loop PHI initializer phi=" << PN->getName()
             << " reason=main_induction loop_header="
             << L->getHeader()->getName() << "\n";
      continue;
    }
    if (isInductionVariable(cast<Value>(PN)) && PN->getType()->isIntegerTy() &&
        !skipMainIV) {
      errs() << "CBackend: seed main induction PHI phi=" << PN->getName()
             << " loop_header=" << L->getHeader()->getName() << "\n";
    }
    for (unsigned i = 0; i < PN->getNumIncomingValues(); ++i) {
      BasicBlock *predBB = PN->getIncomingBlock(i);
      Loop *predBBL = LI->getLoopFor(predBB);
      if (predBBL && predBBL == L)
        continue;

      errs() << "CBackend: emit loop PHI initializer phi=" << PN->getName()
             << " from_pred=" << predBB->getName()
             << " incoming=" << *PN->getIncomingValue(i)
             << " loop_header=" << L->getHeader()->getName() << "\n";
      Out << GetValueName(PN) << " = ";
      writeOperandInternal(PN->getIncomingValue(i));
      Out << ";\n";
    }
  }
}

void CWriter::emitLoopPHIInitializers(Loop *L, bool skipMainIV) {
  initializeLoopPHIs(L, skipMainIV);
}

void CWriter::printPHIsIfNecessary(BasicBlock *BB) {
  for (auto bb2phi : PHIValues2Print) {
    if (bb2phi.first == BB) {
      PHINode *phi = bb2phi.second;
      if (deadInsts.find(phi) != deadInsts.end())
        continue;
      if (NATURAL_CONTROL_FLOW) {
        Loop *phiLoop = LI->getLoopFor(phi->getParent());
        LoopProfile *phiLP = phiLoop ? findLoopProfile(phiLoop) : nullptr;
        if (phiLoop && phiLoop->getHeader() == phi->getParent() &&
            phiLP && phiLP->isForLoop) {
          errs() << "CBE_DEBUG: skipping predecessor PHI copy for natural for-loop header phi "
                 << *phi << " from pred " << BB->getName() << "\n";
          continue;
        }
        // Do-while loops emit loop-carried header PHI updates explicitly
        // at the end of printDoWhileLoop(). Suppress duplicate edge copies
        // from loop-internal predecessors for those header PHIs.
        if (phiLoop && phiLoop->getHeader() == phi->getParent() &&
            phiLP && !phiLP->isForLoop && phiLoop->contains(BB)) {
          BasicBlock *Latch = phiLoop->getLoopLatch();
          auto *LatchBr =
              dyn_cast_or_null<BranchInst>(Latch ? Latch->getTerminator() : nullptr);
          bool LatchIsExitingCond = false;
          if (LatchBr && LatchBr->isConditional()) {
            bool S0In = phiLoop->contains(LatchBr->getSuccessor(0));
            bool S1In = phiLoop->contains(LatchBr->getSuccessor(1));
            LatchIsExitingCond = (S0In != S1In);
          }
          if (LatchIsExitingCond) {
            errs() << "CBE_DEBUG: skipping predecessor PHI copy for do-while header phi "
                   << *phi << " from pred " << BB->getName() << "\n";
            continue;
          }
        }
      }
      // Only suppress PHI copies for canonical main integer IVs in for-loops.
      // Extra IVs (e.g. p+1 carriers) still need edge copies so their entry
      // initialization is materialized correctly.
      if (isInductionVariable(phi) && phi->getType()->isIntegerTy()) {
        Loop *phiLoop = LI->getLoopFor(phi->getParent());
        LoopProfile *phiLP = phiLoop ? findLoopProfile(phiLoop) : nullptr;
        if (phiLP && phiLP->isForLoop)
          continue;
      }
      Value *incomingVal = phi->getIncomingValueForBlock(BB);
      // Canonicalize simple predecessor-local LCSSA carriers. Without this,
      // name coalescing can turn merge copies like "retval = lg.lcssa" into
      // a useless self-assignment "retval = retval".
      if (auto *incomingPN = dyn_cast<PHINode>(incomingVal)) {
        if (incomingPN->getParent() == BB &&
            incomingPN->getNumIncomingValues() == 1)
          incomingVal = incomingPN->getIncomingValue(0);
      }
      errs() << "CBE_DEBUG: emitting PHI copy from pred "
             << BB->getParent()->getName() << "::" << BB->getName()
             << " to phi " << *phi << "\n";
      auto varName = GetValueName(phi);
      if (incomingVal == phi) {
        errs() << "CBE_DEBUG: skipping self PHI copy for "
               << BB->getParent()->getName() << "::" << BB->getName()
               << " var=" << varName << "\n";
        continue;
      }
      if (isa<Instruction>(incomingVal) || isa<Argument>(incomingVal)) {
        auto incomingName = GetValueName(incomingVal);
        // Be conservative when suppressing PHI edge copies:
        // IR naming can intentionally coalesce related values (e.g. loads and
        // their PHI), so name equality alone is not enough to prove a true
        // no-op assignment.
        // Be strict with PHI incoming values: name equality alone is not
        // sufficient to prove a no-op on control-flow merge edges.
        // Example: retval PHI can incoming-from-loop as lg.lcssa and still
        // require an explicit edge copy even when names are coalesced.
        bool isSafeNoOpCarrier = isa<Argument>(incomingVal);
        if (auto *incomingPN = dyn_cast<PHINode>(incomingVal))
          isSafeNoOpCarrier = (incomingPN->getParent() == phi->getParent());
        // Non-load instructions (e.g. arithmetic like %dec) that already
        // materialize into the same C variable name are typically true no-op
        // PHI edge updates and can be elided to avoid duplicate "x = x;".
        if (!isSafeNoOpCarrier && isa<Instruction>(incomingVal) &&
            !isa<PHINode>(incomingVal) &&
            !isIVIncrement(incomingVal) && !isExtraIVIncrement(incomingVal) &&
            !isa<LoadInst>(incomingVal) && !incomingName.empty() &&
            incomingName == varName) {
          errs() << "CBE_DEBUG: skipping non-load no-op PHI copy for "
                 << BB->getParent()->getName() << "::" << BB->getName()
                 << " var=" << varName << "\n";
          continue;
        }
        if (isSafeNoOpCarrier && !incomingName.empty() &&
            incomingName == varName) {
          errs() << "CBE_DEBUG: skipping no-op PHI copy for "
                 << BB->getParent()->getName() << "::" << BB->getName()
                 << " var=" << varName << "\n";
          continue;
        }
      }
      Out << std::string(2, ' ');
      if (declaredLocals.find(varName) == declaredLocals.end() &&
          omp_declaredLocals.find(varName) == omp_declaredLocals.end()) {
        errs() << "CBE_DEBUG: PHI edge-local declaration fallback for "
               << BB->getParent()->getName() << "::" << BB->getName()
               << " var=" << varName << " phi=" << *phi << "\n";
        printTypeName(Out, phi->getType(), false) << ' ';
        errs() << "SUSAN: printing varname 6842: " << varName << "\n";
        if (!IS_OPENMP_FUNCTION)
          declaredLocals.insert(varName);
        else
          omp_declaredLocals.insert(varName);
      }
      Out << GetValueName(phi) << " = ";
      // YEBIN: cheat for cuda dims
      // TODO: deal with collapse
      if (phi->getMetadata("tulip.cuda.indvar")) {
        errs() << "YEBIN: PRINT CUDA INDVAR\n";
        Out << "blockDim.x * blockIdx.x + threadIdx.x + ";
      }
      errs() << "CBE_DEBUG: PHI incoming value: "
             << *incomingVal << "\n";
      if (Instruction *incomingInst = dyn_cast<Instruction>(incomingVal)) {
        std::string incomingName = GetValueName(incomingInst);
        if (isIVIncrement(incomingInst) &&
            (incomingName.empty() || incomingName == varName)) {
          errs() << "CBE_DEBUG: materializing exit-carried IV increment inline for phi "
                 << *phi << " from " << *incomingInst << "\n";
          writeInstComputationInline(*incomingInst);
        } else {
          writeOperandInternal(incomingVal);
        }
      } else {
        writeOperandInternal(incomingVal);
      }
      Out << ";\n";
    }
  }
}

void CWriter::emitPHIsForPredecessor(BasicBlock *BB) {
  printPHIsIfNecessary(BB);
}

void CWriter::searchForBlocksToSkip(Loop *L,
                                    std::set<BasicBlock *> &skipBlocks) {
  BasicBlock *skipBlock = nullptr;
  if (L->getBlocks().size() > 1) {
    skipBlock = findDoWhileExitingLatchBlock(L);
    skipBlocks.insert(skipBlock);
  }

  for (unsigned i = 0, e = L->getBlocks().size(); i != e; ++i) {
    BasicBlock *BB = L->getBlocks()[i];
    if (BB == L->getHeader()) {
      for (auto succ = succ_begin(BB); succ != succ_end(BB); ++succ) {
        BasicBlock *succBB = *succ;
        if (skipBlock && succBB == skipBlock) {
          skipBlocks.insert(BB);
          return;
        }
      }
    }
  }
}

void CWriter::FindLiveInsFor(Loop *L, Value *val) {
  errs() << "SUSAN: finding liveins for Loop" << L->getHeader()->getName()
         << "\n";
  Instruction *inst = dyn_cast<Instruction>(val);
  if (!inst)
    return;
  if (isExtraInductionVariable(inst))
    return;

  bool isLiveIn = true;
  for (unsigned i = 0, e = L->getBlocks().size(); i != e; ++i) {
    BasicBlock *BB = L->getBlocks()[i];
    if (BB == inst->getParent()) {
      isLiveIn = false;
      break;
    }
  }

  if (!isDirectAlloca(inst) && !isInlinableInst(*inst) && isLiveIn) {
    errs() << "SUSAN: found livein" << *inst << "\n";
    omp_liveins[L].insert(inst);
  }

  std::queue<Instruction *> toVisit;
  std::set<Instruction *> visited;
  toVisit.push(inst);
  visited.insert(inst);
  while (!toVisit.empty()) {
    Instruction *currInst = toVisit.front();
    toVisit.pop();

    for (Value *opnd : currInst->operands()) {
      Instruction *usedInst = dyn_cast<Instruction>(opnd);
      if (!usedInst)
        continue;

      bool skipInst = false;
      for (unsigned i = 0, e = L->getBlocks().size(); i != e; ++i) {
        BasicBlock *BB = L->getBlocks()[i];
        if (BB == usedInst->getParent()) {
          skipInst = true;
          break;
        }
      }
      if (skipInst)
        continue;

      if (visited.find(usedInst) == visited.end() &&
          !isDirectAlloca(usedInst) && !isInlinableInst(*usedInst)) {
        toVisit.push(usedInst);
        visited.insert(usedInst);
        errs() << "SUSAN: found livein at used" << *usedInst << "\n";
        omp_liveins[L].insert(usedInst);
      }
    }

    for (User *U : currInst->users()) {
      Instruction *userInst = dyn_cast<Instruction>(U);
      if (!userInst)
        continue;

      bool skipInst = false;
      for (unsigned i = 0, e = L->getBlocks().size(); i != e; ++i) {
        BasicBlock *BB = L->getBlocks()[i];
        if (BB == userInst->getParent()) {
          skipInst = true;
          break;
        }
      }
      if (skipInst)
        continue;

      if (StoreInst *store = dyn_cast<StoreInst>(U)) {
        if (store->getPointerOperand() == cast<Value>(currInst) &&
            !isInlinableInst(*userInst)) {
          toVisit.push(store);
          visited.insert(store);
          errs() << "SUSAN: found livein at store" << *store << "\n";
          omp_liveins[L].insert(store);
        }
      }
    }
  }
}

bool is_number(const std::string &s) {
  return !s.empty() && std::find_if(s.begin(), s.end(), [](unsigned char c) {
                         return !std::isdigit(c);
                       }) == s.end();
}

bool CWriter::isSkipableInst(Instruction *inst) {
  // if(noneSkipAllocaInsts.find(inst) != noneSkipAllocaInsts.end())return
  // false;
  if (isAtomicAddLoweringCandidate(inst))
    return false;
  if (omp_SkipVals.find(inst) != omp_SkipVals.end())
    return true;
  // if(skipInstsForPhis.find(inst) != skipInstsForPhis.end()) return true;
  if (deadInsts.find(inst) != deadInsts.end()) {
    errs() << "Dead inst " << *inst << "\n";
    return true;
  }
  if (deleteAndReplaceInsts.find(inst) != deleteAndReplaceInsts.end())
    return true;
  if (isa<PHINode>(inst))
    return true;
  if (isInlinableInst(*inst)) {
    errs() << "Inlinable inst " << *inst << "\n";
    return true;
  }
  if (isDirectAlloca(inst))
    return true;
  if (isIVIncrement(inst))
    return true;
  if (isExtraIVIncrement(inst))
    return true;

  if (CallInst *CI = dyn_cast<CallInst>(inst))
    if (Function *F = CI->getCalledFunction())
      if (F->getIntrinsicID() == Intrinsic::dbg_value ||
          F->getIntrinsicID() == Intrinsic::dbg_declare) {
        return true;
      }

  return false;
}

void CWriter::OMP_RecordLiveIns(LoopProfile *LP) {
  Loop *L = LP->L;
  errs() << "SUSAN: recording livein for loop: " << *L << "\n";
  std::set<BasicBlock *> skipBlocks;
  searchForBlocksToSkip(L, skipBlocks);
  for (unsigned i = 0, e = L->getBlocks().size(); i != e; ++i) {
    BasicBlock *BB = L->getBlocks()[i];
    if (skipBlocks.find(BB) != skipBlocks.end())
      continue;
    errs() << "SUSAN: finding live-in for" << BB->getName() << "\n";

    for (auto &I : *BB) {
      bool negateCondition;
      if (LP->isOmpLoop && findCondInst(L, negateCondition) == &I)
        continue;
      FindLiveInsFor(L, &I);
    }
  }
  errs() << "SUSAN: finding live-in for lb" << *LP->lb << "\n";
  FindLiveInsFor(L, LP->lb);
  errs() << "SUSAN: finding live-in for ub" << *LP->ub << "\n";
  FindLiveInsFor(L, LP->ub);
}

void CWriter::printBasicBlock(BasicBlock *BB, std::set<Value *> skipInsts) {
  errs() << "CBEBackend: printing bb 7082 " << BB->getName() << "\n";

  if (NATURAL_CONTROL_FLOW) {
    // Don't print the label for the basic block if there are no uses, or if
    // the only terminator use is the predecessor basic block's terminator.
    // We have to scan the use list because PHI nodes use basic blocks too but
    // do not require a label to be generated.

    if (printLabels.find(BB) != printLabels.end()) {
      Out << GetValueName(BB) << ":";
      // A label immediately before a late variable declaration is problematic,
      // because "a label can only be part of a statement and a declaration is
      // not a statement" (GCC). Adding a ";" is a simple workaround.
      if (DeclareLocalsLate) {
        Out << ";";
      }
      Out << "\n";
    }
  } else {
    bool NeedsLabel = false;
    for (pred_iterator PI = pred_begin(BB), E = pred_end(BB); PI != E; ++PI)
      if (isGotoCodeNecessary(*PI, BB)) {
        NeedsLabel = true;
        break;
      }

    if (NeedsLabel) {
      Out << GetValueName(BB) << ":";
      // A label immediately before a late variable declaration is problematic,
      // because "a label can only be part of a statement and a declaration is
      // not a statement" (GCC). Adding a ";" is a simple workaround.
      if (DeclareLocalsLate) {
        Out << ";";
      }
      Out << "\n";
    }
  }

  // Output all of the instructions in the basic block...
  for (BasicBlock::iterator II = BB->begin(), E = --BB->end(); II != E; ++II) {
    Instruction *inst = &*II;
    if (isSkipableInst(inst)) {
      errs() << "Skipping inst " << *inst << "\n";
      continue;
    }
    if (emitOpenMPAtomicForInst(inst))
      continue;

    /*
     * OpenMP: translate omp master
     */
    if (CallInst *CI = dyn_cast<CallInst>(inst)) {
      if (Function *ompCall = CI->getCalledFunction()) {
        if (ompCall->getName().contains("__kmpc_master")) {
          Out << "#pragma omp master\n{\n";
          continue;
        }
        if (ompCall->getName().contains("__kmpc_end_master")) {
          Out << "}\n";
          continue;
        }
      }
    }

    // if(noneSkipAllocaInsts.find(&*II) != noneSkipAllocaInsts.end()){
    //   AllocaInst *alloca = dyn_cast<AllocaInst>(&*II);
    //   PointerType *ptrTy = dyn_cast<PointerType>(II->getType());
    //   if(ptrTy){
    //     errs() << "SUSAN: printing type name at 7200 " << *alloca << "\n";
    //     printTypeName(Out, ptrTy->getPointerElementType(), false) << ' ';
    //   }
    //   Out << GetValueName(&*II);

    //  Type *currTy = alloca->getAllocatedType();
    //  while(ArrayType *arrTy = dyn_cast<ArrayType>(currTy)){
    //    Out << '[' << arrTy->getNumElements() << ']';
    //    currTy = arrTy->getElementType();
    //  }
    //  Out << ";\n";
    //} else
    if(isInlinableInst(*II)) errs() << *II << " is inlinable\n";
    if (!isInlinableInst(*II)) {
      bool emitsLHS = ((&*II)->user_begin() != (&*II)->user_end() &&
                       !isEmptyType(II->getType()) && !isInlineAsm(*II));
      if (!emitsLHS && !II->mayHaveSideEffects()) {
        errs() << "CBEBackend: skipping side-effect-free stmt " << *II << "\n";
        continue;
      }
      errs() << "SUSAN: printing instruction " << *II << " at 6678\n";
      if (CallInst *CI = dyn_cast<CallInst>(&*II)) {
        if (Function *Called = CI->getCalledFunction()) {
          if (Called->getName().contains("cudaMemcpy")) {
            errs() << "ANDREW: printBasicBlock dispatching cudaMemcpy to "
                      "printInstruction: "
                   << *CI << "\n";
            printInstruction(CI, true);
            continue;
          }
        }
      }
      if (!isEmptyType(II->getType()) || isa<StoreInst>(&*II))
        Out << "  ";

      if (emitsLHS) {
        auto varName = GetValueName(&*II);
        if (declaredLocals.find(varName) == declaredLocals.end() &&
            omp_declaredLocals.find(varName) == omp_declaredLocals.end()) {
          auto varName = GetValueName(&*II, true);
          auto typeName = II->getType();
          for (auto [argInput, type] : type2declare) {
            auto argInputName = GetValueName(argInput);
            if (argInputName == varName)
              typeName = type;
          }
          if (mallocType.find(&*II) != mallocType.end()) {
            typeName = mallocType[&*II];
          }
          printTypeName(Out, typeName, false) << ' ';
          errs() << "SUSAN: printing varname 7310: " << varName << "\n";
          if (!IS_OPENMP_FUNCTION)
            declaredLocals.insert(varName);
          else
            omp_declaredLocals.insert(varName);
          nameDict.LocalVars[demangleFunctionName(II->getFunction()->getName())].insert(varName);
        }
        Out << varName << " = ";
      }
      writeInstComputationInline(*II);

      if (!isEmptyType(II->getType()) || isa<StoreInst>(&*II))
        Out << ";\n";
    }
  }

  ////check if a phi value need to be printed
  printPHIsIfNecessary(BB);

  // Don't emit prefix or suffix for the terminator.
  visit(*BB->getTerminator());
}

// Specific Instruction type classes... note that all of the casts are
// necessary because we use the instruction classes as opaque types...
void CWriter::visitReturnInst(ReturnInst &I) {
  CurInstr = &I;
  if (deadInsts.find(&I) != deadInsts.end())
    return;

  // If this is a struct return function, return the temporary struct.
  bool isStructReturn = I.getParent()->getParent()->hasStructRetAttr();

  if (isStructReturn) {
    Out << "  return StructReturn;\n";
    return;
  }

  // Don't output a void return if this is the last basic block in the function
  // unless that would make the basic block empty
  if (I.getNumOperands() == 0 &&
      &*--I.getParent()->getParent()->end() == I.getParent() &&
      &*I.getParent()->begin() != &I) {
    return;
  }

  Out << "  return";
  if (I.getNumOperands()) {
    Out << ' ';
    writeOperand(I.getOperand(0), ContextCasted);
  }
  Out << ";\n";
}

void CWriter::naturalSwitchTranslation(SwitchInst &SI) {
  CurInstr = &SI;
  BasicBlock *switchBB = SI.getParent();

  Value *Cond = SI.getCondition();
  unsigned NumBits = cast<IntegerType>(Cond->getType())->getBitWidth();

  if (SI.getNumCases() == 0) { // unconditional branch
    // printPHICopiesForSuccessor(SI.getParent(), SI.getDefaultDest(), 2);
    printBranchToBlock(SI.getParent(), SI.getDefaultDest(), 2);
    Out << "\n";

  } else if (NumBits <= 64) { // model as a switch statement
    Out << "  switch (";
    writeOperand(Cond);
    // Out << ") {\n  default:\n";
    Out << ") {\n";
    // printPHICopiesForSuccessor(SI.getParent(), SI.getDefaultDest(), 2);
    // emitSwitchBlock(SI.getDefaultDest(), switchBB);
    // Out << "    break;\n";

    std::map<BasicBlock *, std::set<ConstantInt *>> groupSameCases;
    for (SwitchInst::CaseIt i = SI.case_begin(), e = SI.case_end(); i != e;
         ++i) {
      ConstantInt *CaseVal = i->getCaseValue();
      BasicBlock *Succ = i->getCaseSuccessor();
      if (groupSameCases.find(Succ) == groupSameCases.end())
        groupSameCases[Succ] = {CaseVal};
      else
        groupSameCases[Succ].insert(CaseVal);
    }

    for (auto [caseBB, caseVals] : groupSameCases) {
      for (auto caseVal : caseVals) {
        Out << "  case ";
        writeOperand(caseVal);
        Out << ":\n";
      }
      emitSwitchBlock(caseBB, switchBB);

      // if succBB is exiting function, then don't print break but print return
      if (BasicBlock *ret = isExitingFunction(caseBB))
        printInstruction(ret->getTerminator());
      else
        Out << "    break;\n";
    }

    Out << "  default:\n";
    // printPHICopiesForSuccessor(SI.getParent(), SI.getDefaultDest(), 2);
    emitSwitchBlock(SI.getDefaultDest(), switchBB);
    Out << "    break;\n";

    Out << "  }\n";

  } else { // model as a series of if statements
    Out << "  ";
    for (SwitchInst::CaseIt i = SI.case_begin(), e = SI.case_end(); i != e;
         ++i) {
      Out << "if (";
      ConstantInt *CaseVal = i->getCaseValue();
      BasicBlock *Succ = i->getCaseSuccessor();
      ICmpInst *icmp = new ICmpInst(CmpInst::ICMP_EQ, Cond, CaseVal);
      visitICmpInst(*icmp);
      delete icmp;
      Out << ") {\n";
      // printPHICopiesForSuccessor(SI.getParent(), Succ, 2);
      printBranchToBlock(SI.getParent(), Succ, 2);
      Out << "  } else ";
    }
    Out << "{\n";
    // printPHICopiesForSuccessor(SI.getParent(), SI.getDefaultDest(), 2);
    printBranchToBlock(SI.getParent(), SI.getDefaultDest(), 2);
    Out << "  }\n";
  }
  Out << "\n";
}

void CWriter::visitSwitchInst(SwitchInst &SI) {
  if (NATURAL_CONTROL_FLOW) {
    naturalSwitchTranslation(SI);
    return;
  }

  CurInstr = &SI;

  Value *Cond = SI.getCondition();
  unsigned NumBits = cast<IntegerType>(Cond->getType())->getBitWidth();

  if (SI.getNumCases() == 0) { // unconditional branch
    // printPHICopiesForSuccessor(SI.getParent(), SI.getDefaultDest(), 2);
    printBranchToBlock(SI.getParent(), SI.getDefaultDest(), 2);
    Out << "\n";

  } else if (NumBits <= 64) { // model as a switch statement
    Out << "  switch (";
    writeOperand(Cond);
    Out << ") {\n  default:\n";
    // printPHICopiesForSuccessor(SI.getParent(), SI.getDefaultDest(), 2);
    printBranchToBlock(SI.getParent(), SI.getDefaultDest(), 2);

    // Skip the first item since that's the default case.
    for (SwitchInst::CaseIt i = SI.case_begin(), e = SI.case_end(); i != e;
         ++i) {
      ConstantInt *CaseVal = i->getCaseValue();
      BasicBlock *Succ = i->getCaseSuccessor();
      Out << "  case ";
      writeOperand(CaseVal);
      Out << ":\n";
      // printPHICopiesForSuccessor(SI.getParent(), Succ, 2);
      if (isGotoCodeNecessary(SI.getParent(), Succ))
        printBranchToBlock(SI.getParent(), Succ, 2);
      else
        Out << "    break;\n";
    }
    Out << "  }\n";

  } else { // model as a series of if statements
    Out << "  ";
    for (SwitchInst::CaseIt i = SI.case_begin(), e = SI.case_end(); i != e;
         ++i) {
      Out << "if (";
      ConstantInt *CaseVal = i->getCaseValue();
      BasicBlock *Succ = i->getCaseSuccessor();
      ICmpInst *icmp = new ICmpInst(CmpInst::ICMP_EQ, Cond, CaseVal);
      visitICmpInst(*icmp);
      delete icmp;
      Out << ") {\n";
      // printPHICopiesForSuccessor(SI.getParent(), Succ, 2);
      printBranchToBlock(SI.getParent(), Succ, 2);
      Out << "  } else ";
    }
    Out << "{\n";
    // printPHICopiesForSuccessor(SI.getParent(), SI.getDefaultDest(), 2);
    printBranchToBlock(SI.getParent(), SI.getDefaultDest(), 2);
    Out << "  }\n";
  }
  Out << "\n";
}

void CWriter::visitIndirectBrInst(IndirectBrInst &IBI) {
  CurInstr = &IBI;

  Out << "  goto *(void*)(";
  writeOperand(IBI.getOperand(0));
  Out << ");\n";
}

void CWriter::visitUnreachableInst(UnreachableInst &I) {
  CurInstr = &I;

  //Out << "exit(0);\n\n";
  //headerUseUnreachable();
  //Out << "  __builtin_unreachable();\n\n";
}

bool CWriter::isGotoCodeNecessary(BasicBlock *From, BasicBlock *To) {
  if (NATURAL_CONTROL_FLOW)
    return false;
  else
    return true; // Not the direct successor, we need a goto.
}

bool CWriter::alreadyPrintedPHIVal(BasicBlock *predBB, PHINode *phi) {

  for (auto bb2phi : PHIValues2Print)
    if (bb2phi.first == predBB && bb2phi.second == phi)
      return false;

  return true;
}

void CWriter::printPHICopiesForAllPhis(BasicBlock *CurBlock, unsigned Indent) {
  std::queue<BasicBlock *> toVisit;
  std::set<BasicBlock *> visited;
  toVisit.push(CurBlock);
  visited.insert(CurBlock);
  while (!toVisit.empty()) {
    BasicBlock *currBB = toVisit.front();
    toVisit.pop();
    for (auto succ = succ_begin(currBB); succ != succ_end(currBB); ++succ) {
      BasicBlock *succBB = *succ;
      if (visited.find(succBB) == visited.end()) {

        // print phi value for the successors
        for (BasicBlock::iterator I = succBB->begin(); isa<PHINode>(I); ++I) {
          PHINode *PN = cast<PHINode>(I);
          if (PN->getBasicBlockIndex(CurBlock) >= 0 &&
              !alreadyPrintedPHIVal(CurBlock, PN)) {
            // printedPHIValues.insert(std::make_pair(CurBlock, PN));
            //  Now we have to do the printing.
            Value *IV = PN->getIncomingValueForBlock(CurBlock);
            if (!isa<UndefValue>(IV) && !isEmptyType(IV->getType())) {
              Out << std::string(Indent, ' ');
              Out << "  " << GetValueName(&*I) << "__PHI_TEMPORARY = ";
              writeOperand(IV, ContextCasted);
              Out << ";   /* for PHI node */\n";
            }
          }
        }

        visited.insert(succBB);
        toVisit.push(succBB);
      }
    }
  }
}
void CWriter::printPHICopiesForSuccessor(BasicBlock *CurBlock,
                                         BasicBlock *Successor,
                                         unsigned Indent) {
  for (BasicBlock::iterator I = Successor->begin(); isa<PHINode>(I); ++I) {
    PHINode *PN = cast<PHINode>(I);
    Value *IV = PN->getIncomingValueForBlock(CurBlock);
    if (!isa<UndefValue>(IV) && !isEmptyType(IV->getType()) &&
        !alreadyPrintedPHIVal(CurBlock, PN)) {
      // printedPHIValues.insert(std::make_pair(CurBlock, PN));

      Out << " " << GetValueName(PN) << " = ";
      writeOperandInternal(IV);
      Out << ";\n";
    }
  }
}

void CWriter::printBranchToBlock(BasicBlock *CurBB, BasicBlock *Succ,
                                 unsigned Indent) {
  if (NATURAL_CONTROL_FLOW) {
    BranchInst *br = dyn_cast<BranchInst>(CurBB->getTerminator());
    if (gotoBranches.find(br) != gotoBranches.end()) {
      Out << std::string(Indent, ' ') << "  goto ";
      writeOperand(Succ);
      Out << ";\n";
    }
    return;
  }

  if (isGotoCodeNecessary(CurBB, Succ)) {
    Out << std::string(Indent, ' ') << "  goto ";
    writeOperand(Succ);
    Out << ";\n";
  }
}

void CWriter::emitSwitchBlock(BasicBlock *start, BasicBlock *brBlock) {

  Region *swRegion = RI->getRegionFor(brBlock);

  // if switch statement is captured by region, translate it using regioninfo
  if (swRegion->getEntry() == brBlock) {

    auto times2bePrintedBefore = times2bePrinted;
    BasicBlock *exitBB = swRegion->getExit();
    for (Region::block_iterator I = swRegion->block_begin(),
                                E = swRegion->block_end();
         I != E; ++I) {
      BasicBlock *currBB = cast<BasicBlock>(*I);
      if (directPathFromAtoBwithoutC(start, currBB, exitBB) &&
          times2bePrinted[currBB] == times2bePrintedBefore[currBB]) {
        printBasicBlock(currBB);
        times2bePrinted[currBB]--;
      }
    }
  } else {
    std::set<BasicBlock *> visited;
    std::queue<BasicBlock *> toVisit;
    visited.insert(start);
    toVisit.push(start);
    while (!toVisit.empty()) {
      BasicBlock *currBB = toVisit.front();

      // TODO: need a systematic way to check when does a branch end
      // currently just adding patches here and there
      // e.x.,: currBB == otherStart is added from supermutation
      if (PDT->dominates(currBB, brBlock)) {
        break;
      }

      if (!times2bePrinted[currBB]) {
        toVisit.pop();
        continue;
      }

      printBasicBlock(currBB);
      times2bePrinted[currBB]--;

      toVisit.pop();

      for (auto succ = succ_begin(currBB); succ != succ_end(currBB); ++succ) {
        BasicBlock *succBB = *succ;
        if (visited.find(succBB) == visited.end()) {
          visited.insert(succBB);
          toVisit.push(succBB);
        }
      }
    }
  }
}

bool CWriter::edgeBelongsToSubRegions(BasicBlock *fromBB, BasicBlock *toBB,
                                      CBERegion *R, bool isElseBranch) {
  auto currRedges = isElseBranch ? R->elseEdges : R->thenEdges;
  for (auto edge : currRedges)
    if (edge.first == fromBB && edge.second == toBB)
      return true;

  std::queue<CBERegion *> toVisit;
  if (isElseBranch)
    for (auto subR : R->elseSubRegions)
      toVisit.push(subR);
  else
    for (auto subR : R->thenSubRegions)
      toVisit.push(subR);

  while (!toVisit.empty()) {
    CBERegion *currNode = toVisit.front();
    toVisit.pop();

    for (auto edge : currNode->thenEdges)
      if (edge.first == fromBB && edge.second == toBB)
        return true;
    for (auto edge : currNode->elseEdges)
      if (edge.first == fromBB && edge.second == toBB)
        return true;

    CBERegionMap[currNode->entryBlock] = currNode;
    for (CBERegion *subRegion : currNode->thenSubRegions) {
      toVisit.push(subRegion);
    }
    for (CBERegion *subRegion : currNode->elseSubRegions) {
      toVisit.push(subRegion);
    }
  }

  return false;
}

bool CWriter::nodeBelongsToRegion(BasicBlock *BB, CBERegion *R,
                                  bool isElseBranch) {

  if (isElseBranch && std::count(R->elseBBs.begin(), R->elseBBs.end(), BB))
    return true;
  if (!isElseBranch && std::count(R->thenBBs.begin(), R->thenBBs.end(), BB))
    return true;

  return false;
}

void CWriter::createSubRegionOrRecordCurrentRegion(BasicBlock *predBB,
                                                   BasicBlock *currBB,
                                                   CBERegion *R,
                                                   bool isElseBranch) {
  if (!edgeBelongsToSubRegions(predBB, currBB, R, isElseBranch)) {
    if (isElseBranch)
      R->elseEdges.push_back(std::make_pair(predBB, currBB));
    else
      R->thenEdges.push_back(std::make_pair(predBB, currBB));

    if (!nodeBelongsToRegion(currBB, R, isElseBranch)) {
      Instruction *br = currBB->getTerminator();
      if (std::count(ifBranches.begin(), ifBranches.end(), br)) {
        errs() << "creating subregion: " << currBB->getName() << "\n";

        CBERegion *newR = createNewRegion(currBB, R, isElseBranch);
        markBranchRegion(br, newR);
      }

      if (isElseBranch) {
        errs() << "SUSAN: adding block to else branch" << *currBB << "\n";
        R->elseBBs.push_back(currBB);
      } else {
        errs() << "SUSAN: adding block to if branch" << *currBB << "\n";
        R->thenBBs.push_back(currBB);
      }
    }
  }
}

void CWriter::recordTimes2bePrintedForBranch(BasicBlock *start,
                                             BasicBlock *brBlock,
                                             BasicBlock *otherStart,
                                             CBERegion *R, bool isElseBranch) {
  std::set<std::pair<BasicBlock *, BasicBlock *>> visited;
  std::set<BasicBlock *> visitedNodes;
  std::queue<std::pair<BasicBlock *, BasicBlock *>> toVisit;
  visited.insert(std::make_pair(brBlock, start));
  visitedNodes.insert(start);
  toVisit.push(std::make_pair(brBlock, start));

  while (!toVisit.empty()) {
    BasicBlock *currBB = toVisit.front().second;
    BasicBlock *predBB = toVisit.front().first;

    if (PDT->dominates(currBB, brBlock) || currBB == otherStart) {
      if (isExitingFunction(otherStart)) {
        if (isa<ReturnInst>(currBB->getTerminator()) ||
            isa<UnreachableInst>(currBB->getTerminator())) {
          createSubRegionOrRecordCurrentRegion(predBB, currBB, R, isElseBranch);
        }
      }
      break;
    }

    createSubRegionOrRecordCurrentRegion(predBB, currBB, R, isElseBranch);

    toVisit.pop();

    BranchInst *br = dyn_cast<BranchInst>(currBB->getTerminator());

    for (auto succ = succ_begin(currBB); succ != succ_end(currBB); ++succ) {
      BasicBlock *succBB = *succ;
      bool alreadyVisited = false;
      for (auto visitedEdge : visited)
        if (visitedEdge.first == currBB && visitedEdge.second == succBB)
          alreadyVisited = true;

      bool backEdgeDetected = false;
      for (auto backedge : backEdges)
        if (backedge.first == currBB && backedge.second == succBB)
          backEdgeDetected = true;

      Loop *L = LI->getLoopFor(succBB);
      if (L && L->getLoopLatch() == currBB) {
        errs() << "SUSAN: found latch" << currBB->getName() << "\n";
        backEdgeDetected = true;
      }

      if (!alreadyVisited && !backEdgeDetected) {
        visitedNodes.insert(succBB);
        visited.insert(std::make_pair(currBB, succBB));
        toVisit.push(std::make_pair(currBB, succBB));
      }
    }
  }
}

bool CWriter::noElseRegion(bool trueBranch, BasicBlock *brBB) {
  BranchInst *br = dyn_cast<BranchInst>(brBB->getTerminator());
  errs() << "SUSAN: noElseRegion " << trueBranch << "for br: " << *br << "\n";
  BasicBlock *startBB = trueBranch ? br->getSuccessor(1) : br->getSuccessor(0);

  // find immediate PD BB
  BasicBlock *pdBB = nullptr;
  std::queue<BasicBlock *> toVisit;
  std::set<BasicBlock *> visited;
  toVisit.push(startBB);
  visited.insert(startBB);
  std::set<BasicBlock *> inBetweenBBs;
  while (!toVisit.empty()) {
    BasicBlock *currBB = toVisit.front();
    toVisit.pop();

    if (PDT->dominates(currBB, brBB)) {
      pdBB = currBB;
      break;
    }
    inBetweenBBs.insert(currBB);

    for (auto succ = succ_begin(currBB); succ != succ_end(currBB); ++succ) {
      BasicBlock *succBB = *succ;
      if (visited.find(succBB) == visited.end()) {
        visited.insert(succBB);
        toVisit.push(succBB);
      }
    }
  }

  assert(pdBB && "pdBB not found\n");
  errs() << "SUSAN: found pdBB " << *pdBB << "\n";

  bool NoElseRegion = true;
  for (auto bb : inBetweenBBs) {
    for (auto &I : *bb) {
      if (!isa<BranchInst>(&I)) {
        NoElseRegion = false;
        break;
      }
    }
  }

  errs() << NoElseRegion << "\n";
  return NoElseRegion;
}

int CWriter::dominatedByReturn(BasicBlock *brBB) {
  Function *F = brBB->getParent();
  auto br = dyn_cast<BranchInst>(brBB->getTerminator());
  if (br->isConditional()) {
    auto succ0 = br->getSuccessor(0);
    auto succ1 = br->getSuccessor(1);
    auto singleSucc = succ0->getSingleSuccessor();
    if (singleSucc && isa<ReturnInst>(singleSucc->getTerminator()))
      return 0;
    singleSucc = succ1->getSingleSuccessor();
    if (singleSucc && isa<ReturnInst>(singleSucc->getTerminator()))
      return 1;
  }

  return -1;
}

// Branch instruction printing - Avoid printing out a branch to a basic block
// that immediately succeeds the current one.
void CWriter::visitBranchInst(BranchInst &I) {
  if (!I.isConditional()) {
    // printPHICopiesForSuccessor(I.getParent(), I.getSuccessor(0), 0);
    errs() << "printing unconditional branch " << I << "\n";
    printBranchToBlock(I.getParent(), I.getSuccessor(0), 0);
    return;
  }
}

// PHI nodes get copied into temporary values at the end of predecessor basic
// blocks.  We now need to copy these temporary values into the REAL value for
// the PHI.
void CWriter::visitPHINode(PHINode &I) {
  CurInstr = &I;

  // Canonical for-loop IVs (main and extra) are materialized as named symbols
  // in structured loop emission. Extra IVs still receive explicit edge copies
  // in printPHIsIfNecessary(), but their uses must not reference a
  // "__PHI_TEMPORARY" suffix.
  bool usePhiTemporary = true;
  if ((isInductionVariable(&I) || isExtraInductionVariable(&I)) &&
      I.getType()->isIntegerTy()) {
    if (Loop *phiLoop = LI->getLoopFor(I.getParent()))
      if (LoopProfile *phiLP = findLoopProfile(phiLoop))
        if (phiLP->isForLoop)
          usePhiTemporary = false;
  }

  writeOperand(&I);
  if (usePhiTemporary)
    Out << "__PHI_TEMPORARY";
}

void CWriter::visitUnaryOperator(UnaryOperator &I) {
  CurInstr = &I;

  // Currently the only unary operator supported is FNeg, which was introduced
  // in LLVM 8, although not fully exploited until later LLVM versions.
  // Older code uses a pseudo-FNeg pattern (-0.0 - x) which is matched in
  // visitBinaryOperator instead.
  if (I.getOpcode() != Instruction::FNeg) {
    DBG_ERRS("Invalid operator type !" << I);
    errorWithMessage("invalid operator type");
  }

  Value *X = I.getOperand(0);

  // We must cast the results of operations which might be promoted.
  bool needsCast = false;
  if ((I.getType() == Type::getInt8Ty(I.getContext())) ||
      (I.getType() == Type::getInt16Ty(I.getContext())) ||
      (I.getType() == Type::getFloatTy(I.getContext()))) {
    // types too small to work with directly
    needsCast = true;
  } else if (I.getType()->getPrimitiveSizeInBits() > 64) {
    // types too big to work with directly
    needsCast = true;
  }

  if (I.getType()->isVectorTy() || needsCast) {
    Type *VTy = I.getOperand(0)->getType();
    Out << "llvm_neg_";
    printTypeString(Out, VTy, false);
    Out << "(";
    writeOperand(X, ContextCasted);
    Out << ")";
    InlineOpDeclTypes.insert(std::pair<unsigned, Type *>(BinaryNeg, VTy));
  } else {
    Out << "-(";
    writeOperand(X);
    Out << ")";
  }
}

void CWriter::visitBinaryOperator(BinaryOperator &I) {
  using namespace PatternMatch;

  CurInstr = &I;
  // binary instructions, shift instructions, setCond instructions.
  cwriter_assert(!I.getType()->isPointerTy());

  // We must cast the results of binary operations which might be promoted.
  bool needsCast = false;
  if ((I.getType() == Type::getInt8Ty(I.getContext())) ||
      (I.getType() == Type::getInt16Ty(I.getContext())) ||
      (I.getType() == Type::getFloatTy(I.getContext()))) {
    // types too small to work with directly
    needsCast = true;
  } else if (I.getType()->getPrimitiveSizeInBits() > 64) {
    // types too big to work with directly
    needsCast = true;
  }
  bool shouldCast;
  bool castIsSigned;
  opcodeNeedsCast(I.getOpcode(), shouldCast, castIsSigned);

  if (I.getType()->isVectorTy() || needsCast || shouldCast) {
    Type *VTy = I.getOperand(0)->getType();
    unsigned opcode;
    Value *X;
    if (match(&I, m_Neg(m_Value(X)))) {
      opcode = BinaryNeg;
      // Out << "llvm_neg_";
      // printTypeString(Out, VTy, false);
      // Out << "(";
      // writeOperand(X, ContextCasted);
      Out << "-";
      writeOperand(X, ContextCasted);
    } else if (match(&I, m_FNeg(m_Value(X)))) {
      opcode = BinaryNeg;
      // Out << "llvm_neg_";
      // printTypeString(Out, VTy, false);
      // Out << "(";
      // writeOperand(X, ContextCasted);
      Out << "-";
      writeOperand(X, ContextCasted);
    } else if (match(&I, m_Not(m_Value(X)))) {
      opcode = BinaryNot;
      // Out << "llvm_not_";
      // printTypeString(Out, VTy, false);
      // Out << "(";
      // writeOperand(X, ContextCasted);
      Out << "~";
      writeOperand(X, ContextCasted);
    } else {
      opcode = I.getOpcode();
      auto writeInlineOperandWithParens = [&](Value *V) {
        bool needsParens =
            isa<Instruction>(V) && isInlinableInst(*cast<Instruction>(V));
        if (needsParens)
          Out << "(";
        writeOperand(V, ContextCasted);
        if (needsParens)
          Out << ")";
      };
      auto writeOperand1WithParensForNonAssoc = [&](unsigned Opc, Value *V) {
        bool needsParens =
            isa<Instruction>(V) && isInlinableInst(*cast<Instruction>(V)) &&
            (Opc == Instruction::Sub || Opc == Instruction::FSub ||
             Opc == Instruction::UDiv || Opc == Instruction::SDiv ||
             Opc == Instruction::FDiv || Opc == Instruction::URem ||
             Opc == Instruction::SRem || Opc == Instruction::FRem ||
             Opc == Instruction::Shl || Opc == Instruction::LShr ||
             Opc == Instruction::AShr);
        if (needsParens)
          Out << "(";
        writeOperand(V, ContextCasted);
        if (needsParens)
          Out << ")";
      };
      if (opcode == Instruction::Add || opcode == Instruction::FAdd) {
        if (ConstantInt *opnd0 = dyn_cast<ConstantInt>(I.getOperand(0))) {
          if (ConstantInt *opnd1 = dyn_cast<ConstantInt>(I.getOperand(1)))
            Out << (opnd0->getSExtValue() + opnd1->getSExtValue());
          else {
            if (addParenthesis.find(&I) != addParenthesis.end())
              Out << "(";
            writeInlineOperandWithParens(I.getOperand(0));
            Out << " + ";
            writeInlineOperandWithParens(I.getOperand(1));
            if (addParenthesis.find(&I) != addParenthesis.end())
              Out << ")";
          }
        } else {
          if (addParenthesis.find(&I) != addParenthesis.end())
            Out << "(";
          writeInlineOperandWithParens(I.getOperand(0));
          Out << " + ";
          writeInlineOperandWithParens(I.getOperand(1));
          if (addParenthesis.find(&I) != addParenthesis.end())
            Out << ")";
        }
      } else if (opcode == Instruction::Mul || opcode == Instruction::FMul) {
        writeInlineOperandWithParens(I.getOperand(0));
        Out << " * ";
        writeInlineOperandWithParens(I.getOperand(1));
      } else if (opcode == Instruction::URem || opcode == Instruction::FRem) {
        writeInlineOperandWithParens(I.getOperand(0));
        Out << " % ";
        writeOperand1WithParensForNonAssoc(opcode, I.getOperand(1));
      } else if (opcode == Instruction::SRem) {
        Type *op0Ty = (I.getOperand(0))->getType();
        Type *op1Ty = (I.getOperand(1))->getType();
        if (op0Ty->isIntegerTy(32))
          Out << "(int)";
        else if (op0Ty->isIntegerTy(64))
          Out << "(long long)";
        else if (op0Ty->isFloatTy())
          Out << "(float)";
        else if (op0Ty->isDoubleTy())
          Out << "(double)";
        else
          assert(0 && "SUSAN: op0Ty unimplemented cast?\n");
        writeInlineOperandWithParens(I.getOperand(0));
        Out << " % ";
        if (op1Ty->isIntegerTy(32))
          Out << "(int)";
        else if (op1Ty->isIntegerTy(64))
          Out << "(long long)";
        else if (op1Ty->isFloatTy())
          Out << "(float)";
        else if (op1Ty->isDoubleTy())
          Out << "(double)";
        else
          assert(0 && "SUSAN: op1Ty unimplemented cast?\n");
        writeInlineOperandWithParens(I.getOperand(1));
      } else if (opcode == Instruction::Sub || opcode == Instruction::FSub) {
        auto writeSubOperand1 = [&](Value *V) {
          bool needsParens =
              isa<Instruction>(V) && isInlinableInst(*cast<Instruction>(V));
          if (needsParens)
            Out << "(";
          writeOperand(V, ContextCasted);
          if (needsParens)
            Out << ")";
        };
        if (ConstantInt *opnd0 = dyn_cast<ConstantInt>(I.getOperand(0))) {
          if (ConstantInt *opnd1 = dyn_cast<ConstantInt>(I.getOperand(1)))
            Out << (opnd0->getSExtValue() - opnd1->getSExtValue());
          else {
            if (addParenthesis.find(&I) != addParenthesis.end())
              Out << "(";
            writeInlineOperandWithParens(I.getOperand(0));
            Out << " - ";
            writeSubOperand1(I.getOperand(1));
            if (addParenthesis.find(&I) != addParenthesis.end())
              Out << ")";
          }
        } else {
          Out << "(";
          writeInlineOperandWithParens(I.getOperand(0));
          Out << " - ";
          writeSubOperand1(I.getOperand(1));
          Out << ")";
        }
      } else if (opcode == Instruction::UDiv || opcode == Instruction::FDiv) {
        Out << "(";
        writeInlineOperandWithParens(I.getOperand(0));
        Out << " / ";
        writeOperand1WithParensForNonAssoc(opcode, I.getOperand(1));
        Out << ")";
      } else if (opcode == Instruction::SDiv) {
        // if(op0Ty->isIntegerTy(32))
        //   Out << "(int)";
        // else if(op0Ty->isIntegerTy(64))
        //   Out << "(long long)";
        // else if(op0Ty->isFloatTy())
        //   Out << "(float)";
        // else if(op0Ty->isDoubleTy())
        //   Out << "(double)";
        // else assert(0 && "SUSAN: op0Ty unimplemented cast?\n");
        writeInlineOperandWithParens(I.getOperand(0));
        Out << " / ";
        // if(op1Ty->isIntegerTy(32))
        //   Out << "(int)";
        // else if(op1Ty->isIntegerTy(64))
        //   Out << "(long long)";
        // else if(op1Ty->isFloatTy())
        //   Out << "(float)";
        // else if(op1Ty->isDoubleTy())
        //   Out << "(double)";
        // else assert(0 && "SUSAN: op1Ty unimplemented cast?\n");
        writeOperand1WithParensForNonAssoc(opcode, I.getOperand(1));
      } else if (opcode == Instruction::LShr || opcode == Instruction::AShr) {
        if (addParenthesis.find(&I) != addParenthesis.end())
          Out << "(";
        writeInlineOperandWithParens(I.getOperand(0));
        Out << " >> ";
        writeInlineOperandWithParens(I.getOperand(1));
        if (addParenthesis.find(&I) != addParenthesis.end())
          Out << ")";
      } else if (opcode == Instruction::Shl) {
        if (addParenthesis.find(&I) != addParenthesis.end())
          Out << "(";
        writeInlineOperandWithParens(I.getOperand(0));
        Out << " << ";
        writeInlineOperandWithParens(I.getOperand(1));
        if (addParenthesis.find(&I) != addParenthesis.end())
          Out << ")";
      } else if (opcode == Instruction::Xor) {
        writeInlineOperandWithParens(I.getOperand(0));
        Out << " ^ ";
        writeInlineOperandWithParens(I.getOperand(1));
      } else if (opcode == Instruction::Or) {
        writeInlineOperandWithParens(I.getOperand(0));
        Out << " | ";
        writeInlineOperandWithParens(I.getOperand(1));
      } else if (opcode == Instruction::And) {
        writeInlineOperandWithParens(I.getOperand(0));
        Out << " & ";
        writeInlineOperandWithParens(I.getOperand(1));
      } else {
        Out << "llvm_" << Instruction::getOpcodeName(opcode) << "_";
        printTypeString(Out, VTy, false);
        writeInlineOperandWithParens(I.getOperand(0));
        Out << ", ";
        writeInlineOperandWithParens(I.getOperand(1));
      }
    }
    // Out << ")";
    InlineOpDeclTypes.insert(std::pair<unsigned, Type *>(opcode, VTy));
    return;
  }

  // If this is a negation operation, print it out as such.  For FP, we don't
  // want to print "-0.0 - X".
  Value *X;
  if (match(&I, m_Neg(m_Value(X)))) {
    Out << "-(";
    writeOperand(X);
    Out << ")";
  } else if (match(&I, m_FNeg(m_Value(X)))) {
    Out << "-(";
    writeOperand(X);
    Out << ")";
  } else if (match(&I, m_Not(m_Value(X)))) {
    Out << "~(";
    writeOperand(X);
    Out << ")";
  } else if (I.getOpcode() == Instruction::FRem) {
    // Output a call to fmod/fmodf instead of emitting a%b
    if (I.getType() == Type::getFloatTy(I.getContext()))
      Out << "fmodf(";
    else if (I.getType() == Type::getDoubleTy(I.getContext()))
      Out << "fmod(";
    else // all 3 flavors of long double
      Out << "fmodl(";
    writeOperand(I.getOperand(0), ContextCasted);
    Out << ", ";
    writeOperand(I.getOperand(1), ContextCasted);
    Out << ")";
  } else {
    auto writeCastedOperandWithParens = [&](Value *V) {
      bool needsParens =
          isa<Instruction>(V) && isInlinableInst(*cast<Instruction>(V));
      if (needsParens)
        Out << "(";
      writeOperandWithCast(V, I.getOpcode());
      if (needsParens)
        Out << ")";
    };

    // Write out the cast of the instruction's value back to the proper type
    // if necessary.
    bool NeedsClosingParens = writeInstructionCast(I);

    // Certain instructions require the operand to be forced to a specific type
    // so we use writeOperandWithCast here instead of writeOperand. Similarly
    // below for operand 1
    bool foldedConstant = false;
    if (I.getOpcode() == Instruction::Sub) {
      if (ConstantInt *opnd0 = dyn_cast<ConstantInt>(I.getOperand(0)))
        if (ConstantInt *opnd1 = dyn_cast<ConstantInt>(I.getOperand(1))) {
          foldedConstant = true;
          Out << (opnd0->getSExtValue() - opnd1->getSExtValue());
        }
    }
    if (!foldedConstant && (I.getOpcode() == Instruction::Add ||
                            I.getOpcode() == Instruction::FAdd ||
                            I.getOpcode() == Instruction::Mul ||
                            I.getOpcode() == Instruction::FMul ||
                            I.getOpcode() == Instruction::SDiv ||
                            I.getOpcode() == Instruction::UDiv ||
                            I.getOpcode() == Instruction::FDiv ||
                            I.getOpcode() == Instruction::Sub ||
                            I.getOpcode() == Instruction::FSub ||
                            I.getOpcode() == Instruction::Shl ||
                            I.getOpcode() == Instruction::LShr ||
                            I.getOpcode() == Instruction::AShr))
      Out << "(";
    writeCastedOperandWithParens(I.getOperand(0));

    errs() << "SUSAN: am I here 8049?\n";
    switch (I.getOpcode()) {
    case Instruction::Add:
    case Instruction::FAdd:
      Out << " + ";
      break;
    case Instruction::Sub:
    case Instruction::FSub:
      Out << " - ";
      break;
    case Instruction::Mul:
    case Instruction::FMul:
      Out << " * ";
      break;
    case Instruction::URem:
    case Instruction::SRem:
    case Instruction::FRem:
      Out << " % ";
      break;
    case Instruction::UDiv:
    case Instruction::SDiv:
    case Instruction::FDiv:
      Out << " / ";
      break;
    case Instruction::And:
      Out << " & ";
      break;
    case Instruction::Or:
      Out << " | ";
      break;
    case Instruction::Xor:
      Out << " ^ ";
      break;
    case Instruction::Shl:
      Out << " << ";
      break;
    case Instruction::LShr:
    case Instruction::AShr:
      Out << " >> ";
      break;
    default:
      DBG_ERRS("Invalid operator type !" << I);
      errorWithMessage("invalid operator type");
    }

    writeCastedOperandWithParens(I.getOperand(1));
    if (I.getOpcode() == Instruction::Add ||
        I.getOpcode() == Instruction::FAdd ||
        I.getOpcode() == Instruction::Mul ||
        I.getOpcode() == Instruction::FMul ||
        I.getOpcode() == Instruction::SDiv ||
        I.getOpcode() == Instruction::UDiv ||
        I.getOpcode() == Instruction::FDiv ||
        I.getOpcode() == Instruction::Sub ||
        I.getOpcode() == Instruction::FSub ||
        I.getOpcode() == Instruction::Shl ||
        I.getOpcode() == Instruction::LShr ||
        I.getOpcode() == Instruction::AShr)
      Out << ")";
    if (NeedsClosingParens)
      Out << "))";
  }
}

void CWriter::visitICmpInst(ICmpInst &I) {
  CurInstr = &I;

  if (I.getType()->isVectorTy() ||
      I.getOperand(0)->getType()->getPrimitiveSizeInBits() > 64) {
    Out << "llvm_icmp_" << getCmpPredicateName(I.getPredicate()) << "_";
    printTypeString(Out, I.getOperand(0)->getType(), I.isSigned());
    Out << "(";
    writeOperand(I.getOperand(0), ContextCasted);
    Out << ", ";
    writeOperand(I.getOperand(1), ContextCasted);
    Out << ")";
    if (VectorType *VTy = dyn_cast<VectorType>(I.getOperand(0)->getType())) {
      CmpDeclTypes.insert(
          std::pair<CmpInst::Predicate, VectorType *>(I.getPredicate(), VTy));
      TypedefDeclTypes.insert(
          I.getType()); // insert type not necessarily visible above
    }
    return;
  }

  // Write out the cast of the instruction's value back to the proper type
  // if necessary.
  bool NeedsClosingParens = writeInstructionCast(I);

  // Certain icmp predicate require the operand to be forced to a specific type
  // so we use writeOperandWithCast here instead of writeOperand. Similarly
  // below for operand 1

  writeOperandWithCast(I.getOperand(0), I);
  printCmpOperator(&I);
  writeOperandWithCast(I.getOperand(1), I);
  if (NeedsClosingParens)
    Out << "))";
}

void CWriter::visitFCmpInst(FCmpInst &I) {
  CurInstr = &I;

  if (I.getType()->isVectorTy()) {
    Out << "llvm_fcmp_" << getCmpPredicateName(I.getPredicate()) << "_";
    printTypeString(Out, I.getOperand(0)->getType(), I.isSigned());
    Out << "(";
    writeOperand(I.getOperand(0), ContextCasted);
    Out << ", ";
    writeOperand(I.getOperand(1), ContextCasted);
    Out << ")";
    if (VectorType *VTy = dyn_cast<VectorType>(I.getOperand(0)->getType())) {
      CmpDeclTypes.insert(
          std::pair<CmpInst::Predicate, VectorType *>(I.getPredicate(), VTy));
      TypedefDeclTypes.insert(
          I.getType()); // insert type not necessarily visible above
    }
    return;
  }

  const auto Pred = I.getPredicate();
  headerUseFCmpOp(Pred);
  Out << "llvm_fcmp_" << getCmpPredicateName(Pred) << "(";
  // Write the first operand
  writeOperand(I.getOperand(0), ContextCasted);
  Out << ", ";
  // Write the second operand
  writeOperand(I.getOperand(1), ContextCasted);
  Out << ")";
}

static const char *getFloatBitCastField(Type *Ty) {
  switch (Ty->getTypeID()) {
  default:
    llvm_unreachable("Invalid Type");
  case Type::FloatTyID:
    return "Float";
  case Type::DoubleTyID:
    return "Double";
  case Type::IntegerTyID: {
    unsigned NumBits = cast<IntegerType>(Ty)->getBitWidth();
    if (NumBits <= 32)
      return "Int32";
    else
      return "Int64";
  }
  }
}

void CWriter::visitCastInst(CastInst &I) {
  CurInstr = &I;
  errs() << "SUSAN: visiting cast: " << I << "\n";
  Type *DstTy = I.getType();
  Type *SrcTy = I.getOperand(0)->getType();
  if (isa<TruncInst>(&I)) {
    writeOperand(I.getOperand(0), ContextCasted);
    return;
  }
  if ((isa<ZExtInst>(&I) || isa<SExtInst>(&I)) && isa<IntegerType>(DstTy) &&
      isa<IntegerType>(SrcTy)) {
    unsigned dstBits = cast<IntegerType>(DstTy)->getBitWidth();
    unsigned srcBits = cast<IntegerType>(SrcTy)->getBitWidth();

    if (srcBits <= 32 && dstBits <= 64) {
      writeOperand(I.getOperand(0), ContextCasted);
      return;
    }
  }
  if (isa<SIToFPInst>(&I)) {
    // Preserve signed integer semantics for sitofp even when the underlying C
    // variable was declared unsigned.
    Out << "((";
    printTypeName(Out, DstTy);
    Out << ")((";
    printSimpleType(Out, SrcTy, true);
    Out << ")";
    writeOperand(I.getOperand(0), ContextCasted);
    Out << "))";
    return;
  }
  // skip translating this cast if not needed
  SExtInst *sextinst = dyn_cast<SExtInst>(&I);
  if (sextinst &&
      declareAsCastedType.find(sextinst) != declareAsCastedType.end()) {
    writeOperand(I.getOperand(0));
    return;
  }

  if (DstTy->isVectorTy() || SrcTy->isVectorTy() ||
      DstTy->getPrimitiveSizeInBits() > 64 ||
      SrcTy->getPrimitiveSizeInBits() > 64) {
    Out << "llvm_" << I.getOpcodeName() << "_";
    printTypeString(Out, SrcTy, false);
    Out << "_";
    printTypeString(Out, DstTy, false);
    Out << "(";
    writeOperand(I.getOperand(0), ContextCasted);
    Out << ")";
    CastOpDeclTypes.insert(
        std::pair<Instruction::CastOps, std::pair<Type *, Type *>>(
            I.getOpcode(), std::pair<Type *, Type *>(SrcTy, DstTy)));
    return;
  }

  if (isFPIntBitCast(I)) {
    Out << '(';
    // These int<->float and long<->double casts need to be handled specially
    Out << GetValueName(&I) << "__BITCAST_TEMPORARY."
        << getFloatBitCastField(I.getOperand(0)->getType()) << " = ";
    writeOperand(I.getOperand(0), ContextCasted);
    Out << ", " << GetValueName(&I) << "__BITCAST_TEMPORARY."
        << getFloatBitCastField(I.getType());
    Out << ')';
    return;
  }

  Out << '(';

  printCast(I.getOpcode(), SrcTy, DstTy);

  // Make a sext from i1 work by subtracting the i1 from 0 (an int).
  if (SrcTy == Type::getInt1Ty(I.getContext()) &&
      I.getOpcode() == Instruction::SExt)
    Out << "0-";

  // Pointer-to-pointer bitcasts used as memory operands must preserve the
  // address of scalar GEP lvalues. Otherwise a store like
  //   store i64 %v, i64* bitcast(double* %arrayidx to i64*)
  // becomes invalid C such as "((uint64_t*)arr[idx])" instead of
  // "((uint64_t*)(&arr[idx]))".
  if (DstTy->isPointerTy() && !SrcTy->isPointerTy() &&
      SrcTy->isAggregateType()) {
    writePointerCastSource(I.getOperand(0), PointerCastGeneral,
                           ContextCasted, false);
  } else if (SrcTy->isPointerTy() && DstTy->isPointerTy()) {
    bool usedAsDirectMemoryOperand = false;
    for (const User *U : I.users()) {
      if (isa<LoadInst>(U) || isa<StoreInst>(U) || isa<AtomicRMWInst>(U) ||
          isa<AtomicCmpXchgInst>(U)) {
        usedAsDirectMemoryOperand = true;
        break;
      }
    }
    writePointerCastSource(
        I.getOperand(0),
        usedAsDirectMemoryOperand ? PointerCastDirectMemoryOperand
                                  : PointerCastGeneral,
        ContextCasted, true);
  } else {
    writeOperand(I.getOperand(0), ContextCasted);
  }

  if (DstTy == Type::getInt1Ty(I.getContext()) &&
      (I.getOpcode() == Instruction::Trunc ||
       I.getOpcode() == Instruction::FPToUI ||
       I.getOpcode() == Instruction::FPToSI ||
       I.getOpcode() == Instruction::PtrToInt)) {
    // Make sure we really get a trunc to bool by anding the operand with 1
    Out << "&1u";
  }
  Out << ')';
}

void CWriter::visitSelectInst(SelectInst &I) {
  CurInstr = &I;
  Out << "(";
  writeOperand(I.getCondition(), ContextCasted);
  Out << " ? ";
  writeOperand(I.getTrueValue(), ContextCasted);
  Out << " : ";
  writeOperand(I.getFalseValue(), ContextCasted);
  Out << ")";
  // Out << "llvm_select_";
  // printTypeString(Out, I.getType(), false);
  // Out << "(";
  // writeOperand(I.getCondition(), ContextCasted);
  // Out << ", ";
  // writeOperand(I.getTrueValue(), ContextCasted);
  // Out << ", ";
  // writeOperand(I.getFalseValue(), ContextCasted);
  // Out << ")";
  // SelectDeclTypes.insert(I.getType());
  // cwriter_assert(
  //     I.getCondition()->getType()->isVectorTy() ==
  //     I.getType()->isVectorTy()); // TODO: might be scalarty == vectorty
}

// Returns the macro name or value of the max or min of an integer type
// (as defined in limits.h).
static void printLimitValue(IntegerType &Ty, bool isSigned, bool isMax,
                            raw_ostream &Out) {
  const char *type;
  const char *sprefix = "";

  unsigned NumBits = Ty.getBitWidth();
  if (NumBits <= 8) {
    type = "CHAR";
    sprefix = "S";
  } else if (NumBits <= 16) {
    type = "SHRT";
  } else if (NumBits <= 32) {
    type = "INT";
  } else if (NumBits <= 64) {
    type = "LLONG";
  } else {
    llvm_unreachable("Bit widths > 64 not implemented yet");
  }

  if (isSigned)
    Out << sprefix << type << (isMax ? "_MAX" : "_MIN");
  else
    Out << "U" << type << (isMax ? "_MAX" : "0");
}

#ifndef NDEBUG
static bool isSupportedIntegerSize(IntegerType &T) {
  return T.getBitWidth() == 8 || T.getBitWidth() == 16 ||
         T.getBitWidth() == 32 || T.getBitWidth() == 64 ||
         T.getBitWidth() == 128;
}
#endif

void CWriter::printIntrinsicDefinition(FunctionType *funT, unsigned Opcode,
                                       std::string OpName, raw_ostream &Out) {
  Type *retT = funT->getReturnType();
  Type *elemT = funT->getParamType(0);
  IntegerType *elemIntT = dyn_cast<IntegerType>(elemT);
  char i, numParams = funT->getNumParams();
  bool isSigned;
  switch (Opcode) {
  default:
    isSigned = false;
    break;
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::smul_with_overflow:
  case Intrinsic::nvvm_mul24_i:
    isSigned = true;
    break;
  }
  cwriter_assert(numParams > 0 && numParams < 26);

  if (isa<VectorType>(retT)) {
    // this looks general, but is only actually used for ctpop, ctlz, cttz
    Type **devecFunParams = (Type **)alloca(sizeof(Type *) * numParams);
    for (i = 0; i < numParams; i++) {
      devecFunParams[(int)i] = funT->params()[(int)i]->getScalarType();
    }
    FunctionType *devecFunT = FunctionType::get(
        funT->getReturnType()->getScalarType(),
        makeArrayRef(devecFunParams, numParams), funT->isVarArg());
    printIntrinsicDefinition(devecFunT, Opcode, OpName + "_devec", Out);
  }

  // static __forceinline Rty _llvm_op_ixx(unsigned ixx a, unsigned ixx b) {
  //   Rty r;
  //   <opcode here>
  //   return r;
  // }
  Out << "static __forceinline ";
  printTypeName(Out, retT);
  Out << " ";
  Out << OpName;
  Out << "(";
  for (i = 0; i < numParams; i++) {
    switch (Opcode) {
    // optional intrinsic validity cwriter_assertion checks
    default:
      // default case: assume all parameters must have the same type
      cwriter_assert(elemT == funT->getParamType(i));
      break;
    case Intrinsic::ctlz:
    case Intrinsic::cttz:
    case Intrinsic::powi:
      break;
    }
    printTypeNameUnaligned(Out, funT->getParamType(i), isSigned);
    Out << " " << (char)('a' + i);
    if (i != numParams - 1)
      Out << ", ";
  }
  Out << ") {\n  ";
  printTypeName(Out, retT);
  Out << " r;\n";

  if (isa<VectorType>(retT)) {
    for (i = 0; i < numParams; i++) {
      Out << "  r.vector[" << (int)i << "] = " << OpName << "_devec(";
      for (char j = 0; j < numParams; j++) {
        Out << (char)('a' + j);
        if (isa<VectorType>(funT->params()[j]))
          Out << ".vector[" << (int)i << "]";
        if (j != numParams - 1)
          Out << ", ";
      }
      Out << ");\n";
    }
  } else if (Opcode == Intrinsic::nvvm_d2i_hi ||
             Opcode == Intrinsic::nvvm_d2i_lo) {
    // nvvm_d2i_{hi,lo} extract the upper/lower 32 bits of a double's bit
    // representation.
    cwriter_assert(numParams == 1);
    cwriter_assert(retT->isIntegerTy(32) && elemT->isDoubleTy());
    headerUseBitCastUnion();
    Out << "  llvmBitCastUnion bc;\n";
    Out << "  bc.Double = a;\n";
    if (Opcode == Intrinsic::nvvm_d2i_hi)
      Out << "  r = (uint32_t)(bc.Int64 >> 32);\n";
    else
      Out << "  r = (uint32_t)(bc.Int64 & UINT64_C(0xFFFFFFFF));\n";
  } else if (Opcode == Intrinsic::nvvm_lohi_i2d) {
    // nvvm_lohi_i2d assembles a double from low/high 32-bit words.
    cwriter_assert(numParams == 2);
    cwriter_assert(retT->isDoubleTy() && elemT->isIntegerTy(32) &&
                   funT->getParamType(1)->isIntegerTy(32));
    headerUseBitCastUnion();
    Out << "  llvmBitCastUnion bc;\n";
    Out << "  bc.Int64 = ((uint64_t)(uint32_t)b << 32) | (uint64_t)(uint32_t)a;\n";
    Out << "  r = bc.Double;\n";
  } else if (elemIntT) {
    // handle integer ops
    cwriter_assert(isSupportedIntegerSize(*elemIntT) &&
                   "CBackend does not support arbitrary size integers.");
    switch (Opcode) {
    default:
      DBG_ERRS("Unsupported Intrinsic!" << Opcode);
      errorWithMessage("unsupported instrinsic");

    case Intrinsic::uadd_with_overflow:
      //   r.field0 = a + b;
      //   r.field1 = (r.field0 < a);
      cwriter_assert(cast<StructType>(retT)->getElementType(0) == elemT);
      Out << "  r.field0 = a + b;\n";
      Out << "  r.field1 = (a >= -b);\n";
      break;

    case Intrinsic::sadd_with_overflow:
      headerUseLimits(); // _MAX and _MIN definitions
      //   r.field0 = a + b;
      //   r.field1 = (b > 0 && a > XX_MAX - b) ||
      //              (b < 0 && a < XX_MIN - b);
      cwriter_assert(cast<StructType>(retT)->getElementType(0) == elemT);
      Out << "  r.field0 = a + b;\n";
      Out << "  r.field1 = (b >= 0 ? a > ";
      printLimitValue(*elemIntT, true, true, Out);
      Out << " - b : a < ";
      printLimitValue(*elemIntT, true, false, Out);
      Out << " - b);\n";
      break;

    case Intrinsic::usub_with_overflow:
      cwriter_assert(cast<StructType>(retT)->getElementType(0) == elemT);
      Out << "  r.field0 = a - b;\n";
      Out << "  r.field1 = (a < b);\n";
      break;

    case Intrinsic::ssub_with_overflow:
      headerUseLimits(); // _MAX and _MIN definitions
      cwriter_assert(cast<StructType>(retT)->getElementType(0) == elemT);
      Out << "  r.field0 = a - b;\n";
      Out << "  r.field1 = (b <= 0 ? a > ";
      printLimitValue(*elemIntT, true, true, Out);
      Out << " + b : a < ";
      printLimitValue(*elemIntT, true, false, Out);
      Out << " + b);\n";
      break;

    case Intrinsic::umul_with_overflow:
      cwriter_assert(cast<StructType>(retT)->getElementType(0) == elemT);
      Out << "  r.field1 = LLVMMul_uov(8 * sizeof(a), &a, &b, &r.field0);\n";
      break;

    case Intrinsic::smul_with_overflow:
      cwriter_assert(cast<StructType>(retT)->getElementType(0) == elemT);
      Out << "  r.field1 = LLVMMul_sov(8 * sizeof(a), &a, &b, &r.field0);\n";
      break;

    case Intrinsic::nvvm_mul24_i:
      // cwriter_assert(cast<StructType>(retT)->getElementType(0) == elemT);
      Out << "r = a * b; \n";
      break;

    case Intrinsic::bswap:
      cwriter_assert(retT == elemT);
      Out << "  LLVMFlipAllBits(8 * sizeof(a), &a, &r);\n";
      break;

    case Intrinsic::ctpop:
      cwriter_assert(retT == elemT);
      Out << "  r = ";
      if (retT->getPrimitiveSizeInBits() > 64)
        Out << "llvm_ctor_u128(0, ";
      Out << "LLVMCountPopulation(8 * sizeof(a), &a)";
      if (retT->getPrimitiveSizeInBits() > 64)
        Out << ")";
      Out << ";\n";
      break;

    case Intrinsic::ctlz:
      cwriter_assert(retT == elemT);
      Out << "  (void)b;\n  r = ";
      if (retT->getPrimitiveSizeInBits() > 64)
        Out << "llvm_ctor_u128(0, ";
      Out << "LLVMCountLeadingZeros(8 * sizeof(a), &a)";
      if (retT->getPrimitiveSizeInBits() > 64)
        Out << ")";
      Out << ";\n";
      break;

    case Intrinsic::cttz:
      cwriter_assert(retT == elemT);
      Out << "  (void)b;\n  r = ";
      if (retT->getPrimitiveSizeInBits() > 64)
        Out << "llvm_ctor_u128(0, ";
      Out << "LLVMCountTrailingZeros(8 * sizeof(a), &a)";
      if (retT->getPrimitiveSizeInBits() > 64)
        Out << ")";
      Out << ";\n";
      break;
    }

  } else {
    // handle FP ops
    const char *suffix;
    cwriter_assert(retT == elemT);
    if (elemT->isFloatTy() || elemT->isHalfTy()) {
      suffix = "f";
    } else if (elemT->isDoubleTy()) {
      suffix = "";
    } else if (elemT->isFP128Ty()) {
    } else if (elemT->isX86_FP80Ty()) {
    } else if (elemT->isPPC_FP128Ty()) {
      suffix = "l";
    } else {
      DBG_ERRS("Unsupported Intrinsic!" << Opcode);
      errorWithMessage("unsupported instrinsic");
    }

    switch (Opcode) {
    default:
      DBG_ERRS("Unsupported Intrinsic!" << Opcode);
      errorWithMessage("unsupported instrinsic");

    case Intrinsic::ceil:
      headerUseMath();
      Out << "  r = ceil" << suffix << "(a);\n";
      break;

    case Intrinsic::fabs:
      headerUseMath();
      Out << "  r = fabs" << suffix << "(a);\n";
      break;

    case Intrinsic::floor:
      headerUseMath();
      Out << "  r = floor" << suffix << "(a);\n";
      break;

    case Intrinsic::fma:
    case Intrinsic::nvvm_fma_rn_d:
      headerUseMath();
      Out << "  r = fma" << suffix << "(a, b, c);\n";
      break;

    case Intrinsic::fmuladd:
      Out << "  r = a * b + c;\n";
      break;

    case Intrinsic::nvvm_mul_rn_d:
      Out << "  r = a * b;\n";
      break;
    case Intrinsic::nvvm_add_rn_d:
      Out << "  r = a + b;\n";
      break;

    case Intrinsic::pow:
    case Intrinsic::powi:
      headerUseMath();
      Out << "  r = pow" << suffix << "(a, b);\n";
      break;

    case Intrinsic::rint:
      headerUseMath();
      Out << "  r = rint" << suffix << "(a);\n";
      break;

    case Intrinsic::sqrt:
      headerUseMath();
      Out << "  r = sqrt" << suffix << "(a);\n";
      break;

    case Intrinsic::trunc:
      headerUseMath();
      Out << "  r = trunc" << suffix << "(a);\n";
      break;
    }
  }

  Out << "  return r;\n}\n";
}

void CWriter::printIntrinsicDefinition(Function &F, raw_ostream &Out) {
  FunctionType *funT = F.getFunctionType();
  unsigned Opcode = F.getIntrinsicID();
  std::string OpName = GetValueName(&F);
  printIntrinsicDefinition(funT, Opcode, OpName, Out);
}

bool CWriter::lowerIntrinsics(Function &F) {
  bool LoweredAny = false;

  // Examine all the instructions in this function to find the intrinsics that
  // need to be lowered.
  for (auto &BB : F) {
    for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E;) {
      if (CallInst *CI = dyn_cast<CallInst>(I++)) {
        if (Function *F = CI->getCalledFunction()) {
          switch (F->getIntrinsicID()) {
          case Intrinsic::not_intrinsic:
          case Intrinsic::vastart:
          case Intrinsic::vacopy:
          case Intrinsic::vaend:
          case Intrinsic::returnaddress:
          case Intrinsic::frameaddress:
// LLVM 10 doesn't have setjmp/longjmp as intrinsics.
// TODO: figure this out.
#if LLVM_VERSION_MAJOR < 10
          case Intrinsic::setjmp:
          case Intrinsic::longjmp:
          case Intrinsic::sigsetjmp:
          case Intrinsic::siglongjmp:
#endif
          case Intrinsic::prefetch:
          case Intrinsic::x86_sse_cmp_ss:
          case Intrinsic::x86_sse_cmp_ps:
          case Intrinsic::x86_sse2_cmp_sd:
          case Intrinsic::x86_sse2_cmp_pd:
          case Intrinsic::ppc_altivec_lvsl:
          case Intrinsic::uadd_with_overflow:
          case Intrinsic::sadd_with_overflow:
          case Intrinsic::usub_with_overflow:
          case Intrinsic::ssub_with_overflow:
          case Intrinsic::umul_with_overflow:
          case Intrinsic::smul_with_overflow:
          case Intrinsic::bswap:
          case Intrinsic::ceil:
          case Intrinsic::ctlz:
          case Intrinsic::ctpop:
          case Intrinsic::cttz:
          case Intrinsic::fabs:
          case Intrinsic::floor:
          case Intrinsic::fma:
          case Intrinsic::fmuladd:
          case Intrinsic::pow:
          case Intrinsic::powi:
          case Intrinsic::rint:
          case Intrinsic::sqrt:
          case Intrinsic::trunc:
          case Intrinsic::trap:
          case Intrinsic::stackprotector:
          case Intrinsic::dbg_value:
          case Intrinsic::dbg_declare:
          case Intrinsic::nvvm_mul24_i:
          case Intrinsic::nvvm_fma_rn_d:
          case Intrinsic::nvvm_add_rn_d:
          case Intrinsic::nvvm_mul_rn_d:
          case Intrinsic::nvvm_lohi_i2d:
          case Intrinsic::nvvm_d2i_hi:
          case Intrinsic::nvvm_d2i_lo:
          case Intrinsic::nvvm_barrier0:
          case Intrinsic::nvvm_fabs_d:
          case Intrinsic::nvvm_fabs_f:
            // We directly implement these intrinsics
            break;

          default:
            // All other intrinsic calls we must lower.
            LoweredAny = true;

            Instruction *Before = (CI == &BB.front())
                                      ? nullptr
                                      : &*std::prev(BasicBlock::iterator(CI));

            IL->LowerIntrinsicCall(CI);
            if (Before) { // Move iterator to instruction after call
              I = BasicBlock::iterator(Before);
              ++I;
            } else {
              I = BB.begin();
            }

            // If the intrinsic got lowered to another call, and that call has
            // a definition, then we need to make sure its prototype is emitted
            // before any calls to it.
            if (CallInst *Call = dyn_cast<CallInst>(I))
              if (Function *NewF = Call->getCalledFunction())
                if (!NewF->isDeclaration())
                  prototypesToGen.push_back(NewF);
            break;
          }
        }
      }
    }
  }

  return LoweredAny;
}

void CWriter::omp_searchForUsesToDelete(std::set<Value *> values2delete,
                                        Function &F) {
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    Instruction *inst = &*I;
    for (unsigned i = 0, e = inst->getNumOperands(); i != e; ++i) {
      Value *opnd = inst->getOperand(i);
      if (values2delete.find(opnd) != values2delete.end()) {
        omp_SkipVals.insert(cast<Value>(inst));
      }
    }
  }
}

void MallocRelatedToArgInput(Value *argInput, std::set<Value *> doubleMallocs,
                             std::map<Value *, int> &malloc2idx) {
  std::map<int, Value *> gep2Object;
  std::map<int, int> gep2typeWidth;
  std::map<int, int> gep2Align;
  for (auto U : argInput->users()) {
    GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(U);
    if (!gep)
      continue;
    ConstantInt *constint = dyn_cast<ConstantInt>(gep->getOperand(2));
    auto extractedIdx = constint->getSExtValue();
    StructType *sourceElTy = dyn_cast<StructType>(gep->getSourceElementType());
    Type *ty = sourceElTy->getElementType(extractedIdx);
    if (ty->isDoubleTy() || ty->isPointerTy())
      gep2typeWidth[extractedIdx] = 8;
    else if (IntegerType *intTy = dyn_cast<IntegerType>(ty))
      gep2typeWidth[extractedIdx] = intTy->getBitWidth() / 8;

    for (auto bitcastU : gep->users())
      if (BitCastInst *bitcast = dyn_cast<BitCastInst>(bitcastU))
        for (auto storeU : bitcast->users()) {
          if (StoreInst *store = dyn_cast<StoreInst>(storeU)) {
            gep2Object[extractedIdx] = store->getOperand(0);
            gep2Align[extractedIdx] = store->getAlignment();
          }
        }
      else if (StoreInst *store = dyn_cast<StoreInst>(bitcastU)) {
        gep2Object[extractedIdx] = store->getOperand(0);
        gep2Align[extractedIdx] = store->getAlignment();
      }
  }

  // figure out the stored location
  int currentIdx = 0;
  for (auto [index, object] : gep2Object) {
    if (currentIdx % gep2Align[index])
      currentIdx = (currentIdx / gep2Align[index] + 1) * gep2Align[index];
    for (auto mallocInst : doubleMallocs) {
      if (object == mallocInst) {
        malloc2idx[mallocInst] = currentIdx;
      }
    }
    currentIdx += gep2typeWidth[index];
  }
}

void Arg2Args(Value *arg, std::map<Value *, int> &args) {
  for (auto U : arg->users()) {
    Instruction *inst = dyn_cast<Instruction>(U);
    if (!inst)
      continue;
    if (isa<CastInst>(inst)) {
      for (auto ldU : inst->users()) {
        LoadInst *ld = dyn_cast<LoadInst>(ldU);
        if (!ld)
          continue;
        args[ld] = 0;
        errs() << "SUSAN: found load for struct 9084: 0" << *ld << "\n";
      }
    }

    if (GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(inst)) {
      ConstantInt *constint = dyn_cast<ConstantInt>(gep->getOperand(1));
      auto argidx = constint->getSExtValue();
      for (auto castU : gep->users()) {
        CastInst *cast = dyn_cast<CastInst>(castU);
        if (!cast)
          continue;
        for (auto ldU : cast->users()) {
          LoadInst *ld = dyn_cast<LoadInst>(ldU);
          if (!ld)
            continue;
          args[ld] = argidx;
          errs() << "SUSAN: argidx: " << argidx << "\n";
          errs() << "Load: " << *ld << "\n";
        }
      }
    }
  }
}

void CWriter::findMallocType(Function &F) {
  std::set<Value *> doubleMallocs;
  for (inst_iterator I = inst_begin(&F), E = inst_end(&F); I != E; ++I) {
    Instruction *inst = &*I;
    if (CallInst *CI = dyn_cast<CallInst>(inst))
      if (Function *F = CI->getCalledFunction())
        if (F->getName() == "malloc")
          for (auto user : CI->users()) {
            BitCastInst *castInst = dyn_cast<BitCastInst>(user);
            if (castInst) {
              PointerType *ptrTy = dyn_cast<PointerType>(castInst->getDestTy());
              if (ptrTy && ptrTy->getElementType()->isDoubleTy()) {
                errs() << "SUSAN: found double mallocs! \n";
                errs() << "malloc:" << *CI << "\n";
                errs() << "castInst: " << *castInst << "\n";
                doubleMallocs.insert(CI);
              }
            }
          }
  }

  for (auto [call, utask] : ompFuncs) {
    int numArgs = std::distance(utask->arg_begin(), utask->arg_end()) - 2;
    for (auto idx = 3; idx < numArgs + 3; ++idx) {
      Value *argInput = call->getArgOperand(idx);
      // Value *arg = utask->getArg(idx-1);
      // Hailong Jiang 03/23/2023
      Value *arg = utask->arg_begin() + (idx - 1);
      PointerType *ptrTy = dyn_cast<PointerType>(argInput->getType());
      if (ptrTy && isa<StructType>(ptrTy->getPointerElementType())) {
        std::map<Value *, int> malloc2idx, args;
        MallocRelatedToArgInput(argInput, doubleMallocs, malloc2idx);
        Arg2Args(arg, args);
        for (auto [malloc, idx] : malloc2idx) {
          for (auto [ld, idx2] : args) {
            if (idx2 == idx) {
              errs() << "SUSAN: tagged malloc " << *malloc
                     << " with type: " << *ld->getType();
              mallocType[malloc] = ld->getType();
            }
          }
        }
      }
    }
  }
}

bool CWriter::RunAllAnalysis(Function &F) {
  LI = &getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();
  PDT = &getAnalysis<PostDominatorTreeWrapperPass>(F).getPostDomTree();
  DT = &getAnalysis<DominatorTreeWrapperPass>(F).getDomTree();
  RI = &getAnalysis<RegionInfoPass>(F).getRegionInfo();
  SE = &getAnalysis<ScalarEvolutionWrapperPass>(F).getSE();
  AA = &getAnalysis<AAResultsWrapperPass>(F).getAAResults();
  // RI->dump();
  //  Get rid of intrinsics we can't handle.
  bool Modified = lowerIntrinsics(F);

  /*
   * OpenMP: preprosessings
   */
  LoopProfiles.clear();
  IRNaming.clear();
  argNameOverrides.clear();
  omp_declaredLocals.clear();
  // declaredLocals.clear();
  omp_liveins.clear();
  if (IS_OPENMP_FUNCTION)
    omp_preprossesing(F);
  preprocessSkippableInsts(F);
  preprocessLoopProfiles(F);
  deadBranches.clear();
  preprocessSkippableBranches(F);
  PDT->recalculate(F);
  DT->recalculate(F);
  SE = &getAnalysis<ScalarEvolutionWrapperPass>(F).getSE();
  LI = &getAnalysis<LoopInfoWrapperPass>(F).getLoopInfo();
  AA = &getAnalysis<AAResultsWrapperPass>(F).getAAResults();
  std::set<LoopProfile *> refreshedLoopProfiles;
  for (auto LP : LoopProfiles) {
    if (!LP || !LP->LoopHeader)
      continue;
    Loop *ResolvedLoop = LI->getLoopFor(LP->LoopHeader);
    if (!ResolvedLoop) {
      errs() << "ANDREW: dropping stale LoopProfile for header "
             << LP->LoopHeader->getName() << "\n";
      continue;
    }
    LP->L = ResolvedLoop;
    LP->LoopHeader = ResolvedLoop->getHeader();
    refreshedLoopProfiles.insert(LP);
  }
  LoopProfiles.swap(refreshedLoopProfiles);
  // SUSAN: determine whether the function can be compiled without gotos
  std::set<BasicBlock *> visitedBBs;
  markIfBranches(F, &visitedBBs); // 2
  markBackEdges(F);
  determineControlFlowTranslationMethod(F);

  for (auto LP : LoopProfiles) {
    if (LP->isOmpLoop)
      continue;
    LP->IV = getInductionVariable(LP->L);
    LP->IVInc = getIVIncrement(LP->L, LP->IV);
    LP->incr = LP->IVInc;
  }
  // SUSAN: preprocessings
  // 1. mark all the irregular exits of a loop (break/return)
  // 2. find all the branches that can be expressed as if statement before split
  // 3. node splitting on irregular graph
  // 4. identify branches that can be expressed as if statement after split
  // 5. mark each basicblock its number of times to be printed

  markLoopIrregularExits(F); // 1
  markGotoBranches(F);
  preprossesPHIs2Print(F);
  // NodeSplitting(F); PDT->recalculate(F); //3
  // markIfBranches(F, &visitedBBs); //4
  collectNoneArrayGEPs(F);
  collectVariables2Deref(F);
  collectLateDeclares(F);

  EliminateDeadInsts(F);
  FindInductionVariableRelationships();
  preprocessIVIncrements();
  preprocessInsts2AddParenthesis(F);
  buildIVNames();
  collectNotInlinableBinOps(F);
  findMallocType(F);

  return Modified;
}

Value *CWriter::getKernelDim(CallInst *CI) {
  auto *F = CI->getCalledFunction();
  for (auto &BB : *F) {
    for (auto &I : BB) {
      if (I.getMetadata("tulip.cuda.indvar")) {
        errs() << "YEBIN: FOUND TULIP INDVAR\n";
        if (auto *branch = dyn_cast<BranchInst>(BB.getTerminator())) {
          errs() << "FOUND BRANCH " << *branch << "\n";
          if (auto *cond = dyn_cast<CmpInst>(branch->getCondition())) {
            errs() << "FOUND COND " << *cond << "\n";
            Value *limit;
            if (cond->getOperand(0) == &I)
              limit = cond->getOperand(1);
            else
              limit = cond->getOperand(0);

            errs() << "NUM OPS: " << F->getNumOperands() << "\n";
            for (auto &arg : F->args()) {
              errs() << "YEBIN OPERAND: " << arg << "\n";
              int idx = std::distance(F->arg_begin(), &arg);
              if (&arg == limit)
                return CI->getOperand(idx);
            }
          }
        }
      }
    }
  }
  return nullptr;
}

void CWriter::runAnalysisOnKernelCaller(Function &F) {
  assert(F.getMetadata("tulip.cuda.kernel.caller") &&
         "Function does not call CUDA kernel(s)!");
  DevVarDecls.clear();
  KernelCallDims.clear();
  LiveOuts.clear();
  MergedHostDeviceMode = false;
  std::map<CallInst *, std::map<Value *, Value *>> CallArgsMap;

  for (auto &BB : F) {
    for (auto &I : BB) {
      if (auto *CallI = dyn_cast<CallInst>(&I))
        if (CallI->getCalledFunction()->getMetadata("tulip.cuda.kernel")) {
          int num = CallI->getCalledFunction()->getName().back() - '0';
          auto *iter = getKernelDim(CallI);
          assert(iter && "Unable to find kernel limit!\n");
          KernelCallDims[CallI] =
              std::make_pair(num, std::make_pair("threadsPerBlock", iter));
          unsigned i = 0;
          for (auto &arg : CallI->getCalledFunction()->args()) {
            auto *op = CallI->getArgOperand(i++);
            // only need device vars for those passed by reference
            // TODO: assumes no casting
            if (Times2Dereference[op]) {
              DevVarDecls["dev_" + GetValueName(op)] = op;
              CallArgsMap[CallI][&arg] = op;
            }
          }
        }
    }
  }
  for (auto call : KernelCallDims) {
    auto *F = call.first->getCalledFunction();

    for (auto &BB : *F) {
      for (auto &I : BB) {
        if (auto *store = dyn_cast<StoreInst>(&I)) {
          auto Ptr = store->getPointerOperand();
          while (auto GEP = dyn_cast<GetElementPtrInst>(Ptr)) {
            Ptr = GEP->getPointerOperand();
          }
          if (auto *op = CallArgsMap[call.first][Ptr])
            // TODO: insert the call inst argument
            LiveOuts.insert(op);
        }
      }
    }
  }
  for (auto call : KernelCallDims) {
    auto name = call.first->getCalledFunction()->getName();
    if (name.contains("cudakernel0")) {
      auto &C = call.first->getContext();
      MDNode *MD = MDNode::get(C, MDString::get(C, ""));
      call.first->setMetadata("tulip.kernel.region.begin", MD);
    }
    if (name.contains("cudakernel" +
                      std::to_string(KernelCallDims.size() - 1))) {
      auto &C = call.first->getContext();
      MDNode *MD = MDNode::get(C, MDString::get(C, ""));
      call.first->setMetadata("tulip.kernel.region.end", MD);
    }
  }
  return;
}

void CWriter::omp_findInlinedStructInputs(Value *argInput,
                                          std::map<int, Value *> &argInputs) {
  std::map<int, Value *> gep2argInput;
  std::map<int, int> gep2typeWidth;
  std::map<int, int> gep2Align;
  for (auto U : argInput->users()) {
    GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(U);
    BitCastInst *bitcastInst = dyn_cast<BitCastInst>(U);
    Instruction *userInst = nullptr;
    int idx = 0;
    Type *ty = nullptr;
    if (gep) {
      userInst = gep;
      ConstantInt *constint = dyn_cast<ConstantInt>(gep->getOperand(2));
      idx = constint->getSExtValue();
      StructType *sourceElTy =
          dyn_cast<StructType>(gep->getSourceElementType());
      ty = sourceElTy->getElementType(idx);
    } else if (bitcastInst) {
      userInst = bitcastInst;
      int idx = 0;
      PointerType *sourcePtrTy = dyn_cast<PointerType>(bitcastInst->getSrcTy());
      StructType *sourceTy =
          dyn_cast<StructType>(sourcePtrTy->getPointerElementType());
      ty = sourceTy->getElementType(idx);
    } else
      continue;

    if (ty->isDoubleTy() || ty->isPointerTy())
      gep2typeWidth[idx] = 8;
    else if (IntegerType *intTy = dyn_cast<IntegerType>(ty))
      gep2typeWidth[idx] = intTy->getBitWidth() / 8;

    for (auto storeU : userInst->users()) {
      if (StoreInst *store = dyn_cast<StoreInst>(storeU)) {
        errs() << "SUSAN: found store for struct 9066: " << *store << "\n";
        auto originalVal = findOriginalValue(store->getOperand(0));
        errs() << "SUSAN: original Val: " << *originalVal;
        gep2argInput[idx] = originalVal;
        errs() << "SUSAN: alignment: " << store->getAlignment();
        gep2Align[idx] = store->getAlignment();
      } else if (CastInst *cast = dyn_cast<CastInst>(storeU)) {
        for (auto storeU : cast->users()) {
          StoreInst *store = dyn_cast<StoreInst>(storeU);
          if (!store)
            continue;
          errs() << "SUSAN: found store for struct 9095: " << *store << "\n";
          gep2argInput[idx] = store->getOperand(0);
          gep2Align[idx] = store->getAlignment();
          errs() << "SUSAN: aligment: " << store->getAlignment() << "\n";
        }
      }
    }
  }

  // figure out the stored location
  int smallestIdx = 999;
  for (auto [idx, argInput] : gep2argInput) {
    if (smallestIdx > idx)
      smallestIdx = idx;
  }
  int currentIdx = smallestIdx;
  for (auto [idx, argInput] : gep2argInput) {
    errs() << "SUSAN: idx: " << idx << "\n";
    if (currentIdx % gep2Align[idx])
      currentIdx = (currentIdx / gep2Align[idx] + 1) * gep2Align[idx];

    errs() << "SUSAN: currIdx 9609: " << currentIdx << "\n";
    argInputs[currentIdx] = argInput;

    currentIdx += gep2typeWidth[idx];
  }
}

void CWriter::omp_findCorrespondingUsesOfStruct(Value *arg,
                                                std::map<int, Value *> &args) {
  errs() << "SUSAN: trying to find corresponding uses: " << *arg << "\n";
  for (auto U : arg->users()) {
    Instruction *inst = dyn_cast<Instruction>(U);
    if (!inst)
      continue;
    if (isa<CastInst>(inst)) {
      for (auto ldU : inst->users()) {
        LoadInst *ld = dyn_cast<LoadInst>(ldU);
        if (!ld)
          continue;
        args[0] = ld;
        errs() << "SUSAN: found load for struct 9084: 0" << *ld << "\n";
      }
    }
    if (GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(inst)) {
      ConstantInt *constint = dyn_cast<ConstantInt>(gep->getOperand(1));
      auto argidx = constint->getSExtValue();
      for (auto castU : gep->users()) {
        CastInst *cast = dyn_cast<CastInst>(castU);
        if (!cast)
          continue;
        for (auto ldU : cast->users()) {
          LoadInst *ld = dyn_cast<LoadInst>(ldU);

          if (!ld)
            continue;
          args[argidx] = ld;
          errs() << "SUSAN: argidx: " << argidx << "\n";
          errs() << "Load: " << *ld << "\n";

          if (PointerType *ptrTy = dyn_cast<PointerType>(cast->getDestTy())) {
            if (PointerType *elPtrTy =
                    dyn_cast<PointerType>(ptrTy->getPointerElementType())) {
              if (elPtrTy->getPointerElementType()->isDoubleTy())
                valuesCast2Double.insert(ld);
            }
          }

          for (auto ldValU : ld->users()) {
            LoadInst *ldVal = dyn_cast<LoadInst>(ldValU);
            if (!ldVal)
              continue;
            addressExposedLoads.insert(ldVal);
          }
        }
      }
    }
  }
}

void CWriter::inlineNameForArg(Value *argInput, Value *arg) {
  auto argName = GetValueName(argInput);

  if (ConstantInt *constant = dyn_cast<ConstantInt>(argInput)) {
    inlinedArgNames[arg] = std::to_string(constant->getSExtValue());
  } else if (ConstantFP *constant = dyn_cast<ConstantFP>(argInput)) {
    inlinedArgNames[arg] = ftostr(constant->getValueAPF());
  } else if (declaredLocals.find(argName) == declaredLocals.end()) {
    if (Instruction *argInputInst = dyn_cast<Instruction>(argInput)) {
      if (!isa<CastInst>(argInputInst)) {
        errs() << "SUSAN: inlining argInput: " << *argInput << "\n";
        errs() << "SUSAN: inlining arg: " << *arg << "\n";
        printTypeName(Out, argInputInst->getType(), false) << ' ';
        Out << GetValueName(argInputInst, true) << " = ";
        writeInstComputationInline(*argInputInst);
        Out << ";\n";
      }
    }
    inlinedArgNames[arg] = argName;
  } else {
    inlinedArgNames[arg] = argName;
  }
}

void CWriter::visitCallInst(CallInst &I) {
  CurInstr = &I;

  // skip barrier
  if (Function *F = I.getCalledFunction())
    if (F->getName() == "__kmpc_barrier")
      return;

  bool hasMergedSafeDecision = false;
  bool mergedSafeDecision = false;
  if (MDNode *mergedSafeMD = I.getMetadata("tulip.cudamemcpy.merged_safe")) {
    hasMergedSafeDecision = true;
    bool mergedSafe = false;
    if (mergedSafeMD->getNumOperands() > 0) {
      if (auto *S = dyn_cast_or_null<MDString>(mergedSafeMD->getOperand(0))) {
        StringRef V = S->getString();
        mergedSafe = (V == "true" || V == "1");
      } else {
        // Non-string payloads are treated as affirmative for compatibility.
        mergedSafe = true;
      }
    }
    mergedSafeDecision = mergedSafe;
    if (mergedSafe)
      MergedHostDeviceMode = true;
  }
  if (!hasMergedSafeDecision &&
      (I.getMetadata("tulip.target.mapdata.to") ||
       I.getMetadata("tulip.target.mapdata.from") ||
       I.getMetadata("tulip.cudamemcpy.direction") ||
       I.getMetadata("tulip.cudamemcpy.size") ||
       I.getMetadata("tulip.cudamemcpy.original") ||
       I.getMetadata("tulip.cudamemcpy.device"))) {
    MergedHostDeviceMode = true;
  }

  // Runtime memcpy calls are expected to be lowered/erased in Tulip passes.
  // Only suppress emission when merge pass marked this call safe to drop.
  if (Function *F = I.getCalledFunction())
    if (F->getName().contains("cudaMemcpy"))
      if (hasMergedSafeDecision && mergedSafeDecision)
        return;

  if (I.getMetadata("tulip.kernel.region.begin")) {
    if (!MergedHostDeviceMode) {
      for (auto devVar : DevVarDecls) {
        Out << *devVar.second->getType() << " " << devVar.first << ";\n";
      }
      Out << "\n";
      // FIXME: make sure deref is correct
      // FIXME: assuming type is not safe...
      // TODO: how to get size across functions?
      for (auto devVar : DevVarDecls) {
        auto *ty = devVar.second->getType();
        if (auto ptrTy = dyn_cast<PointerType>(ty))
          ty = ptrTy->getPointerElementType();
        Out << "cudaMalloc(&" << devVar.first << ", " << "size_"
            << GetValueName(devVar.second) << "*sizeof(" << *ty << "));\n";
      }
      Out << "\n";
      for (auto devVar : DevVarDecls) {
        auto *ty = devVar.second->getType();
        if (auto ptrTy = dyn_cast<PointerType>(ty))
          ty = ptrTy->getPointerElementType();
        Out << "cudaMemcpy(" << devVar.first << ", "
            << GetValueName(devVar.second) << ", "
            << "size_" << GetValueName(devVar.second) << "*sizeof(" << *ty
            << "), " << "cudaMemcpyHostToDevice);\n";
      }
      Out << "\n";
    }
  }

  /*
   * OpenMP: skip omp runtime call
   */
  if (ompFuncs.find(&I) != ompFuncs.end()) {
    Out << "//START OUTLINED\n";
    Out << "  #pragma omp parallel \n" << "{\n";
    // Create a Call to omp_outlined
    auto utask = ompFuncs[&I];

    inlinedArgNames.clear();
    // build arg->argInput table
    int numArgs = std::distance(utask->arg_begin(), utask->arg_end()) - 2;
    for (auto idx = 3; idx < numArgs + 3; ++idx) {
      Value *argInput = I.getArgOperand(idx);
      // Value *arg = utask->getArg(idx-1);
      // Hailong Jiang 03/23/2023
      Value *arg = utask->arg_begin() + (idx - 1);
      errs() << "SUSAN: argInput: " << *argInput << "\n";
      errs() << "SUSAN: arg: " << *arg << "\n";

      if (isAddressExposed(argInput)) {
        for (inst_iterator I = inst_begin(utask), E = inst_end(utask); I != E;
             ++I) {
          if (!isa<LoadInst>(&*I))
            continue;
          LoadInst *ldInst = cast<LoadInst>(&*I);
          if (ldInst->getPointerOperand() == arg)
            addressExposedLoads.insert(ldInst);
        }
      }

      // unroll structs
      PointerType *ptrTy = dyn_cast<PointerType>(argInput->getType());
      if (ptrTy && isa<StructType>(ptrTy->getPointerElementType())) {
        std::map<int, Value *> argInputs, args;
        omp_findInlinedStructInputs(argInput, argInputs);
        omp_findCorrespondingUsesOfStruct(arg, args);
        for (auto [idx, arg] : args) {
          auto argInput = argInputs[idx];
          if (!argInput) {
            errs() << "SUSAN: didn't find argInput for idx: " << idx << "\n";
            continue;
          }
          PointerType *ptrTy = dyn_cast<PointerType>(argInput->getType());
          if (!ptrTy)
            valuesCast2Double.erase(arg);
          if (ptrTy && ptrTy->getPointerElementType()->isDoubleTy() &&
              valuesCast2Double.find(arg) != valuesCast2Double.end())
            valuesCast2Double.erase(arg);

          inlineNameForArg(argInput, arg);
        }
      } else {
        if (auto alloca = isDirectAlloca(argInput))
          for (auto user : alloca->users())
            if (StoreInst *store = dyn_cast<StoreInst>(user))
              if (store->getPointerOperand() == alloca)
                argInput = store->getOperand(0);
        errs() << "SUSAN: argInput updated:" << *argInput << "\n";
        inlineNameForArg(argInput, arg);
      }
    }

    /*Out << "  " << GetValueName(utask) << "(";

    int numArgs = std::distance(utask->arg_begin(), utask->arg_end()) - 2;
    bool printComma = false;
    for(auto idx = 3; idx < 3+numArgs; ++idx) {
      if (printComma)
        Out << ", ";
      Value *arg = I.getArgOperand(idx);
      writeOperand(arg, ContextCasted);
      printComma = true;
    }
    Out << ");\n";
    */
    // directly inline omp_outlined function
    // 1. save all the data that current function has
    auto IS_OPENMP_FUNCTION_SAVE = IS_OPENMP_FUNCTION;
    auto LoopProfiles_s = LoopProfiles;
    auto omp_liveins_s = omp_liveins;
    auto omp_SkipVals_s = omp_SkipVals;
    auto deleteAndReplaceInsts_s = deleteAndReplaceInsts;
    auto deadBranches_s = deadBranches;
    auto ifBranches_s = ifBranches;
    auto backEdges_s = backEdges;
    auto topRegion_s = topRegion;
    auto times2bePrinted_s = times2bePrinted;
    auto returnDominated_s = returnDominated;
    auto irregularLoopExits_s = irregularLoopExits;
    auto gotoBranches_s = gotoBranches;
    auto PHIValues2Print_s = PHIValues2Print;
    auto InstsToReplaceByPhi_s = InstsToReplaceByPhi;
    auto NoneArrayGEPs_s = NoneArrayGEPs;
    auto Times2Dereference_s = Times2Dereference;
    auto deadInsts_s = deadInsts;
    auto IVMap_s = IVMap;
    auto addParenthesis_s = addParenthesis;
    auto phiVars_s = phiVars;
    auto allVars_s = allVars;
    auto accessGEPMemory_s = accessGEPMemory;
    auto GEPPointers_s = GEPPointers;
    auto currValue2DerefCnt_s = currValue2DerefCnt;
    auto printLabels_s = printLabels;
    auto loopCondCalls_s = loopCondCalls;
    auto CBERegionMap_s = CBERegionMap;
    auto recordedRegionBBs_s = recordedRegionBBs;
    auto gepStart_s = gepStart;
    auto NATURAL_CONTROL_FLOW_S = NATURAL_CONTROL_FLOW;
    auto signedInsts_s = signedInsts;
    auto declareAsCastedType_s = declareAsCastedType;
    auto ompFuncs_s = ompFuncs;
    auto GEPNeedsReference_s = GEPNeedsReference;
    auto omp_declarePrivate_s = omp_declarePrivate;
    auto IVInc2IV_s = IVInc2IV;
    auto UpperBoundArgs_s = UpperBoundArgs;
    auto IRNaming_s = IRNaming;
    auto CurLoop_s = CurLoop;
    auto CurInstr_s = CurInstr;
    // auto toDeclareLocal_s = toDeclareLocal;

    // inline the function
    IS_OPENMP_FUNCTION = true;
    RunAllAnalysis(*utask);
    findDoubleGEP(*utask);
    // Output all floating point constants that cannot be printed accurately.
    printFloatingPointConstants(*utask);
    printFunction(*utask, true);

    IS_OPENMP_FUNCTION = IS_OPENMP_FUNCTION_SAVE;
    LoopProfiles = LoopProfiles_s;
    omp_liveins = omp_liveins_s;
    omp_SkipVals = omp_SkipVals_s;
    deleteAndReplaceInsts = deleteAndReplaceInsts_s;
    deadBranches = deadBranches_s;
    ifBranches = ifBranches_s;
    backEdges = backEdges_s;
    topRegion = topRegion_s;
    times2bePrinted = times2bePrinted_s;
    returnDominated = returnDominated_s;
    irregularLoopExits = irregularLoopExits_s;
    gotoBranches = gotoBranches_s;
    PHIValues2Print = PHIValues2Print_s;
    InstsToReplaceByPhi = InstsToReplaceByPhi_s;
    NoneArrayGEPs = NoneArrayGEPs_s;
    Times2Dereference = Times2Dereference_s;
    deadInsts = deadInsts_s;
    IVMap = IVMap_s;
    addParenthesis = addParenthesis_s;
    phiVars = phiVars_s;
    allVars = allVars_s;
    accessGEPMemory = accessGEPMemory_s;
    GEPPointers = GEPPointers_s;
    currValue2DerefCnt = currValue2DerefCnt_s;
    printLabels = printLabels_s;
    loopCondCalls = loopCondCalls_s;
    CBERegionMap = CBERegionMap_s;
    recordedRegionBBs = recordedRegionBBs_s;
    gepStart = gepStart_s;
    NATURAL_CONTROL_FLOW = NATURAL_CONTROL_FLOW_S;
    signedInsts = signedInsts_s;
    declareAsCastedType = declareAsCastedType_s;
    ompFuncs = ompFuncs_s;
    GEPNeedsReference = GEPNeedsReference_s;
    omp_declarePrivate = omp_declarePrivate_s;
    IVInc2IV = IVInc2IV_s;
    UpperBoundArgs = UpperBoundArgs_s;
    IRNaming = IRNaming_s;
    CurLoop = CurLoop_s;
    CurInstr = CurInstr_s;
    Function *F = I.getParent()->getParent();
    LI = &getAnalysis<LoopInfoWrapperPass>(*F).getLoopInfo();
    PDT = &getAnalysis<PostDominatorTreeWrapperPass>(*F).getPostDomTree();
    DT = &getAnalysis<DominatorTreeWrapperPass>(*F).getDomTree();
    RI = &getAnalysis<RegionInfoPass>(*F).getRegionInfo();
    SE = &getAnalysis<ScalarEvolutionWrapperPass>(*F).getSE();
    AA = &getAnalysis<AAResultsWrapperPass>(*F).getAAResults();
    // toDeclareLocal = toDeclareLocal_s;

    Out << "}\n";
    Out << "//END OUTLINED\n";
    return;
  }

  if (isa<InlineAsm>(I.getCalledOperand()))
    return visitInlineAsm(I);

  // Handle intrinsic function calls first...
  if (Function *F = I.getCalledFunction()) {
    auto ID = F->getIntrinsicID();
    if (ID != Intrinsic::not_intrinsic && visitBuiltinCall(I, ID))
      return;
  }

  Value *Callee = I.getCalledOperand();

  PointerType *PTy = cast<PointerType>(Callee->getType());
  FunctionType *FTy = cast<FunctionType>(PTy->getElementType());

  // If this is a call to a struct-return function, assign to the first
  // parameter instead of passing it to the call.
  const AttributeList &PAL = I.getAttributes();
  bool hasByVal = I.hasByValArgument();
  bool isStructRet = I.hasStructRetAttr();
  if (isStructRet) {
    writeOperandDeref(I.getArgOperand(0));
    Out << " = ";
  }

  // if (I.isTailCall())
  //   Out << " /*tail*/ ";

  // If this is an indirect call to a struct return function, we need to cast
  // the pointer. Ditto for indirect calls with byval arguments.
  bool NeedsCast =
      (hasByVal || isStructRet || I.getCallingConv() != CallingConv::C) &&
      !isa<Function>(Callee);

  // GCC is a real PITA.  It does not permit codegening casts of functions to
  // function pointers if they are in a call (it generates a trap instruction
  // instead!).  We work around this by inserting a cast to void* in between
  // the function and the function pointer cast.  Unfortunately, we can't just
  // form the constant expression here, because the folder will immediately
  // nuke it.
  //
  // Note finally, that this is completely unsafe.  ANSI C does not guarantee
  // that void* and function pointers have the same size. :( To deal with this
  // in the common case, we handle casts where the number of arguments passed
  // match exactly.
  if (ConstantExpr *CE = dyn_cast<ConstantExpr>(Callee))
    if (CE->isCast())
      if (Function *RF = dyn_cast<Function>(CE->getOperand(0))) {
        NeedsCast = true;
        Callee = RF;
      }

  if (NeedsCast) {
    // Ok, just cast the pointer type.
    Out << "((";
    printTypeName(Out, I.getCalledOperand()->getType()->getPointerElementType(),
                  false, std::make_pair(PAL, I.getCallingConv()));
    Out << "*)(void*)";
  }
  // This is where Callee name is printed
  writeOperand(Callee, ContextCasted);
  if(Callee->getName() == "fprintf") errs() << "YEBIN: FPRINTF\n";
  if (NeedsCast)
    Out << ')';

  // YEBIN: add grid for cuda kernels
  // TODO: add grid/block to IR and use those
  // TODO: add dim3 before function delcaration
  Function *F = I.getCalledFunction();
  if (F && F->getName().contains("cudakernel")) {
    std::string num = std::to_string(KernelCallDims[&I].first);
    Out << "<<<grid" << num << ", block" << num << ">>>";
  }

  Out << '(';

  bool PrintedArg = false;
  if (FTy->isVarArg() && !FTy->getNumParams()) {
    Out << "0 /*dummy arg*/";
    PrintedArg = true;
  }

  unsigned NumDeclaredParams = FTy->getNumParams();
  auto CS(&I);
  auto AI = CS->arg_begin(), AE = CS->arg_end();
  unsigned ArgNo = 0;
  if (isStructRet) { // Skip struct return argument.
    ++AI;
    ++ArgNo;
  }

  if (F) {
    StringRef Name = F->getName();
    // emit cast for the first argument to type expected by header prototype
    // the jmp_buf type is an array, so the array-to-pointer decay adds the
    // strange extra *'s
    if (Name == "sigsetjmp")
      Out << "*(sigjmp_buf*)";
    else if (Name == "setjmp")
      Out << "*(jmp_buf*)";
  }

  for (; AI != AE; ++AI, ++ArgNo) {
    if (PrintedArg)
      Out << ", ";
    if (ArgNo < NumDeclaredParams &&
        (*AI)->getType() != FTy->getParamType(ArgNo)) { // type casting
      Out << '(';
      printTypeNameUnaligned(
          Out, FTy->getParamType(ArgNo),
          /*isSigned=*/PAL.hasAttribute(ArgNo + 1, Attribute::SExt));
      Out << ')';
    }
    // Check if the argument is expected to be passed by value.
    if (I.getAttributes().hasAttribute(ArgNo + 1, Attribute::ByVal))
      writeOperandDeref(*AI);
    else
      writeOperand(*AI, ContextCasted);
    PrintedArg = true;
  }
  if (isEmptyType(I.getType()))
    Out << ");\n";
  else
    Out << ")";

  if (I.getMetadata("tulip.kernel.region.end") && !MergedHostDeviceMode) {
    Out << "\n";
    for (auto devVar : DevVarDecls) {
      // FIXME: use writeOperand instead of GetValueName directly
      if (LiveOuts.count(devVar.second)) {
        auto *ty = devVar.second->getType();
        if (auto ptrTy = dyn_cast<PointerType>(ty))
          ty = ptrTy->getPointerElementType();
        Out << "cudaMemcpy(" << GetValueName(devVar.second) << ", "
            << devVar.first << ", "
            << "size_" << GetValueName(devVar.second) << "*sizeof(" << *ty
            << "), " << "cudaMemcpyDeviceToHost);\n";
      }
    }
  }
}

/// visitBuiltinCall - Handle the call to the specified builtin.  Returns true
/// if the entire call is handled, return false if it wasn't handled
bool CWriter::visitBuiltinCall(CallInst &I, Intrinsic::ID ID) {
  CurInstr = &I;

  switch (ID) {
  default: {
    DBG_ERRS("Unknown LLVM intrinsic! " << I);
    errorWithMessage("unknown llvm instrinsic");
    return false;
  }
  case Intrinsic::nvvm_barrier0:
    Out << "#pragma omp barrier\n";
    return true;
  case Intrinsic::dbg_value:
  case Intrinsic::dbg_declare:
    return true; // ignore these intrinsics
  case Intrinsic::vastart:
    headerUseStdarg();
    Out << "0; ";

    Out << "va_start(*(va_list*)";
    writeOperand(I.getArgOperand(0), ContextCasted);
    Out << ", ";
    // Output the last argument to the enclosing function.
    if (I.getParent()->getParent()->arg_empty())
      Out << "vararg_dummy_arg";
    else {
      Function::arg_iterator arg_end = I.getParent()->getParent()->arg_end();
      writeOperand(--arg_end);
    }
    Out << ')';
    return true;
  case Intrinsic::vaend:
    headerUseStdarg();
    if (!isa<ConstantPointerNull>(I.getArgOperand(0))) {
      Out << "0; va_end(*(va_list*)";
      writeOperand(I.getArgOperand(0), ContextCasted);
      Out << ')';
    } else {
      Out << "va_end(*(va_list*)0)";
    }
    return true;
  case Intrinsic::vacopy:
    headerUseStdarg();
    Out << "0; ";
    Out << "va_copy(*(va_list*)";
    writeOperand(I.getArgOperand(0), ContextCasted);
    Out << ", *(va_list*)";
    writeOperand(I.getArgOperand(1), ContextCasted);
    Out << ')';
    return true;
  case Intrinsic::returnaddress:
    Out << "__builtin_return_address(";
    writeOperand(I.getArgOperand(0), ContextCasted);
    Out << ')';
    return true;
  case Intrinsic::frameaddress:
    Out << "__builtin_frame_address(";
    writeOperand(I.getArgOperand(0), ContextCasted);
    Out << ')';
    return true;
// LLVM 10 doesn't have setjmp/longjmp as intrinsics.
// TODO: figure this out.
#if LLVM_VERSION_MAJOR < 10
  case Intrinsic::setjmp:
    headerUseSetjmp();
    Out << "setjmp(*(jmp_buf*)";
    writeOperand(I.getArgOperand(0), ContextCasted);
    Out << ')';
    return true;
  case Intrinsic::longjmp:
    headerUseSetjmp();
    Out << "longjmp(*(jmp_buf*)";
    writeOperand(I.getArgOperand(0), ContextCasted);
    Out << ", ";
    writeOperand(I.getArgOperand(1), ContextCasted);
    Out << ')';
    return true;
  case Intrinsic::sigsetjmp:
    headerUseSetjmp();
    Out << "sigsetjmp(*(sigjmp_buf*)";
    writeOperand(I.getArgOperand(0), ContextCasted);
    Out << ',';
    writeOperand(I.getArgOperand(1), ContextCasted);
    Out << ')';
    return true;
  case Intrinsic::siglongjmp:
    headerUseSetjmp();
    Out << "siglongjmp(*(sigjmp_buf*)";
    writeOperand(I.getArgOperand(0), ContextCasted);
    Out << ", ";
    writeOperand(I.getArgOperand(1), ContextCasted);
    Out << ')';
    return true;
#endif
  case Intrinsic::prefetch:
    Out << "LLVM_PREFETCH((const void *)";
    writeOperand(I.getArgOperand(0), ContextCasted);
    Out << ", ";
    writeOperand(I.getArgOperand(1), ContextCasted);
    Out << ", ";
    writeOperand(I.getArgOperand(2), ContextCasted);
    Out << ")";
    return true;
  case Intrinsic::stacksave:
    // Emit this as: Val = 0; *((void**)&Val) = __builtin_stack_save()
    // to work around GCC bugs (see PR1809).
    headerUseStackSaveRestore();
    Out << "0; *((void**)&" << GetValueName(&I) << ") = __builtin_stack_save()";
    return true;
  case Intrinsic::x86_sse_cmp_ss:
  case Intrinsic::x86_sse_cmp_ps:
  case Intrinsic::x86_sse2_cmp_sd:
  case Intrinsic::x86_sse2_cmp_pd:
    Out << '(';
    printTypeName(Out, I.getType());
    Out << ')';
    // Multiple GCC builtins multiplex onto this intrinsic.
    switch (cast<ConstantInt>(I.getArgOperand(2))->getZExtValue()) {
    default:
      errorWithMessage("Invalid llvm.x86.sse.cmp!");
    case 0:
      Out << "__builtin_ia32_cmpeq";
      break;
    case 1:
      Out << "__builtin_ia32_cmplt";
      break;
    case 2:
      Out << "__builtin_ia32_cmple";
      break;
    case 3:
      Out << "__builtin_ia32_cmpunord";
      break;
    case 4:
      Out << "__builtin_ia32_cmpneq";
      break;
    case 5:
      Out << "__builtin_ia32_cmpnlt";
      break;
    case 6:
      Out << "__builtin_ia32_cmpnle";
      break;
    case 7:
      Out << "__builtin_ia32_cmpord";
      break;
    }
    if (ID == Intrinsic::x86_sse_cmp_ps || ID == Intrinsic::x86_sse2_cmp_pd)
      Out << 'p';
    else
      Out << 's';
    if (ID == Intrinsic::x86_sse_cmp_ss || ID == Intrinsic::x86_sse_cmp_ps)
      Out << 's';
    else
      Out << 'd';

    Out << "(";
    writeOperand(I.getArgOperand(0), ContextCasted);
    Out << ", ";
    writeOperand(I.getArgOperand(1), ContextCasted);
    Out << ")";
    return true;
  case Intrinsic::ppc_altivec_lvsl:
    Out << '(';
    printTypeName(Out, I.getType());
    Out << ')';
    Out << "__builtin_altivec_lvsl(0, (void*)";
    writeOperand(I.getArgOperand(0), ContextCasted);
    Out << ")";
    return true;
  case Intrinsic::stackprotector:
    writeOperandDeref(I.getArgOperand(1));
    Out << " = ";
    writeOperand(I.getArgOperand(0), ContextCasted);
    return true;
  case Intrinsic::uadd_with_overflow:
  case Intrinsic::sadd_with_overflow:
  case Intrinsic::usub_with_overflow:
  case Intrinsic::ssub_with_overflow:
  case Intrinsic::umul_with_overflow:
  case Intrinsic::smul_with_overflow:
  case Intrinsic::bswap:
  case Intrinsic::ceil:
  case Intrinsic::ctlz:
  case Intrinsic::ctpop:
  case Intrinsic::cttz:
  case Intrinsic::fabs:
  case Intrinsic::floor:
  case Intrinsic::fma:
  case Intrinsic::fmuladd:
  case Intrinsic::pow:
  case Intrinsic::powi:
  case Intrinsic::rint:
  case Intrinsic::sqrt:
  case Intrinsic::trap:
  case Intrinsic::trunc:
  case Intrinsic::nvvm_mul24_i:
  case Intrinsic::nvvm_fma_rn_d:
  case Intrinsic::nvvm_add_rn_d:
  case Intrinsic::nvvm_mul_rn_d:
  case Intrinsic::nvvm_lohi_i2d:
  case Intrinsic::nvvm_d2i_hi:
  case Intrinsic::nvvm_d2i_lo:
    return false; // these use the normal function call emission
  case Intrinsic::nvvm_fabs_d:
    Out << "fabs(";
    writeOperand(I.getArgOperand(0), ContextCasted);
    Out << ")";
    return true;
  case Intrinsic::nvvm_fabs_f:
    Out << "fabsf(";
    writeOperand(I.getArgOperand(0), ContextCasted);
    Out << ")";
    return true;
  }
}

// This converts the llvm constraint string to something gcc is expecting.
// TODO: work out platform independent constraints and factor those out
//      of the per target tables
//      handle multiple constraint codes
std::string CWriter::InterpretASMConstraint(InlineAsm::ConstraintInfo &c) {
  return TargetLowering::AsmOperandInfo(c).ConstraintCode;
#if 0
  cwriter_assert(c.Codes.size() == 1 && "Too many asm constraint codes to handle");

  // Grab the translation table from MCAsmInfo if it exists.
  const MCRegisterInfo *MRI;
  const MCAsmInfo *TargetAsm;
  std::string Triple = TheModule->getTargetTriple();
  if (Triple.empty())
    Triple = llvm::sys::getDefaultTargetTriple();

  std::string E;
  if (const Target *Match = TargetRegistry::lookupTarget(Triple, E)) {
    MRI = Match->createMCRegInfo(Triple);
    TargetAsm = Match->createMCAsmInfo(*MRI, Triple);
  } else {
    return c.Codes[0];
  }

  const char *const *table = TargetAsm->getAsmCBE();

  // Search the translation table if it exists.
  for (int i = 0; table && table[i]; i += 2)
    if (c.Codes[0] == table[i]) {
      delete TargetAsm;
      delete MRI;
      return table[i+1];
    }

  // Default is identity.
  delete TargetAsm;
  delete MRI;
  return c.Codes[0];
#endif
}

// TODO: import logic from AsmPrinter.cpp
static std::string gccifyAsm(std::string asmstr) {
  for (std::string::size_type i = 0; i != asmstr.size(); ++i)
    if (asmstr[i] == '\n')
      asmstr.replace(i, 1, "\\n");
    else if (asmstr[i] == '\t')
      asmstr.replace(i, 1, "\\t");
    else if (asmstr[i] == '$') {
      if (asmstr[i + 1] == '{') {
        std::string::size_type a = asmstr.find_first_of(':', i + 1);
        std::string::size_type b = asmstr.find_first_of('}', i + 1);
        std::string n = "%" + asmstr.substr(a + 1, b - a - 1) +
                        asmstr.substr(i + 2, a - i - 2);
        asmstr.replace(i, b - i + 1, n);
        i += n.size() - 1;
      } else
        asmstr.replace(i, 1, "%");
    } else if (asmstr[i] == '%') // grr
    {
      asmstr.replace(i, 1, "%%");
      ++i;
    }

  return asmstr;
}

// TODO: assumptions about what consume arguments from the call are likely wrong
//      handle communitivity
void CWriter::visitInlineAsm(CallInst &CI) {
  CurInstr = &CI;

  InlineAsm *as = cast<InlineAsm>(CI.getCalledOperand());
  InlineAsm::ConstraintInfoVector Constraints = as->ParseConstraints();

  std::vector<std::pair<Value *, int>> ResultVals;
  if (CI.getType() == Type::getVoidTy(CI.getContext()))
    ;
  else if (StructType *ST = dyn_cast<StructType>(CI.getType())) {
    for (unsigned i = 0, e = ST->getNumElements(); i != e; ++i)
      ResultVals.push_back(std::make_pair(&CI, (int)i));
  } else {
    ResultVals.push_back(std::make_pair(&CI, -1));
  }

  // Fix up the asm string for gcc and emit it.
  Out << "__asm__ volatile (\"" << gccifyAsm(as->getAsmString()) << "\"\n";
  Out << "        :";

  unsigned ValueCount = 0;
  bool IsFirst = true;

  // Convert over all the output constraints.
  for (InlineAsm::ConstraintInfoVector::iterator I = Constraints.begin(),
                                                 E = Constraints.end();
       I != E; ++I) {

    if (I->Type != InlineAsm::isOutput) {
      ++ValueCount;
      continue; // Ignore non-output constraints.
    }

    cwriter_assert(I->Codes.size() == 1 &&
                   "Too many asm constraint codes to handle");
    std::string C = InterpretASMConstraint(*I);
    if (C.empty())
      continue;

    if (!IsFirst) {
      Out << ", ";
      IsFirst = false;
    }

    // Unpack the dest.
    Value *DestVal;
    int DestValNo = -1;

    if (ValueCount < ResultVals.size()) {
      DestVal = ResultVals[ValueCount].first;
      DestValNo = ResultVals[ValueCount].second;
    } else
      DestVal = CI.getArgOperand(ValueCount - ResultVals.size());

    if (I->isEarlyClobber)
      C = "&" + C;

    Out << "\"=" << C << "\"(" << GetValueName(DestVal);
    if (DestValNo != -1) {
      if (StructType *ST = dyn_cast<StructType>(DestVal->getType()))
        Out << "." << getFieldName(ST) << DestValNo; // Multiple retvals.
      else
        Out << ".field" << DestValNo; // Fallback for unexpected non-struct.
    }
    Out << ")";
    ++ValueCount;
  }

  // Convert over all the input constraints.
  Out << "\n        :";
  IsFirst = true;
  ValueCount = 0;
  for (InlineAsm::ConstraintInfoVector::iterator I = Constraints.begin(),
                                                 E = Constraints.end();
       I != E; ++I) {
    if (I->Type != InlineAsm::isInput) {
      ++ValueCount;
      continue; // Ignore non-input constraints.
    }

    cwriter_assert(I->Codes.size() == 1 &&
                   "Too many asm constraint codes to handle");
    std::string C = InterpretASMConstraint(*I);
    if (C.empty())
      continue;

    if (!IsFirst) {
      Out << ", ";
      IsFirst = false;
    }

    cwriter_assert(ValueCount >= ResultVals.size() &&
                   "Input can't refer to result");
    Value *SrcVal = CI.getArgOperand(ValueCount - ResultVals.size());

    Out << "\"" << C << "\"(";
    if (!I->isIndirect)
      writeOperand(SrcVal);
    else
      writeOperandDeref(SrcVal);
    Out << ")";
  }

  // Convert over the clobber constraints.
  IsFirst = true;
  for (InlineAsm::ConstraintInfoVector::iterator I = Constraints.begin(),
                                                 E = Constraints.end();
       I != E; ++I) {
    if (I->Type != InlineAsm::isClobber)
      continue; // Ignore non-input constraints.

    cwriter_assert(I->Codes.size() == 1 &&
                   "Too many asm constraint codes to handle");
    std::string C = InterpretASMConstraint(*I);
    if (C.empty())
      continue;

    if (!IsFirst) {
      Out << ", ";
      IsFirst = false;
    }

    Out << '\"' << C << '"';
  }

  Out << ")";
}

void CWriter::visitAllocaInst(AllocaInst &I) {
  CurInstr = &I;

  headerUseBuiltinAlloca();

  Out << '(';
  printTypeName(Out, I.getType());
  Out << ") alloca(sizeof(";
  printTypeName(Out, I.getType()->getElementType());
  if (I.isArrayAllocation()) {
    Out << ") * (";
    writeOperand(I.getArraySize(), ContextCasted);
  }
  Out << "))";
}

Value *CWriter::findUnderlyingObject(Value *Ptr) {
  if (!Ptr)
    return Ptr;
  Value *Current = Ptr;
  std::set<Value *> Visited;
  while (Current && Visited.insert(Current).second) {
    auto it = Times2Dereference.find(Current);
    if (it != Times2Dereference.end())
      return Current;

    // Peel nested GEP chains uniformly for instructions and ConstantExpr GEPs.
    if (auto *GEP = dyn_cast<GEPOperator>(Current)) {
      Current = GEP->getPointerOperand();
      continue;
    }

    // Peel instruction casts.
    if (auto *CastI = dyn_cast<CastInst>(Current)) {
      Current = CastI->getOperand(0);
      continue;
    }

    // Peel constant-expression casts (including addrspacecast/bitcast).
    if (auto *CE = dyn_cast<ConstantExpr>(Current)) {
      if (CE->isCast()) {
        Current = CE->getOperand(0);
        continue;
      }
    }

    // Follow load roots to their pointer operands for deref accounting.
    if (auto *Ld = dyn_cast<LoadInst>(Current)) {
      Current = Ld->getPointerOperand();
      continue;
    }

    break;
  }

  if (Current && Times2Dereference.find(Current) != Times2Dereference.end())
    return Current;

  // Failed to recover a tracked underlying object.
  return nullptr;
}

bool CWriter::printGEPExpressionStruct(Value *Ptr, gep_type_iterator I,
                                       gep_type_iterator E, bool accessMemory,
                                       bool printReference) {
  auto resolveTrivialPointerAlias = [&](auto &&Self, Value *V,
                                        SmallPtrSetImpl<Value *> &Visited)
      -> Value * {
    if (!V || !Visited.insert(V).second)
      return V;

    if (Instruction *Inst = dyn_cast<Instruction>(V)) {
      auto DeleteIt = deleteAndReplaceInsts.find(Inst);
      if (DeleteIt != deleteAndReplaceInsts.end())
        return Self(Self, DeleteIt->second, Visited);
    }

    if (auto *PN = dyn_cast<PHINode>(V)) {
      Value *ResolvedIncoming = nullptr;
      for (unsigned Idx = 0; Idx < PN->getNumIncomingValues(); ++Idx) {
        Value *Incoming = PN->getIncomingValue(Idx);
        if (Incoming == PN || isa<UndefValue>(Incoming))
          continue;
        Value *Resolved = Self(Self, Incoming, Visited);
        if (!ResolvedIncoming) {
          ResolvedIncoming = Resolved;
          continue;
        }
        if (ResolvedIncoming != Resolved)
          return V;
      }
      if (ResolvedIncoming)
        return ResolvedIncoming;
    }

    auto PhiIt = InstsToReplaceByPhi.find(V);
    if (PhiIt != InstsToReplaceByPhi.end() && !isa<PHINode>(PhiIt->second))
      return Self(Self, PhiIt->second, Visited);

    return V;
  };
  auto getLinearizedElementCount = [](Type *Ty) -> uint64_t {
    uint64_t Count = 1;
    while (auto *AT = dyn_cast<ArrayType>(Ty)) {
      Count *= AT->getNumElements();
      Ty = AT->getElementType();
    }
    return Count;
  };
  auto getFlatArrayCastBase = [](Value *V) -> Value * {
    if (auto *CI = dyn_cast<CastInst>(V))
      return CI->getOperand(0);
    if (auto *CE = dyn_cast<ConstantExpr>(V))
      if (CE->isCast() && CE->getNumOperands() >= 1)
        return CE->getOperand(0);
    return nullptr;
  };
  auto getFlatArrayCastSourcePtrTy = [](Value *V) -> PointerType * {
    if (auto *CI = dyn_cast<CastInst>(V))
      return dyn_cast<PointerType>(CI->getOperand(0)->getType());
    if (auto *CE = dyn_cast<ConstantExpr>(V))
      if (CE->isCast() && CE->getNumOperands() >= 1)
        return dyn_cast<PointerType>(CE->getOperand(0)->getType());
    return nullptr;
  };
  auto getNestedArrayLeafType = [](Type *Ty) -> Type * {
    while (auto *AT = dyn_cast<ArrayType>(Ty))
      Ty = AT->getElementType();
    return Ty;
  };
  auto isFlatArrayReinterpretCast = [&](Value *V, ArrayType *DestAT) -> bool {
    auto *SrcPtrTy = getFlatArrayCastSourcePtrTy(V);
    if (!SrcPtrTy || !DestAT)
      return false;

    Type *SrcEltTy = SrcPtrTy->getElementType();
    if (SrcEltTy == DestAT)
      return false;

    Type *DestLeafTy = getNestedArrayLeafType(DestAT);
    if (auto *SrcAT = dyn_cast<ArrayType>(SrcEltTy)) {
      // If the source is already a structured array (e.g. LU's [13 x [5 x double]])
      // then preserving aggregate indexing is correct and flattening loses row
      // strides. Only reinterpret genuinely flat array carriers here.
      if (isa<ArrayType>(SrcAT->getElementType()))
        return false;
      return getNestedArrayLeafType(SrcAT) == DestLeafTy;
    }

    return SrcEltTy == DestLeafTy;
  };
  auto appendGEPIndices = [](SmallVectorImpl<std::pair<Type *, Value *>> &OutIdx,
                             gep_type_iterator Begin,
                             gep_type_iterator End) {
    for (auto It = Begin; It != End; ++It)
      OutIdx.push_back({It.getIndexedType(), It.getOperand()});
  };
  auto collectArrayCastPrefix =
      [&](auto &&Self, Value *V,
          SmallVectorImpl<std::pair<Type *, Value *>> &OutIdx, Value *&BasePtr,
          ArrayType *&TopIndexedTy) -> bool {
    if (!V)
      return false;

    SmallPtrSet<Value *, 8> VisitedAliases;
    Value *ResolvedV =
        resolveTrivialPointerAlias(resolveTrivialPointerAlias, V, VisitedAliases);
    if (ResolvedV != V)
      return Self(Self, ResolvedV, OutIdx, BasePtr, TopIndexedTy);

    if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
      if (!Self(Self, GEP->getPointerOperand(), OutIdx, BasePtr, TopIndexedTy))
        return false;
      appendGEPIndices(OutIdx, gep_type_begin(GEP), gep_type_end(GEP));
      return true;
    }
    if (auto *GEPOp = dyn_cast<GEPOperator>(V)) {
      if (!Self(Self, GEPOp->getPointerOperand(), OutIdx, BasePtr, TopIndexedTy))
        return false;
      appendGEPIndices(OutIdx, gep_type_begin(GEPOp), gep_type_end(GEPOp));
      return true;
    }

    Value *CastBase = getFlatArrayCastBase(V);
    auto *PtrTy = dyn_cast_or_null<PointerType>(V->getType());
    if (!CastBase || !PtrTy)
      return false;
    auto *AT = dyn_cast<ArrayType>(PtrTy->getElementType());
    if (!AT)
      return false;
    if (!isFlatArrayReinterpretCast(V, AT))
      return false;

    BasePtr = CastBase;
    TopIndexedTy = AT;
    return true;
  };
  auto collectFlatArrayCastPrefix =
      [&](auto &&Self, Value *V,
          SmallVectorImpl<std::pair<Type *, Value *>> &OutIdx, Value *&BasePtr,
          Type *&TopIndexedTy) -> bool {
    if (!V)
      return false;

    SmallPtrSet<Value *, 8> VisitedAliases;
    Value *ResolvedV = resolveTrivialPointerAlias(resolveTrivialPointerAlias, V,
                                                  VisitedAliases);
    if (ResolvedV != V)
      return Self(Self, ResolvedV, OutIdx, BasePtr, TopIndexedTy);

    if (auto *GEP = dyn_cast<GetElementPtrInst>(V)) {
      if (!Self(Self, GEP->getPointerOperand(), OutIdx, BasePtr, TopIndexedTy))
        return false;
      appendGEPIndices(OutIdx, gep_type_begin(GEP), gep_type_end(GEP));
      return true;
    }
    if (auto *GEPOp = dyn_cast<GEPOperator>(V)) {
      if (!Self(Self, GEPOp->getPointerOperand(), OutIdx, BasePtr, TopIndexedTy))
        return false;
      appendGEPIndices(OutIdx, gep_type_begin(GEPOp), gep_type_end(GEPOp));
      return true;
    }

    Value *FlatBase = getFlatArrayCastBase(V);
    auto *PtrTy = dyn_cast_or_null<PointerType>(V->getType());
    if (!FlatBase || !PtrTy)
      return false;
    auto *AT = dyn_cast<ArrayType>(PtrTy->getElementType());
    if (!AT)
      return false;
    if (!isFlatArrayReinterpretCast(V, AT))
      return false;
    BasePtr = FlatBase;
    TopIndexedTy = AT;
    return true;
  };
  auto emitArrayViewAccess = [&](Value *BasePtr, ArrayType *TopIndexedTy,
                                 ArrayRef<std::pair<Type *, Value *>> Indices)
      -> bool {
    if (!BasePtr || !TopIndexedTy)
      return false;

    for (Type *Ty = TopIndexedTy; auto *AT = dyn_cast<ArrayType>(Ty);
         Ty = AT->getElementType()) {
      if (AT->getNumElements() == 0)
        return false;
    }

    if (Indices.empty())
      return false;

    auto getArrayRank = [](ArrayType *AT) {
      unsigned Rank = 0;
      for (Type *Ty = AT; auto *Arr = dyn_cast<ArrayType>(Ty);
           Ty = Arr->getElementType())
        ++Rank;
      return Rank;
    };
    auto printPointerToArrayCastType = [&](ArrayType *AT) {
      SmallVector<uint64_t, 8> Extents;
      Type *LeafTy = AT;
      while (auto *Arr = dyn_cast<ArrayType>(LeafTy)) {
        Extents.push_back(Arr->getNumElements());
        LeafTy = Arr->getElementType();
      }
      printTypeName(Out, LeafTy, false);
      Out << " (*)";
      for (uint64_t Extent : Extents)
        Out << "[" << Extent << "]";
    };
    auto normalizeArrayViewIndices =
        [&](ArrayType *AT, ArrayRef<std::pair<Type *, Value *>> RawIndices,
            SmallVectorImpl<Value *> &Normalized) {
          unsigned DesiredCount = getArrayRank(AT) + 1;
          if (RawIndices.size() <= DesiredCount) {
            for (const auto &Indexed : RawIndices)
              Normalized.push_back(Indexed.second);
            return;
          }

          unsigned Cursor = 0;
          if (RawIndices.empty())
            return;
          Normalized.push_back(RawIndices[Cursor++].second);
          while (Cursor < RawIndices.size() && Normalized.size() < DesiredCount) {
            if (Cursor + 1 < RawIndices.size() &&
                isConstantNull(RawIndices[Cursor].second)) {
              ++Cursor;
            }
            if (Cursor < RawIndices.size())
              Normalized.push_back(RawIndices[Cursor++].second);
          }

          while (Cursor < RawIndices.size() && Normalized.size() < DesiredCount)
            Normalized.push_back(RawIndices[Cursor++].second);
        };

    SmallVector<Value *, 8> NormalizedIndices;
    normalizeArrayViewIndices(TopIndexedTy, Indices, NormalizedIndices);
    if (NormalizedIndices.empty())
      return false;

    Out << "((";
    printPointerToArrayCastType(TopIndexedTy);
    Out << ")";
    writePointerCastSource(BasePtr, PointerCastGeneral, ContextNormal,
                           /*startExpression=*/false);
    Out << ")";
    for (Value *Idx : NormalizedIndices) {
      Out << "[";
      writeOperand(Idx, ContextCasted);
      Out << "]";
    }
    return true;
  };
  auto emitLinearizedFlatArrayAccess = [&](Value *BasePtr, Type *TopIndexedTy,
                                           ArrayRef<std::pair<Type *, Value *>> Indices)
      -> bool {
    if (!BasePtr || !TopIndexedTy)
      return false;
    if (!isa<ArrayType>(TopIndexedTy))
      return false;

    // Zero-length array carriers like [0 x double] are used as raw shared-memory
    // pointer bases. Reconstructing them as indexed lvalues turns a no-op base
    // GEP (e.g. gep 0,0) into "ptr[0]", which then scalarizes too early and
    // breaks later subscripting in kernels like CG.
    for (Type *Ty = TopIndexedTy; auto *AT = dyn_cast<ArrayType>(Ty);
         Ty = AT->getElementType()) {
      if (AT->getNumElements() == 0)
        return false;
    }

    Type *LeafTy = TopIndexedTy;
    while (auto *AT = dyn_cast<ArrayType>(LeafTy))
      LeafTy = AT->getElementType();
    if (!LeafTy->isSingleValueType())
      return false;

    SmallVector<std::pair<Value *, uint64_t>, 8> Terms;
    Type *CurTy = TopIndexedTy;
    auto It = Indices.begin();
    if (It == Indices.end())
      return false;

    auto addIndexedTerm = [&](Value *Idx, uint64_t Scale) {
      if (isConstantNull(Idx))
        return;
      Terms.push_back({Idx, Scale});
    };

    uint64_t FirstScale = 1;
    if (isa<ArrayType>(CurTy))
      FirstScale = getLinearizedElementCount(CurTy);
    addIndexedTerm(It->second, FirstScale);

    for (++It; It != Indices.end(); ++It) {
      if (auto *AT = dyn_cast<ArrayType>(CurTy)) {
        CurTy = AT->getElementType();
        addIndexedTerm(It->second, getLinearizedElementCount(CurTy));
        continue;
      }
      if (isa<IntegerType>(CurTy) || isa<PointerType>(CurTy) ||
          CurTy->isFloatingPointTy()) {
        addIndexedTerm(It->second, 1);
        CurTy = It->first;
        continue;
      }
      return false;
    }

    Out << "((";
    printTypeName(Out, PointerType::getUnqual(LeafTy));
    Out << ")";
    writePointerCastSource(BasePtr, PointerCastGeneral, ContextNormal,
                           /*startExpression=*/false);
    if (Terms.empty()) {
      Out << ")";
      return true;
    }

    Out << ")[";
    {
      bool First = true;
      for (auto &[Idx, Scale] : Terms) {
        if (!First)
          Out << " + ";
        First = false;
        if (Scale != 1)
          Out << Scale << " * ";
        bool NeedParens =
            isa<Instruction>(Idx) || isa<ConstantExpr>(Idx) || isa<Operator>(Idx);
        if (NeedParens)
          Out << "(";
        writeOperand(Idx, ContextCasted);
        if (NeedParens)
          Out << ")";
      }
    }
    Out << "]";
    return true;
  };
  auto writeGroupedBaseOperand = [&](Value *Base, bool AccessMemory) {
    bool needsGrouping =
        isa<Instruction>(Base) || isa<ConstantExpr>(Base) ||
        isa<Operator>(Base);
    if (needsGrouping)
      Out << '(';
    writeOperandInternal(Base, ContextNormal, AccessMemory);
    if (needsGrouping)
      Out << ')';
  };

  // If there are no indices, just print out the pointer.
  if (I == E) {
    writeOperand(Ptr);
    return false;
  }

  if (accessMemory) {
    SmallVector<std::pair<Type *, Value *>, 8> ArrayIndices;
    Value *ArrayBase = nullptr;
    ArrayType *ArrayTopIndexedTy = nullptr;
    if (collectArrayCastPrefix(collectArrayCastPrefix, Ptr, ArrayIndices,
                               ArrayBase, ArrayTopIndexedTy)) {
      appendGEPIndices(ArrayIndices, I, E);
      if (emitArrayViewAccess(ArrayBase, ArrayTopIndexedTy, ArrayIndices))
        return false;
    }

    SmallVector<std::pair<Type *, Value *>, 8> FlatIndices;
    Value *FlatBase = nullptr;
    Type *TopIndexedTy = nullptr;
    if (collectFlatArrayCastPrefix(collectFlatArrayCastPrefix, Ptr, FlatIndices,
                                   FlatBase, TopIndexedTy)) {
      appendGEPIndices(FlatIndices, I, E);
      if (emitLinearizedFlatArrayAccess(FlatBase, TopIndexedTy, FlatIndices))
        return false;
    }
  }

  // Find out if the last index is into a vector.  If so, we have to print this
  // specially.  Since vectors can't have elements of indexable type, only the
  // last index could possibly be of a vector element.
  VectorType *LastIndexIsVector = 0;
  {
    for (gep_type_iterator TmpI = I; TmpI != E; ++TmpI)
      LastIndexIsVector = dyn_cast<VectorType>(TmpI.getIndexedType());
  }

  // If the last index is into a vector, we can't print it as &a[i][j] because
  // we can't index into a vector with j in GCC.  Instead, emit this as
  // (((float*)&a[i])+j)
  // TODO: this is no longer true now that we don't represent vectors using
  // gcc-extentions
  if (LastIndexIsVector) {
    Out << "((";
    printTypeName(Out,
                  PointerType::getUnqual(LastIndexIsVector->getElementType()));
    Out << ")(";
  }

  if (gepStart) {
    Value *UO = findUnderlyingObject(Ptr);
    currValue2DerefCnt = std::pair<Value *, int>(UO, Times2Dereference[UO]);
  }

  if (!accessMemory && gepStart) {
    // check every gep whether there is a deference operation
    bool dereferenced = false;
    auto it = I;
    Type *idxType = it.getIndexedType();
    if (isa<StructType>(idxType) && (++it != E)) {
      dereferenced = true;
    }
    it = I;
    if (isa<ArrayType>(idxType)) {
      if ((++it != E) && !isConstantNull(it.getOperand()))
        dereferenced = true;
    }
    /*GetElementPtrInst *gepInst = dyn_cast<GetElementPtrInst>(Ptr);
    while(gepInst){
      it = gep_type_begin(gepInst);
      Type *idxType = it.getIndexedType();
      if(isa<StructType>(idxType) && (++it != E)){
        dereferenced = true;
      }
      gepInst = dyn_cast<GetElementPtrInst>(gepInst->getPointerOperand());
    }*/

    // In memory-access contexts, the GEP already denotes the address to
    // dereference; emitting an extra '&' turns value loads/stores into
    // address arithmetic. This also covers nested GEP-printing while
    // materializing a load/store pointer expression.
    bool inMemoryAccessPath = isa<LoadInst>(CurInstr) || isa<StoreInst>(CurInstr);
    if (!inMemoryAccessPath) {
      if (const auto *CurGEP = dyn_cast_or_null<GetElementPtrInst>(CurInstr)) {
        auto *CurGEPNonConst = const_cast<GetElementPtrInst *>(CurGEP);
        inMemoryAccessPath =
            accessGEPMemory.find(CurGEPNonConst) != accessGEPMemory.end() ||
            GEPAccessesMemory(CurGEPNonConst);
      }
    }
    bool resultPointsToAggregate = true;
    bool feedsAnotherGEP = false;
    if (const auto *CurGEP = dyn_cast_or_null<GetElementPtrInst>(CurInstr)) {
      if (Type *ResultEltTy = CurGEP->getResultElementType())
        resultPointsToAggregate = ResultEltTy->isAggregateType();
      for (const User *U : CurGEP->users()) {
        if (isa<GEPOperator>(U)) {
          feedsAnotherGEP = true;
          break;
        }
      }
    }
    // Emit '&' only when this GEP result still denotes an aggregate object in
    // a non-memory-access context; scalar-field GEPs should stay as direct
    // lvalues to avoid malformed forms like '&(ptr)->field' on atomic LHS.
    bool emittedAddressOf = dereferenced && !inMemoryAccessPath &&
                            resultPointsToAggregate && !feedsAnotherGEP;
    if (emittedAddressOf)
      Out << '&';
    if (dereferenced) {
      errs() << "ANDREW: printGEPExpressionStruct dereferenced=true for Ptr=" << *Ptr
             << ", CurInstr=";
      if (CurInstr)
        errs() << *CurInstr;
      else
        errs() << "<null>";
      errs() << ", emitted_address_of=" << emittedAddressOf << "\n";
    }
  }

  std::set<Value *> NegOpnd;

  // check if there are negative indexes
  auto it = I;
  for (; it != E; ++it) {
    Value *Opnd = it.getOperand();
    if (ConstantInt *intOpnd = dyn_cast<ConstantInt>(Opnd))
      if (intOpnd->getSExtValue() < 0)
        NegOpnd.insert(intOpnd);
  }

  Type *IntoT = I.getIndexedType();
  Value *FirstOp = I.getOperand();
  auto nextIndexIt = I;
  ++nextIndexIt;
  bool hasRemainingIndices = (nextIndexIt != E);
  bool firstIndexFeedsAnotherGEP = false;
  if (const auto *CurGEP = dyn_cast_or_null<GetElementPtrInst>(CurInstr)) {
    for (const User *U : CurGEP->users()) {
      if (isa<GEPOperator>(U)) {
        firstIndexFeedsAnotherGEP = true;
        break;
      }
    }
  }
  // Zero first-index GEPs are the common "stay at base object" form and should
  // not be wrapped as '&(...)' because that turns valid field access into
  // invalid address-of rvalues (e.g. '(&arr[i]).field').
  bool addAddressWrapper = printReference && isNegative(FirstOp);
  if (IntoT->isIntegerTy() || IntoT->isPointerTy() ||
      IntoT->isFloatingPointTy()) {
    // Scalar/pointer GEP stepping already yields an address expression like
    // "(base + idx)". Wrapping it in "&(...)" turns it into pointer-to-pointer
    // and breaks loads/stores on shared-memory reductions.
    addAddressWrapper = false;
  }
  if (ArrayType *ArrTy = dyn_cast<ArrayType>(IntoT)) {
    // Shared-memory lowering uses [0 x T] as a carrier type. Wrapping
    // "&(... + 0)" here takes address of a pointer rvalue and produces invalid
    // C (e.g. '&(((double*)base)+0)'). Keep it as pointer arithmetic instead.
    if (ArrTy->getNumElements() == 0)
      addAddressWrapper = false;
  }
  if (addAddressWrapper)
    Out << "(&";

  bool currGEPisPointer = !(isa<GetElementPtrInst>(Ptr) ||
                            isa<AllocaInst>(Ptr) || isa<GlobalVariable>(Ptr));
  auto getLinearizedArrayElements = [](Type *Ty) -> uint64_t {
    uint64_t Count = 1;
    while (auto *AT = dyn_cast<ArrayType>(Ty)) {
      Count *= AT->getNumElements();
      Ty = AT->getElementType();
    }
    return Count;
  };
  // Constant-expression casts of fixed-size arrays (e.g. addrspacecast @ce)
  // should keep aggregate indexing form (ce[i][j]), not linearized pointer
  // stepping (ce+stride*i) which loses one level of indexing in C.
  if (auto *CE = dyn_cast<ConstantExpr>(Ptr)) {
    if (CE->isCast()) {
      if (auto *DstPtrTy = dyn_cast<PointerType>(CE->getType())) {
        if (auto *DstArrTy = dyn_cast<ArrayType>(DstPtrTy->getElementType())) {
          if (DstArrTy->getNumElements() > 0)
            currGEPisPointer = false;
        }
      }
    }
  }
  bool openPointerExprParen = false;
  // first index
  bool flattened3D = false;
  uint64_t nextSize = 0;
  if (isa<StructType>(IntoT) || isa<ArrayType>(IntoT)) {
    // if it's a struct or array, whether it's pointer or not, first index is
    // offset and zero can be eliminated
    errs() << "SUSAN: first index is struct or array type\n";
    if (!isConstantNull(FirstOp)) {
      Out << '(';
      writeGroupedBaseOperand(Ptr, accessMemory);
      if (!isNegative(FirstOp)) {
        Out << '+';
        if (ArrayType *arrTy = dyn_cast<ArrayType>(IntoT)) {
          auto size = arrTy->getNumElements();
          Out << size << "*";
          if (ArrayType *elemArrTy =
                  dyn_cast<ArrayType>(arrTy->getElementType())) {
            nextSize = getLinearizedArrayElements(elemArrTy);
            // Flatten multidimensional indexing only when emitting an address
            // expression for memory access; value-expression printing should
            // keep the normal bracketed indexing path and balanced parens.
            if (accessMemory) {
              Out << nextSize << "*";
              flattened3D = true;
            }
          }
        }
        writeOperand(FirstOp);
        if (!flattened3D)
          Out << ')';
      } else {
        errs() << "SUSAN: found negative int" << *FirstOp << "\n";
        writeOperand(FirstOp);
        Out << ')';
      }
    } else {
      errs() << "SUSAN: printing Ptr 9975 " << *Ptr << "\n";
      // Keep a grouping paren for zero-first-index aggregate GEPs in memory
      // access contexts so later linearized row steps bind with the base before
      // a trailing "[idx]" is emitted. Do this unconditionally in accessMemory
      // mode because Ptr may itself print as a complex GEP expression.
      if (accessMemory) {
        Out << '(';
        openPointerExprParen = true;
      }
      writeGroupedBaseOperand(Ptr, accessMemory);
    }
  } else if (isa<IntegerType>(IntoT) || isa<PointerType>(IntoT) ||
             IntoT->isDoubleTy() || IntoT->isFloatTy()) {
    errs() << "SUSAN: first index is integer/pointertype type\n";
    // if indexed type is an integer, it means accessing an array, or a block of
    // allocated memory
    if (accessMemory) {
      // If the base pointer comes from a cast that uses a zero-length array
      // carrier type ([0 x T]), keep this step in pointer-arithmetic form to
      // avoid scalarizing too early into "(casted_ptr)[idx]".
      // Do NOT apply this to general aggregate->byte-pointer casts (e.g.
      // "(uint8_t*)&obj"), where indexed lvalue form is required for stores.
      bool preferPointerArithmetic = false;
      if (auto *CastPtr = dyn_cast<CastInst>(Ptr)) {
        if (auto *SrcPtrTy = dyn_cast<PointerType>(CastPtr->getSrcTy())) {
          if (auto *SrcArrTy = dyn_cast<ArrayType>(SrcPtrTy->getElementType()))
            preferPointerArithmetic |= (SrcArrTy->getNumElements() == 0);
        }
        if (auto *DstPtrTy = dyn_cast<PointerType>(CastPtr->getDestTy())) {
          if (auto *DstArrTy = dyn_cast<ArrayType>(DstPtrTy->getElementType()))
            preferPointerArithmetic |= (DstArrTy->getNumElements() == 0);
        }
      } else if (auto *CE = dyn_cast<ConstantExpr>(Ptr)) {
        if (CE->isCast()) {
          if (auto *SrcPtrTy =
                  dyn_cast<PointerType>(CE->getOperand(0)->getType())) {
            if (auto *SrcArrTy = dyn_cast<ArrayType>(SrcPtrTy->getElementType()))
              preferPointerArithmetic |= (SrcArrTy->getNumElements() == 0);
          }
          if (auto *DstPtrTy = dyn_cast<PointerType>(CE->getType())) {
            if (auto *DstArrTy = dyn_cast<ArrayType>(DstPtrTy->getElementType()))
              preferPointerArithmetic |= (DstArrTy->getNumElements() == 0);
          }
        }
      }

      // If this scalar/pointer-step GEP feeds another GEP, keep it in address
      // form. Emitting "ptr[idx]" here scalarizes too early and can produce
      // invalid chained forms like "base[idx0][idx1]".
      bool keepPointerForChainedGEP = firstIndexFeedsAnotherGEP;

      // if index is negative, it's treated as a block of memory, and should be
      // translated as *(x-offset) (Hofstadter-Q-sequence)
      if (preferPointerArithmetic || keepPointerForChainedGEP) {
        Out << '(';
        writeGroupedBaseOperand(Ptr, accessMemory);
        if (!isConstantNull(FirstOp)) {
          Out << '+';
          writeOperand(FirstOp);
        }
        if (hasRemainingIndices)
          openPointerExprParen = true;
        else
          Out << ')';
      } else if (currValue2DerefCnt.second) {
        errs() << "SUSAN: writing ptr 10000:" << *Ptr << "\n";
        currValue2DerefCnt.second--;
        if (NegOpnd.find(FirstOp) != NegOpnd.end()) {
          Out << "*(";
          writeGroupedBaseOperand(Ptr, accessMemory);
          Out << '+';
          writeOperand(FirstOp);
          Out << ')';
          currGEPisPointer = false;
        } else if (isConstantNull(FirstOp) && firstIndexFeedsAnotherGEP) {
          // For chained GEPs only, leading zero index is an address no-op.
          // Emitting "[0]" here dereferences too early and breaks the next GEP
          // (e.g. "((ptr)[0]+...)[m]").
          writeGroupedBaseOperand(Ptr, accessMemory);
          currGEPisPointer = true;
        } else {
          errs() << "SUSAN: writing ptr 9994: " << *Ptr << "\n";
          writeGroupedBaseOperand(Ptr, accessMemory);
          Out << '[';
          writeOperand(FirstOp);
          Out << ']';
          currGEPisPointer = false;
        }
      } else {
        Out << '(';
        writeGroupedBaseOperand(Ptr, accessMemory);
        Out << '+';
        writeOperand(FirstOp);
        if (hasRemainingIndices)
          openPointerExprParen = true;
        else
          Out << ')';
      }
    } else {

      if (!isConstantNull(FirstOp) && !printReference) {
        errs() << "SUSAN: writing ptr 10029:" << *Ptr << "\n";
        Out << '(';
        writeGroupedBaseOperand(Ptr, false);
        Out << '+';
        writeOperand(FirstOp);
        if (hasRemainingIndices)
          openPointerExprParen = true;
        else
          Out << ')';
      } else {
        writeGroupedBaseOperand(Ptr, false);
      }
    }
  } else {
    assert(0 && "vector type not supported\n");
  }

  I++;

  // check if previous GEP operand is a pointer
  bool prevGEPisPointer = false;
  if (GetElementPtrInst *prevGEP = dyn_cast<GetElementPtrInst>(Ptr)) {
    if (GEPPointers.find(prevGEP) != GEPPointers.end()) {
      prevGEPisPointer = true;
    }
  }
  bool isPointer = currGEPisPointer || prevGEPisPointer;
  // Chained GEPs that are already known pointer-carriers must keep address
  // form through remaining indices; otherwise an intermediate "[]" can
  // scalarize too early and the next GEP tries to subscript a scalar
  // (e.g. "...)[n])[m]"). Do not force this for normal fixed local/global
  // aggregates (e.g. MG ten[2][10], LU d_ce[13][5]) where bracket indexing
  // is correct. Force when: base is a GEP in pointer form, or a ConstantExpr
  // cast from flat memory (BT [1024 x double]/[0 x T]); skip when cast is from
  // structured array-of-arrays (LU [13 x [5 x double]]).
  bool baseIsStructuredArrayCast = false;
  if (auto *CE = dyn_cast<ConstantExpr>(Ptr))
    if (CE->isCast() && CE->getNumOperands() >= 1)
      if (auto *SrcPtrTy = dyn_cast<PointerType>(CE->getOperand(0)->getType()))
        if (auto *SrcElemTy = dyn_cast<ArrayType>(SrcPtrTy->getElementType()))
          baseIsStructuredArrayCast =
              (SrcElemTy->getNumElements() > 0 &&
               isa<ArrayType>(SrcElemTy->getElementType()));
  if (accessMemory && firstIndexFeedsAnotherGEP &&
      (prevGEPisPointer || (isa<ConstantExpr>(Ptr) && !baseIsStructuredArrayCast)))
    isPointer = true;

  Type *prevType = IntoT;
  for (; I != E; ++I) {
    Value *Opnd = I.getOperand();
    if (isa<ArrayType>(prevType)) {
      ArrayType *prevArrTy = cast<ArrayType>(prevType);
      // Zero-length arrays are a pointer-arithmetic carrier in LLVM IR.
      // In memory-address contexts, keep them in pointer form:
      //   - [0] is a no-op
      //   - [idx] lowers to +idx (not dereference via []), otherwise we
      //     scalarize too early and can end up casting a loaded scalar back
      //     to a pointer (e.g. (double*)((double*)base)[idx]).
      if (prevArrTy->getNumElements() == 0) {
        if (accessMemory) {
          if (!isConstantNull(Opnd)) {
            if (!openPointerExprParen) {
              Out << '(';
              openPointerExprParen = true;
            }
            Out << '+';
            writeOperand(Opnd);
          }
          prevType = I.getIndexedType();
          continue;
        }
        if (isConstantNull(Opnd)) {
          prevType = I.getIndexedType();
          continue;
        }
      }
      if (accessMemory) {
        // In chained aggregate-decay GEPs, emitting "[0]" on an already-pointer
        // base expression performs an unintended dereference (e.g. ptr[0]) and
        // turns the expression into a scalar too early. Keep this as a no-op so
        // subsequent indices bind to the address expression.
        auto nextI = I;
        ++nextI;
        bool isLastIndex = (nextI == E);
        if (isConstantNull(Opnd) && !isLastIndex) {
          prevType = I.getIndexedType();
          continue;
        }
        // If this GEP itself feeds another GEP, keep the final array step in
        // address form ("+ idx" / "+ stride*idx") instead of "[]". Using
        // brackets here dereferences too early and makes the chained user
        // subscript a scalar (e.g. "...[n])[m]"), which is invalid C.
        if (isPointer && firstIndexFeedsAnotherGEP && isLastIndex) {
          if (!openPointerExprParen) {
            Out << '(';
            openPointerExprParen = true;
          }
          Out << "+";
          if (ArrayType *ElemArrTy =
                  dyn_cast<ArrayType>(prevArrTy->getElementType()))
            Out << getLinearizedArrayElements(ElemArrTy) << "*";
          writeOperand(Opnd);
          prevType = I.getIndexedType();
          continue;
        }
        // If we still have a raw element pointer but are stepping through an
        // array-of-arrays dimension, preserve pointer form via linearized row
        // stepping ("+ stride*idx") instead of "[idx]". Bracket form here would
        // dereference to a scalar too early and break following indices.
        if (isPointer) {
          if (ArrayType *ElemArrTy =
                  dyn_cast<ArrayType>(prevArrTy->getElementType())) {
            if (!openPointerExprParen) {
              Out << '(';
              openPointerExprParen = true;
            }
            Out << "+";
            Out << getLinearizedArrayElements(ElemArrTy) << "*";
            writeOperand(Opnd);
            prevType = I.getIndexedType();
            continue;
          }
        }
        if (flattened3D) {
          Out << "+" << nextSize << "*";
          writeOperand(Opnd);
          Out << ')';
          flattened3D = false;
        } else {
          if (openPointerExprParen) {
            Out << ')';
            openPointerExprParen = false;
          }
          Out << "[";
          writeOperand(Opnd);
          Out << ']';
          isPointer = false;
        }
      } else if (!isConstantNull(Opnd)) {
        if (currValue2DerefCnt.second) {
          errs() << "SUSAN: 10062: " << *Opnd << "\n";
          currValue2DerefCnt.second--;
          Out << '[';
          writeOperand(Opnd);
          Out << ']';
          isPointer = false;
        } else {
          // Some legal constant-expression GEPs (e.g. nested indexing from a
          // global array symbol) can outnumber our conservative dereference
          // budget. Keep lowering by emitting normal indexing instead of
          // aborting codegen.
          errs() << "ANDREW: fallback indexing with exhausted deref budget for "
                 << *Ptr << ", op=" << *Opnd << "\n";
          Out << '[';
          writeOperand(Opnd);
          Out << ']';
          isPointer = false;
        }
      }
    } else if (isa<StructType>(prevType)) {
      auto *StrTy = dyn_cast<StructType>(prevType);
      errs() << "SUSAN: is StructType 10074\n";
      // YEBIN: change field names to support FIXME
      if (accessMemory) {
        // if(currValue2DerefCnt.second){
        // errs() << "SUSAN: is StructType 10079\n";
        // currValue2DerefCnt.second--;
        if (isPointer)
          Out << "->" + getFieldName(StrTy)
              << cast<ConstantInt>(Opnd)->getZExtValue();
        else
          Out << "." + getFieldName(StrTy)
              << cast<ConstantInt>(Opnd)->getZExtValue();
        isPointer = false;
        //}
        // else{
        //  assert( 0 && "SUSAN: dereferencing more than expected?\n");
        //}
      } else {
        if (isPointer)
          Out << "->" + getFieldName(StrTy)
              << cast<ConstantInt>(Opnd)->getZExtValue();
        else
          Out << "." + getFieldName(StrTy)
              << cast<ConstantInt>(Opnd)->getZExtValue();
        isPointer = false;
      }
    } else {
      assert(0 && "vector type not supported\n");
    }
    prevType = I.getIndexedType();
  }

  if (openPointerExprParen)
    Out << ')';

  // A single-index aggregate GEP can open a flattened linearized expression
  // above (e.g. "(base+N*") without hitting the array-index path that closes
  // it. Close it here so chained GEP users don't attach [] to the wrong token.
  if (flattened3D)
    Out << ')';

  if (addAddressWrapper)
    Out << ")";
  return isPointer;

  // if (!isConstantNull(FirstOp)) {
  //   writeOperand(Ptr);
  //   Out << '[';
  //   writeOperandWithCast(FirstOp, Instruction::GetElementPtr);
  //   Out << ']';
  // } else {
  //   // When the first index is 0 (very common) we can simplify it.
  //   if (isAddressExposed(Ptr)) {
  //     // Print P rather than (&P)[0]
  //     writeOperandInternal(Ptr);
  //   } else if (I != E && I.isStruct()) {
  //     // If the second index is a struct index, print P->f instead of P[0].f
  //     writeOperand(Ptr);
  //     Out << "->field" << cast<ConstantInt>(I.getOperand())->getZExtValue();
  //     // Eat the struct index
  //     IntoT = I.getIndexedType();
  //     ++I;
  //   } else {
  //     // Print (*P)[1] instead of P[0][1] (more idiomatic)
  //     Out << "(*";
  //     writeOperand(Ptr);
  //     Out << ")";
  //   }
  // }

  // for (; I != E; ++I) {
  //   Value *Opnd = I.getOperand();

  //   cwriter_assert(
  //       Opnd
  //           ->getType()
  //           ->isIntegerTy()); // TODO: indexing a Vector with a Vector is
  //           valid,
  //                             // but we don't support it here

  //   if (I.isStruct()) {
  //     Out << ".field" << cast<ConstantInt>(Opnd)->getZExtValue();
  //   } else if (IntoT->isArrayTy()) {
  //     // Zero-element array types are either skipped or, for pointers, peeled
  //     // off by skipEmptyArrayTypes. In this latter case, we can translate
  //     // zero-element array indexing as pointer arithmetic.
  //     if (IntoT->getArrayNumElements() == 0) {
  //       if (!isConstantNull(Opnd)) {
  //         // TODO: The operator precedence here is only correct if there are
  //         no
  //         //       subsequent indexable types other than zero-element arrays.
  //         cwriter_assert(skipEmptyArrayTypes(IntoT)->isSingleValueType());
  //         Out << " + (";
  //         writeOperandWithCast(Opnd, Instruction::GetElementPtr);
  //         Out << ')';
  //       }
  //     } else {
  //       Out << ".array[";
  //       writeOperandWithCast(Opnd, Instruction::GetElementPtr);
  //       Out << ']';
  //     }
  //   } else if (!IntoT->isVectorTy()) {
  //     Out << '[';
  //     writeOperandWithCast(Opnd, Instruction::GetElementPtr);
  //     Out << ']';
  //   } else {
  //     // If the last index is into a vector, then print it out as "+j)". This
  //     // works with the 'LastIndexIsVector' code above.
  //     if (!isConstantNull(Opnd)) {
  //       Out << "))"; // avoid "+0".
  //     } else {
  //       Out << ")+(";
  //       writeOperandWithCast(I.getOperand(), Instruction::GetElementPtr);
  //       Out << "))";
  //     }
  //   }

  //   IntoT = I.getIndexedType();
  // }
}

void CWriter::printGEPExpressionArray(Value *Ptr, gep_type_iterator I,
                                      gep_type_iterator E, bool accessMemory) {
  auto writeGroupedBaseOperand = [&](Value *Base, bool AccessMemory) {
    bool needsGrouping =
        isa<Instruction>(Base) || isa<ConstantExpr>(Base) ||
        isa<Operator>(Base);
    if (needsGrouping)
      Out << '(';
    writeOperandInternal(Base, ContextNormal, AccessMemory);
    if (needsGrouping)
      Out << ')';
  };

  // If there are no indices, just print out the pointer.
  if (I == E) {
    writeOperand(Ptr);
    return;
  }

  // Find out if the last index is into a vector.  If so, we have to print this
  // specially.  Since vectors can't have elements of indexable type, only the
  // last index could possibly be of a vector element.
  VectorType *LastIndexIsVector = 0;
  {
    for (gep_type_iterator TmpI = I; TmpI != E; ++TmpI)
      LastIndexIsVector = dyn_cast<VectorType>(TmpI.getIndexedType());
  }

  // Out << "(";

  // If the last index is into a vector, we can't print it as &a[i][j] because
  // we can't index into a vector with j in GCC.  Instead, emit this as
  // (((float*)&a[i])+j)
  // TODO: this is no longer true now that we don't represent vectors using
  // gcc-extentions
  if (LastIndexIsVector) {
    Out << "((";
    printTypeName(Out,
                  PointerType::getUnqual(LastIndexIsVector->getElementType()));
    Out << ")(";
  }

  Type *IntoT = I.getIndexedType();

  // The first index of a GEP is special. It does pointer arithmetic without
  // indexing into the element type.
  Value *FirstOp = I.getOperand();
  IntoT = I.getIndexedType();
  ++I;
  if (!isConstantNull(FirstOp)) {
    // if it's just pointer operation then translates as ptr+1
    if (!accessMemory) {
      writeGroupedBaseOperand(Ptr, false);
      Out << " + ";
      writeOperand(FirstOp, ContextCasted);
    }
    // if it access memory, then translates as ptr[1]
    else {
      writeGroupedBaseOperand(Ptr, true);
      Out << "[";
      writeOperand(FirstOp, ContextCasted);
      Out << "]";
    }

  } else {
    writeGroupedBaseOperand(Ptr, accessMemory);
  }

  // if(accessMemory){
  for (; I != E; ++I) {
    Value *Opnd = I.getOperand();

    cwriter_assert(
        Opnd->getType()
            ->isIntegerTy()); // TODO: indexing a Vector with a Vector is valid,
                              // but we don't support it here

    if (IntoT->isArrayTy()) {
      if (accessMemory) {
        // Zero-element array types are either skipped or, for pointers, peeled
        // off by skipEmptyArrayTypes. In this latter case, we can translate
        // zero-element array indexing as pointer arithmetic.
        if (IntoT->getArrayNumElements() == 0) {
          if (!isConstantNull(Opnd)) {
            // TODO: The operator precedence here is only correct if there are
            // no
            //       subsequent indexable types other than zero-element arrays.
            cwriter_assert(skipEmptyArrayTypes(IntoT)->isSingleValueType());
            Out << " + (";
            writeOperandWithCast(Opnd, Instruction::GetElementPtr);
            Out << ')';
          }
        } else {
          Out << "[";
          writeOperandWithCast(Opnd, Instruction::GetElementPtr);
          Out << ']';

          GetElementPtrInst *gepPtr = dyn_cast<GetElementPtrInst>(Ptr);
          while (gepPtr) {
            Opnd = gepPtr->getOperand(2);
            Out << "[";
            writeOperandWithCast(Opnd, Instruction::GetElementPtr);
            Out << ']';
            gepPtr = dyn_cast<GetElementPtrInst>(gepPtr->getPointerOperand());
          }
        }
      }
    } else if (!IntoT->isVectorTy()) {
      Out << '[';
      writeOperandWithCast(Opnd, Instruction::GetElementPtr);
      Out << ']';
    } else {
      // If the last index is into a vector, then print it out as "+j)".  This
      // works with the 'LastIndexIsVector' code above.
      if (!isConstantNull(Opnd)) {
        Out << "))"; // avoid "+0".
      } else {
        Out << ")+(";
        writeOperandWithCast(I.getOperand(), Instruction::GetElementPtr);
        Out << "))";
      }
    }

    IntoT = I.getIndexedType();
  }
  //}
  // Out << ")";
}

void CWriter::writeMemoryAccess(Value *Operand, Type *OperandType,
                                bool IsVolatile, unsigned Alignment /*bytes*/) {

  GetElementPtrInst *gepInst = dyn_cast<GetElementPtrInst>(Operand);
  GEPOperator *gepOp = dyn_cast<GEPOperator>(Operand);
  auto isI8PtrType = [](Type *Ty) -> bool {
    if (auto *PTy = dyn_cast<PointerType>(Ty))
      return PTy->getElementType()->isIntegerTy(8);
    return false;
  };
  auto isByteAddressingFromCast = [&](Value *Ptr) -> bool {
    if (!Ptr)
      return false;
    if (auto *CastI = dyn_cast<CastInst>(Ptr))
      return isI8PtrType(CastI->getDestTy()) && !isI8PtrType(CastI->getSrcTy());
    if (auto *CE = dyn_cast<ConstantExpr>(Ptr))
      if (CE->isCast() && CE->getNumOperands() >= 1)
        return isI8PtrType(CE->getType()) &&
               !isI8PtrType(CE->getOperand(0)->getType());
    return false;
  };
  auto isByteCastFromAggregateCarrier = [&](Value *Ptr) -> bool {
    if (!Ptr)
      return false;
    Type *SrcTy = nullptr;
    if (auto *CastI = dyn_cast<CastInst>(Ptr)) {
      if (!isI8PtrType(CastI->getDestTy()))
        return false;
      SrcTy = CastI->getSrcTy();
    } else if (auto *CE = dyn_cast<ConstantExpr>(Ptr)) {
      if (!(CE->isCast() && CE->getNumOperands() >= 1) ||
          !isI8PtrType(CE->getType()))
        return false;
      SrcTy = CE->getOperand(0)->getType();
    } else {
      return false;
    }
    if (auto *SrcPtrTy = dyn_cast_or_null<PointerType>(SrcTy))
      return SrcPtrTy->getElementType()->isAggregateType();
    return false;
  };
  auto isByteCastFromScalarGlobalCarrier = [&](Value *Ptr) -> bool {
    if (!Ptr)
      return false;
    Value *Src = nullptr;
    if (auto *CastI = dyn_cast<CastInst>(Ptr)) {
      if (!isI8PtrType(CastI->getDestTy()))
        return false;
      Src = CastI->getOperand(0);
    } else if (auto *CE = dyn_cast<ConstantExpr>(Ptr)) {
      if (!(CE->isCast() && CE->getNumOperands() >= 1) ||
          !isI8PtrType(CE->getType()))
        return false;
      Src = CE->getOperand(0);
    } else {
      return false;
    }
    auto *GV = dyn_cast_or_null<GlobalVariable>(Src ? Src->stripPointerCasts() : nullptr);
    if (!GV)
      return false;
    Type *ValTy = GV->getValueType();
    return ValTy && !ValTy->isAggregateType();
  };

  if (gepInst) {
    GetElementPtrInst *rootGEP = gepInst;
    bool rootNeedsReference = false;
    errs() << "SUSAN: GEPINST: " << *gepInst << "\n";
    Value *UO = findUnderlyingObject(gepInst->getPointerOperand());
    auto derefIt = (UO ? Times2Dereference.find(UO) : Times2Dereference.end());
    bool hasDerefInfo = derefIt != Times2Dereference.end();
    int dereferenceTimes = hasDerefInfo ? derefIt->second : 0;
    errs() << "SUSAN: dereferenceTimes = " << dereferenceTimes << "\n";
    while (gepInst) {
      if (hasDerefInfo && !dereferenceTimes) {
        GEPNeedsReference.insert(gepInst);
        if (gepInst == rootGEP)
          rootNeedsReference = true;
      }

      accessGEPMemory.insert(gepInst);
      gepInst = dyn_cast<GetElementPtrInst>(gepInst->getPointerOperand());
      if (hasDerefInfo)
        dereferenceTimes--;
    }
    bool rootResultIsAggregate =
        rootGEP->getResultElementType()->isAggregateType();
    bool mustEmitExplicitDeref = false;
    bool mustDerefByteCastLoad = false;
    if (isa<LoadInst>(CurInstr)) {
      // For i8 loads through concrete GEP instructions, the emitted expression
      // is already a byte lvalue/index form. Adding an extra '*' creates
      // invalid code like "*(((uint8_t*)(&x))[i])".
      if (OperandType->isIntegerTy(8))
        mustEmitExplicitDeref = false;
      else
        mustEmitExplicitDeref = rootNeedsReference;
      // Recover scalar memcpy-to-symbol byte loads that print as address
      // arithmetic "(src+idx)" when reference tracking marks the GEP as an
      // address expression. Aggregate carriers must stay on index-lvalue form.
      mustDerefByteCastLoad =
          OperandType->isIntegerTy(8) &&
          isByteAddressingFromCast(rootGEP->getPointerOperand()) &&
          !isByteCastFromAggregateCarrier(rootGEP->getPointerOperand()) &&
          isByteCastFromScalarGlobalCarrier(rootGEP->getPointerOperand());
    } else if (isa<StoreInst>(CurInstr)) {
      // Preserve prior store behavior: only force '*' for aggregate roots.
      mustEmitExplicitDeref = rootNeedsReference && rootResultIsAggregate;
    }
    if (mustEmitExplicitDeref || mustDerefByteCastLoad) {
      // For memory access through GEP we must emit an explicit dereference.
      // Otherwise a GEP that prints as an address expression (e.g. '&field')
      // leaks address syntax into scalar expressions or store LHS.
      Out << "*(";
      writeOperandInternal(Operand);
      Out << ")";
      accessGEPMemory.clear();
      return;
    }

    writeOperandInternal(Operand);
    accessGEPMemory.clear();
    return;
  }

  if (gepOp) {
    // ConstantExpr GEPs do not enter the GetElementPtrInst path above.
    // If we fall through to the generic memory path, it prepends '*' and
    // turns value expressions like "obj.field" into invalid "*(obj.field)".
    // Handle GEPOperator directly to keep load/store scalar field access valid.
    if (isa<StoreInst>(CurInstr)) {
      // Match GetElementPtrInst behavior: emit the GEP expression directly as
      // the store LHS lvalue (e.g. obj.field or arr[idx]), not as *().
      writeOperandInternal(Operand);
      return;
    }
    if (isa<LoadInst>(CurInstr) && OperandType->isIntegerTy(8)) {
      Value *BasePtr = gepOp->getPointerOperand();
      if (isByteAddressingFromCast(BasePtr) &&
          !isByteCastFromAggregateCarrier(BasePtr)) {
        // For scalar cudaMemcpyToSymbol byte copies (e.g. double -> d_dx1),
        // GEPOperator prints address arithmetic "(src+idx)"; materialize byte
        // value here while keeping aggregate carriers on lvalue indexing path.
        Out << "*(";
        writeOperandInternal(Operand);
        Out << ")";
        return;
      }
    }
    writeOperandInternal(Operand);
    return;
  }

  if (isAddressExposed(Operand) && !IsVolatile) {
    writeOperandInternal(Operand);
    return;
  }

  bool IsUnaligned =
      Alignment && Alignment < TD->getABITypeAlignment(OperandType);

  if (!IsUnaligned) {
    Out << '*';
    if (IsVolatile) {
      Out << "(volatile ";
      printTypeName(Out, OperandType, false);
      Out << "*)";
    }
  } else if (IsUnaligned) {
    headerUseUnalignedLoad();
    Out << "__UNALIGNED_LOAD__(";
    printTypeNameUnaligned(Out, OperandType, false);
    if (IsVolatile)
      Out << " volatile";
    Out << ", " << Alignment << ", ";
  }

  writeOperand(Operand);

  if (IsUnaligned) {
    Out << ")";
  }

  // bool IsUnaligned =
  //     Alignment && Alignment < TD->getABITypeAlignment(OperandType);

  // if (!IsUnaligned) {
  //  Out << '*';
  //  if (IsVolatile) {
  //    Out << "(volatile ";
  //    printTypeName(Out, OperandType, false);
  //    Out << "*)";
  //  }
  //}
  /*else if (IsUnaligned) {
    headerUseUnalignedLoad();
    Out << "__UNALIGNED_LOAD__(";
    printTypeNameUnaligned(Out, OperandType, false);
    if (IsVolatile)
      Out << " volatile";
    Out << ", " << Alignment << ", ";
  }*/

  // if (IsUnaligned) {
  //   Out << ")";
  // }
}

void CWriter::visitLoadInst(LoadInst &I) {
  // errs() << "SUSAN: curinstr before loadinst: " << *CurInstr << "\n";
  CurInstr = &I;
  errs() << "SUSAN: loadInst: " << I << "\n";

  if (doubleGeps.find(&I) != doubleGeps.end()) {
    GetElementPtrInst *gep = cast<GetElementPtrInst>(I.getPointerOperand());
    GetElementPtrInst *gep2 = cast<GetElementPtrInst>(gep->getPointerOperand());
    auto ptrVal = gep2->getPointerOperand();

    writeOperand(ptrVal);

    auto firstIdx = gep->getOperand(1);
    Out << "[";
    if (ConstantInt *idx = dyn_cast<ConstantInt>(firstIdx))
      Out << idx->getSExtValue();
    else
      writeOperand(firstIdx);

    auto secondIdx = gep2->getOperand(1);
    Out << "+";
    if (ConstantInt *idx = dyn_cast<ConstantInt>(secondIdx))
      Out << idx->getSExtValue();
    else
      writeOperand(secondIdx);

    Out << "]";
    return;
  }
  // for omp inlining struct
  if (inlinedArgNames.find(&I) != inlinedArgNames.end()) {
    Out << inlinedArgNames[&I];
    if (valuesCast2Double.find(&I) != valuesCast2Double.end())
      Out << "((double*)";
    errs() << "SUSAN: printing inlined name: " << inlinedArgNames[&I];
    if (valuesCast2Double.find(&I) != valuesCast2Double.end())
      Out << ")";
    return;
  }

  // for omp inlining
  if (addressExposedLoads.find(&I) != addressExposedLoads.end()) {
    errs() << "SUSAN: printing inlined load: " << I << "\n";
    // Keep pointer-typed loads in the fast path, but scalar loads must still
    // go through writeMemoryAccess so they print as values, not addresses.
    if (I.getType()->isPointerTy()) {
      errs() << "ANDREW: addressExposedLoads fast-path kept pointer load: " << I
             << "\n";
      return writeOperand(I.getPointerOperand());
    }
    errs() << "ANDREW: addressExposedLoads bypassed for scalar load: " << I
           << "\n";
  }

  errs() << "ANDREW: visitLoadInst using writeMemoryAccess for: " << I << "\n";
  writeMemoryAccess(I.getOperand(0), I.getType(), I.isVolatile(),
                    I.getAlignment());
}

void CWriter::visitStoreInst(StoreInst &I) {
  CurInstr = &I;
  errs() << "CBEBackend: printing store Inst: " << I << "\n";

  AllocaInst *noneSkipAlloca = nullptr;

  auto gep = dyn_cast<GetElementPtrInst>(I.getPointerOperand());
  std::stack<GetElementPtrInst *> geps;
  while (gep) {
    geps.push(gep);
    noneSkipAlloca = dyn_cast<AllocaInst>(gep->getPointerOperand());
    gep = dyn_cast<GetElementPtrInst>(gep->getPointerOperand());
  }

  // if(noneSkipAlloca){
  //   Out << GetValueName(noneSkipAlloca);
  //   while(!geps.empty()){
  //     auto gep = geps.top();
  //     errs() << "SUSAN: printing noneSkipAlloca: " << *gep << "\n";
  //     geps.pop();
  //     Out << '[';
  //     int idx = 0;
  //     std::vector<Value*>vals2write;
  //     for(unsigned int i =1; i!=gep->getNumOperands(); i++){
  //       if(ConstantInt *constInt = dyn_cast<ConstantInt>(gep->getOperand(i)))
  //         idx += constInt->getSExtValue();
  //       if(Instruction *inst = dyn_cast<Instruction>(gep->getOperand(i))){
  //         if(inst->getOpcode() == Instruction::Mul)
  //           vals2write.push_back(inst->getOperand(0));
  //         else
  //           vals2write.push_back(inst);
  //       } else {
  //         vals2write.push_back(gep->getOperand(i));
  //       }
  //     }

  //    writeOperand(vals2write[0]);
  //    for (auto it = begin(vals2write)+1; it != end(vals2write); ++it) {
  //      Out << " + " ;
  //      writeOperand(*it);
  //    }
  //    if(idx)
  //      Out << " + " << idx << ']';
  //    else
  //      Out << ']';
  //  }
  //} else if(doubleGeps.find(&I) != doubleGeps.end()){
  //  GetElementPtrInst *gep = cast<GetElementPtrInst>(I.getPointerOperand());
  //  GetElementPtrInst *gep2 =
  //  cast<GetElementPtrInst>(gep->getPointerOperand()); auto ptrVal =
  //  gep2->getPointerOperand(); Out << GetValueName(ptrVal); auto firstIdx =
  //  gep->getOperand(1); errs() << "SUSAN: gep 10928: " << *gep << "\n"; errs()
  //  << "SUSAN:writing first index: "<< *firstIdx << "\n"; Out << "[";
  //  if(ConstantInt *idx = dyn_cast<ConstantInt>(firstIdx))
  //    Out << idx->getSExtValue();
  // else
  //    writeOperand(firstIdx);

  //  auto secondIdx = gep2->getOperand(1);
  //  Out << "+";
  //  if(ConstantInt *idx = dyn_cast<ConstantInt>(secondIdx))
  //    Out << idx->getSExtValue();
  // else
  //    writeOperand(secondIdx);

  // Out << "]";
  //}
  // else{

  errs() << "CBackend: here? 10442\n";
  writeMemoryAccess(I.getPointerOperand(), I.getOperand(0)->getType(),
                    I.isVolatile(), I.getAlignment());
  errs() << "CBackend: here? 10445\n";
  //}
  Out << " = ";
  Value *Operand = I.getOperand(0);
  unsigned BitMask = 0;
  if (IntegerType *ITy = dyn_cast<IntegerType>(Operand->getType()))
#if LLVM_VERSION_MAJOR <= 10
    if (!ITy->isPowerOf2ByteWidth())
#else
    if (!IsPowerOfTwo(ITy->getBitWidth()))
#endif
      // We have a bit width that doesn't match an even power-of-2 byte
      // size. Consequently we must & the value with the type's bit mask
      BitMask = ITy->getBitMask();
  if (BitMask)
    Out << "((";
  writeOperand(Operand, BitMask ? ContextNormal : ContextCasted);
  errs() << "CBackend: here? 10462\n";
  if (BitMask)
    Out << ") & " << BitMask << ")";
}

void CWriter::visitFenceInst(FenceInst &I) {
  headerUseThreadFence();
  Out << "__atomic_thread_fence(";
  switch (I.getOrdering()) {
  case AtomicOrdering::Acquire:
    Out << "__ATOMIC_ACQUIRE";
    break;
  case AtomicOrdering::Release:
    Out << "__ATOMIC_RELEASE";
    break;
  case AtomicOrdering::AcquireRelease:
    Out << "__ATOMIC_ACQ_REL";
    break;
  case AtomicOrdering::SequentiallyConsistent:
    Out << "__ATOMIC_SEQ_CST";
    break;
  case AtomicOrdering::NotAtomic:
  case AtomicOrdering::Unordered:
  case AtomicOrdering::Monotonic:
    Out << "__ATOMIC_RELAXED";
    break;
  default:
    errorWithMessage("Unhandled atomic ordering for fence instruction");
  }
  Out << ");\n";
}

bool CWriter::GEPAccessesMemory(GetElementPtrInst *I) {
  if (accessGEPMemory.find(I) != accessGEPMemory.end())
    return true;
  SmallPtrSet<Value *, 32> Visited;
  SmallVector<GetElementPtrInst *, 8> SeenGEPs;

  std::function<bool(Value *)> reachesMemory = [&](Value *V) -> bool {
    if (!Visited.insert(V).second)
      return false;

    if (GetElementPtrInst *G = dyn_cast<GetElementPtrInst>(V))
      SeenGEPs.push_back(G);

    for (User *U : V->users()) {
      if (LoadInst *memInst = dyn_cast<LoadInst>(U)) {
        if (memInst->getPointerOperand() == V)
          return true;
      } else if (StoreInst *memInst = dyn_cast<StoreInst>(U)) {
        if (memInst->getPointerOperand() == V)
          return true;
      } else if (GetElementPtrInst *gepInst = dyn_cast<GetElementPtrInst>(U)) {
        if (gepInst->getPointerOperand() == V && reachesMemory(gepInst))
          return true;
      } else if (CastInst *castInst = dyn_cast<CastInst>(U)) {
        if (castInst->getOperand(0) == V && reachesMemory(castInst))
          return true;
      }
    }
    return false;
  };

  if (!reachesMemory(I))
    return false;

  for (GetElementPtrInst *G : SeenGEPs)
    accessGEPMemory.insert(G);
  return true;
}

void CWriter::visitGetElementPtrInst(GetElementPtrInst &I) {
  CurInstr = &I;
  bool accessMemory = false;
  if (accessGEPMemory.find(&I) != accessGEPMemory.end())
    accessMemory = true;
  // Ensure nested GEPs printed through casts are classified before emission.
  // Relying only on the cache can miss first-time visits and fall back to
  // value-style indexing ("[idx]"), which then causes invalid recasts.
  if (!accessMemory)
    accessMemory = GEPAccessesMemory(&I);

  //  bool prevGEPisPointer = false;
  //  if(GetElementPtrInst *prevGEP =
  //  dyn_cast<GetElementPtrInst>(I.getPointerOperand())){
  //    if(GEPPointers.find(prevGEP) != GEPPointers.end())
  //      prevGEPisPointer = true;
  //  }

  errs() << "SUSAN: printing GEP: " << I << "\n";
  bool printReference = false;
  if (GEPNeedsReference.find(&I) != GEPNeedsReference.end())
    printReference = true;

  if (accessMemory)
    errs() << "SUSAN: accessMemory true\n";

  if (printReference)
    errs() << "SUSAN: printReference true\n";

  bool currGEPisPointer =
      printGEPExpressionStruct(I.getPointerOperand(), gep_type_begin(I),
                               gep_type_end(I), accessMemory, printReference);
  if (currGEPisPointer)
    GEPPointers.insert(&I);
}

void CWriter::visitVAArgInst(VAArgInst &I) {
  CurInstr = &I;

  headerUseStdarg();

  Out << "va_arg(*(va_list*)";
  writeOperand(I.getOperand(0), ContextCasted);
  Out << ", ";
  printTypeName(Out, I.getType());
  Out << ");\n ";
}

void CWriter::visitInsertElementInst(InsertElementInst &I) {
  CurInstr = &I;

  // Start by copying the entire aggregate value into the result variable.
  writeOperand(I.getOperand(0));
  Type *EltTy = I.getType()->getElementType();
  cwriter_assert(I.getOperand(1)->getType() == EltTy);
  if (isEmptyType(EltTy))
    return;

  // Then do the insert to update the field.
  Out << ";\n  ";
  Out << GetValueName(&I) << ".vector[";
  writeOperand(I.getOperand(2));
  Out << "] = ";
  writeOperand(I.getOperand(1), ContextCasted);
}

void CWriter::visitExtractElementInst(ExtractElementInst &I) {
  CurInstr = &I;

  cwriter_assert(!isEmptyType(I.getType()));
  if (isa<UndefValue>(I.getOperand(0))) {
    Out << "(";
    printTypeName(Out, I.getType());
    Out << ") 0/*UNDEF*/";
  } else {
    Out << "(";
    writeOperand(I.getOperand(0));
    Out << ").vector[";
    writeOperand(I.getOperand(1));
    Out << "]";
  }
}

// <result> = shufflevector <n x <ty>> <v1>, <n x <ty>> <v2>, <m x i32> <mask>
// ; yields <m x <ty>>
void CWriter::visitShuffleVectorInst(ShuffleVectorInst &SVI) {
  CurInstr = &SVI;

  VectorType *VT = SVI.getType();
  Type *EltTy = VT->getElementType();
  VectorType *InputVT = cast<VectorType>(SVI.getOperand(0)->getType());
  cwriter_assert(!isEmptyType(VT));
  cwriter_assert(InputVT->getElementType() == VT->getElementType());

  CtorDeclTypes.insert(VT);
  Out << "llvm_ctor_";
  printTypeString(Out, VT, false);
  Out << "(";

  Constant *Zero = Constant::getNullValue(EltTy);
  unsigned NumElts = NumberOfElements(VT);
  unsigned NumInputElts = NumberOfElements(InputVT); // n
  for (unsigned i = 0; i != NumElts; ++i) {
    if (i)
      Out << ", ";
    int SrcVal = SVI.getMaskValue(i);
    if ((unsigned)SrcVal >= NumInputElts * 2) {
      Out << "/*undef*/";
      printConstant(Zero, ContextCasted);
    } else {
      // If SrcVal belongs [0, n - 1], it extracts value from <v1>
      // If SrcVal belongs [n, 2 * n - 1], it extracts value from <v2>
      // In C++, the value false is converted to zero and the value true is
      // converted to one
      Value *Op = SVI.getOperand((unsigned)SrcVal >= NumInputElts);
      if (isa<Instruction>(Op)) {
        // Do an extractelement of this value from the appropriate input.
        Out << "(";
        writeOperand(Op);
        Out << ").vector[";
        Out << ((unsigned)SrcVal >= NumInputElts ? SrcVal - NumInputElts
                                                 : SrcVal);
        Out << "]";
      } else if (isa<ConstantAggregateZero>(Op) || isa<UndefValue>(Op)) {
        printConstant(Zero, ContextCasted);
      } else {
        printConstant(
            cast<ConstantVector>(Op)->getOperand(SrcVal & (NumElts - 1)),
            ContextNormal);
      }
    }
  }
  Out << ")";
}

void CWriter::visitInsertValueInst(InsertValueInst &IVI) {
  CurInstr = &IVI;

  // Start by copying the entire aggregate value into the result variable.
  writeOperand(IVI.getOperand(0));
  Type *EltTy = IVI.getOperand(1)->getType();
  if (isEmptyType(EltTy))
    return;

  // Then do the insert to update the field.
  Out << ";\n  ";
  Out << GetValueName(&IVI);
  for (const unsigned *b = IVI.idx_begin(), *i = b, *e = IVI.idx_end(); i != e;
       ++i) {
    Type *IndexedTy = ExtractValueInst::getIndexedType(
        IVI.getOperand(0)->getType(), makeArrayRef(b, i));
    cwriter_assert(IndexedTy);
    if (IndexedTy->isArrayTy()) {
      Out << ".array[" << *i << "]";
    } else {
      cwriter_assert(isa<StructType>(IndexedTy));
      Out << "." << getFieldName(cast<StructType>(IndexedTy)) << *i;
    }
  }
  Out << " = ";
  writeOperand(IVI.getOperand(1), ContextCasted);
}

void CWriter::visitExtractValueInst(ExtractValueInst &EVI) {
  CurInstr = &EVI;

  Out << "(";
  if (isa<UndefValue>(EVI.getOperand(0))) {
    Out << "(";
    printTypeName(Out, EVI.getType());
    Out << ") 0/*UNDEF*/";
  } else {
    writeOperand(EVI.getOperand(0));
    for (const unsigned *b = EVI.idx_begin(), *i = b, *e = EVI.idx_end();
         i != e; ++i) {
      Type *IndexedTy = ExtractValueInst::getIndexedType(
          EVI.getOperand(0)->getType(), makeArrayRef(b, i));
      if (IndexedTy->isArrayTy()) {
        Out << ".array[" << *i << "]";
      } else {
        cwriter_assert(isa<StructType>(IndexedTy));
        Out << "." << getFieldName(cast<StructType>(IndexedTy)) << *i;
      }
    }
  }
  Out << ")";
}

LLVM_ATTRIBUTE_NORETURN void CWriter::errorWithMessage(const char *message) {
#ifndef NDEBUG
  errs() << message;
  errs() << " in: ";
  if (CurInstr != nullptr) {
    errs() << *CurInstr << " @ ";
    CurInstr->getDebugLoc().print(errs());
  } else {
    errs() << "<unknown instruction>";
  }
  errs() << "\n";
#endif

  llvm_unreachable(message);
}

} // namespace llvm_cbe
