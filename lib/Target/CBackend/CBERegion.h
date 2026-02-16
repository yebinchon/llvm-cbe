#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/CodeGen/IntrinsicLowering.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Attributes.h"
#include "llvm/IR/CFG.h"
#if LLVM_VERSION_MAJOR > 10
#include "llvm/IR/AbstractCallSite.h"
#else
#include "llvm/IR/CallSite.h"
#endif
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/GetElementPtrTypeIterator.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstVisitor.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/MCSymbol.h"
#include "llvm/Pass.h"
#include "llvm/Support/FormattedStream.h"
#include "llvm/Transforms/Scalar.h"

#include <set>

// SUSAN ADDED LIBS
#include "llvm/Analysis/PostDominators.h"
#include "llvm/IR/Dominators.h"
#include "llvm/Analysis/RegionInfo.h"
#include "llvm/Analysis/ScalarEvolution.h"
#include <queue>

using namespace llvm;

namespace llvm_cbe {

enum LoopType {
  forLoop,
  whileLoop,
  whileLoopWithContinue,  // while loop with continue statements (outer/inner pattern)
  doWhileLoop,
  unknown
};

class CWriter;
class CBERegion2;
class LoopRegion;
class CBERegion2 {
  public:
  virtual ~CBERegion2(){};
  BasicBlock *getNextEntryBB(){
    return nextEntryBB;
  };
  CBERegion2 (LoopInfo* LI, PostDominatorTree *PDT, DominatorTree* DT, CWriter *cwriter)
    : LI(LI),
      PDT(PDT),
      DT(DT),
      cw(cwriter),
      entryBlock(nullptr),
      nextEntryBB(nullptr),
      parentRegion(nullptr){};

  CBERegion2 (LoopInfo* LI, PostDominatorTree *PDT, DominatorTree *DT, CBERegion2 *parentR, BasicBlock *entryBB, CWriter *cwriter)
    : LI(LI),
      PDT(PDT),
      DT(DT),
      entryBlock(entryBB),
      parentRegion(parentR),
      cw(cwriter){};

  void createCBERegionDAG(BasicBlock *entryBB, CBERegion2 *parentR, BasicBlock *endBB);
  virtual bool isaLoopRegion() {return false;};
  virtual bool isaLinearRegion() {return false;};
  virtual bool isaIfElseRegion() {return false;};
  CBERegion2* getParentRegion(){return parentRegion;};
  virtual void print();
  BasicBlock *getEntryBlock(){return entryBlock;};
  virtual void printRegionDAG();
  virtual bool containsBlock(BasicBlock *BB) { return entryBlock == BB; }

  LoopRegion* getParentLoopRegion(){
    auto ancestorR = parentRegion;
    while(ancestorR){
      if(ancestorR->isaLoopRegion())
        break;
      ancestorR = ancestorR->getParentRegion();
    }
    return (LoopRegion*)ancestorR;
  }
  LoopRegion* getContainingLoopRegion(BasicBlock *BB);

  int whichRegion(BasicBlock *entryBB, LoopInfo *LI);

  protected:
  BasicBlock *entryBlock; //entryBlock itself should belong to parent region
  BasicBlock *nextEntryBB;
  CBERegion2 *parentRegion;
  //CBERegion2 *topRegion;
  LoopInfo *LI;
  PostDominatorTree *PDT;
  DominatorTree *DT;
  CBERegion2* createSubRegions(CBERegion2* parentR, BasicBlock* entryBB, BasicBlock* endBB = nullptr);
  CWriter *cw;

  private:
  std::vector<CBERegion2*>CBERegionDAG;
  std::set<BasicBlock*> regionBuildVisited;
  std::set<std::pair<BasicBlock*, BasicBlock*>> activeCreateSubregions;
};

class LinearRegion : public CBERegion2{
  public:
  LinearRegion(BasicBlock *entryBB, CBERegion2 *parentR, LoopInfo *LI, PostDominatorTree *PDT, DominatorTree *DT, CWriter *cwriter, BasicBlock *endBB);
  bool isaLoopRegion() override {return false;};
  bool isaLinearRegion() override {return true;};
  bool isaIfElseRegion() override {return false;};
  void print() override;
  void printRegionDAG() override;
  bool containsBlock(BasicBlock *BB) override;

  private:
  std::vector<BasicBlock*> BBs;
};

class LoopRegion : public CBERegion2{
  public:
  virtual ~LoopRegion() = default;
  LoopRegion (BasicBlock *entryBB, LoopInfo *LI, PostDominatorTree* PDT, DominatorTree *DT, CBERegion2 *parentR, CWriter *cwriter);
  bool isaLoopRegion() override {return true;};
  bool isaLinearRegion() override {return false;};
  bool isaIfElseRegion() override {return false;};
  void print() override;
  void printRegionDAG() override;
  bool containsBlock(BasicBlock *BB) override;

  Loop* getLoop(){ return loop; }
  void addBBToVisit(BasicBlock* bb){
    remainingBBsToVisit.insert(bb);
  }
  void removeBBToVisit(BasicBlock* bb){
    remainingBBsToVisit.erase(bb);
  }
  bool hasNoRemainingBBs(){
    return remainingBBsToVisit.empty();
  }

  void createCBERegionDAG(BasicBlock *entryBB);

  void printDoWhileLoop();
  void printWhileLoop();
  void printWhileLoopWithContinue();
  void printForLoop();

  BasicBlock *getConditionBlock() const { return conditionBlock; }

  private:
  Loop *loop;
  CBERegion2 *parentRegion;
  std::vector<CBERegion2*>LoopBodyRegionDAG;
  std::set<BasicBlock*> remainingBBsToVisit;
  BasicBlock *latchBB;
  BasicBlock *conditionBlock = nullptr;  // For whileLoopWithContinue: the inner condition block
  Value *ub, *lb, *incr;
  PHINode *IV;
  Instruction *IVInc;
  int nestlevel;
  LoopType loopType;
};

class IfElseRegion : public CBERegion2 {
  public:
  virtual ~IfElseRegion() = default;
  IfElseRegion (BasicBlock *entryBB, CBERegion2 *parentR, PostDominatorTree *PDT, DominatorTree *DT, LoopInfo* LI, CWriter *cwriter);
  bool isaLoopRegion() override {return false;};
  bool isaLinearRegion() override {return false;};
  bool isaIfElseRegion() override {return true;};
  void print() override;
  void printRegionDAG() override;
  bool containsBlock(BasicBlock *BB) override;

  private:
  BasicBlock* createSubIfElseRegions(BasicBlock* start, BasicBlock *brBlock, BasicBlock *stopBB, bool isElseBranch = false);
  void removeIfElseBlockFromLR(LoopRegion* lr, BasicBlock *brBB);

  int dominatedByReturn(BasicBlock* brBB){
    Function *F = brBB->getParent();
    auto br = dyn_cast<BranchInst>(brBB->getTerminator());
    if(br->isConditional()){
      auto succ0 = br->getSuccessor(0);
      auto succ1 = br->getSuccessor(1);
      auto singleSucc = succ0->getSingleSuccessor();
      if(singleSucc && isa<ReturnInst>(singleSucc->getTerminator()))
        return 0;
      singleSucc = succ1->getSingleSuccessor();
      if(singleSucc && isa<ReturnInst>(singleSucc->getTerminator()))
        return 1;
    }
    return -1;
  }

  BasicBlock* isExitingFunction(BasicBlock* bb){
    Instruction *term = bb->getTerminator();
    if(isa<ReturnInst>(term))
      return bb;

    if(term->getNumSuccessors() > 1)
      return nullptr;

    if(isa<UnreachableInst>(term))
      return bb;

    BasicBlock *succ = term->getSuccessor(0);
    Instruction *ret = succ->getTerminator();

    if(isa<ReturnInst>(ret)) return succ;
    
    return nullptr;
  }

  // Check if a basic block is outside the parent loop (for break detection)
  bool isExitingLoop(BasicBlock* bb, BasicBlock *from = nullptr) {
    LoopRegion *lr = getContainingLoopRegion(from ? from : bb);
    if (!lr) return false;
    Loop *L = lr->getLoop();
    return !L->contains(bb);
  }

  // Check whether an edge represents the canonical loop-exit path.
  // This is used to avoid classifying normal loop termination as break.
  bool isCanonicalLoopExitBranch(BasicBlock *from, BasicBlock *to) {
    LoopRegion *lr = getContainingLoopRegion(from);
    if (!lr || !from || !to) return false;

    Loop *L = lr->getLoop();
    if (L->contains(to)) return false;

    BasicBlock *loopExit = lr->getNextEntryBB();
    if (loopExit && to != loopExit) return false;

    BasicBlock *header = L->getHeader();
    BasicBlock *latch = L->getLoopLatch();
    BasicBlock *condition = lr->getConditionBlock();
    return from == header || from == latch ||
           (condition && from == condition);
  }

  // Check if a basic block branches back to the loop header/condition
  // (for continue detection in while loops with continue)
  bool isTrivialContinueBlock(BasicBlock *bb) {
    if (!bb)
      return false;
    Instruction *firstReal = bb->getFirstNonPHIOrDbgOrLifetime();
    return firstReal && firstReal == bb->getTerminator();
  }

  bool isContinuingLoop(BasicBlock* bb, BasicBlock *from = nullptr) {
    BasicBlock *anchor = from ? from : bb;
    LoopRegion *lr = getContainingLoopRegion(anchor);
    if (!lr) return false;
    Loop *L = lr->getLoop();
    // Continue edges leave the current block and jump back to the active loop.
    // Anchor checks avoid resolving against an unrelated inner loop.
    if (!L->contains(anchor)) return false;
    // Check if bb branches to the header, latch, or condition block
    BranchInst *br = dyn_cast<BranchInst>(bb->getTerminator());
    if (!br) return false;
    BasicBlock *header = L->getHeader();
    BasicBlock *latch = L->getLoopLatch();
    BasicBlock *condition = lr->getConditionBlock();
    for (unsigned i = 0; i < br->getNumSuccessors(); i++) {
      BasicBlock *succ = br->getSuccessor(i);
      // For whileLoopWithContinue, continue should target the condition path,
      // not the outer header/latch progression block.
      if (condition) {
        if (succ == condition && isTrivialContinueBlock(bb))
          return true;
        if (succ == header || succ == latch) continue;
        if (L->contains(succ)) {
          BranchInst *succBr = dyn_cast<BranchInst>(succ->getTerminator());
          if (succBr && succBr->isUnconditional() &&
              succBr->getSuccessor(0) == condition &&
              succ != header && succ != latch &&
              isTrivialContinueBlock(bb) &&
              isTrivialContinueBlock(succ))
            return true;
        }
        continue;
      }
      if ((succ == header || succ == latch) && isTrivialContinueBlock(bb))
        return true;
      // Also check if succ is a single-block that just branches to header/latch.
      if (L->contains(succ)) {
        BranchInst *succBr = dyn_cast<BranchInst>(succ->getTerminator());
        if (succBr && succBr->isUnconditional()) {
          BasicBlock *target = succBr->getSuccessor(0);
          if ((target == header || target == latch) &&
              isTrivialContinueBlock(bb) &&
              isTrivialContinueBlock(succ))
            return true;
        }
      }
    }
    return false;
  }

  bool noElseRegion(bool trueBranch) {
    std::set<BasicBlock *> branchBBs = trueBranch ? falseBBs : trueBBs;
    bool NoElseRegion = true;
    for (auto bb : branchBBs) {
      for (auto &I : *bb) {
        // Treat short-circuit CFG blocks as "branch-only" if they only contain
        // compare/phi/debug plumbing plus the branch.
        if (isa<BranchInst>(&I) || isa<CmpInst>(&I) || isa<PHINode>(&I) ||
            isa<DbgInfoIntrinsic>(&I))
          continue;
        if (!isa<BranchInst>(&I)) {
          NoElseRegion = false;
          errs() << "No Else Region is false\n";
          break;
        }
      }
    }
    return NoElseRegion;
  }

  std::vector<CBERegion2*> thenSubRegions;
  std::vector<CBERegion2*> elseSubRegions;
  BasicBlock* brBB;
  BranchInst* brInst;
  BasicBlock* pdBB;
  BasicBlock* trueStartBB;
  BasicBlock* falseStartBB;
  std::set<BasicBlock*> trueBBs;
  std::set<BasicBlock*> falseBBs;
  std::set<BasicBlock*> claimedSubregionEntries;
  
  // For loop break detection
  bool isLoopBreak = false;
  bool negateCond = false;
  
  // For loop continue detection
  bool isLoopContinue = false;
  
  // For function return detection
  bool isFunctionReturn = false;
};

}
