#include "CBackend.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/Host.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/TargetRegistry.h"

#include "TopologicalSorter.h"

#include <algorithm>
#include <cstdio>

#include <iostream>

// SUSAN: added libs
#include "llvm/Analysis/CFG.h"
#include "llvm/Analysis/ScalarEvolutionExpressions.h"
#include "llvm/Transforms/Utils/Cloning.h"

namespace llvm_cbe {

using namespace llvm;

namespace {

static bool reachesLoopExitFrom(BasicBlock *start, BasicBlock *loopExit,
                                Loop *loop) {
  if (!start || !loopExit || !loop)
    return false;

  SmallVector<BasicBlock *, 16> worklist;
  SmallPtrSet<BasicBlock *, 32> visited;
  worklist.push_back(start);
  while (!worklist.empty()) {
    BasicBlock *curr = worklist.pop_back_val();
    if (curr == loopExit)
      return true;
    if (!visited.insert(curr).second)
      continue;

    for (BasicBlock *succ : successors(curr)) {
      if (succ == loopExit)
        return true;
      if (loop->contains(succ))
        worklist.push_back(succ);
    }
  }
  return false;
}

static bool collectCondInputsInBlock(Value *Root, BasicBlock *BB,
                                     SmallPtrSetImpl<Instruction *> &CondInsts) {
  SmallVector<Value *, 16> worklist;
  SmallPtrSet<Value *, 32> visited;
  worklist.push_back(Root);

  while (!worklist.empty()) {
    Value *V = worklist.pop_back_val();
    if (!visited.insert(V).second)
      continue;

    auto *I = dyn_cast<Instruction>(V);
    if (!I)
      continue;

    // Values computed outside this block are fine; they do not affect whether
    // this block is condition-only.
    if (I->getParent() != BB)
      continue;

    CondInsts.insert(I);
    for (Value *Op : I->operands())
      worklist.push_back(Op);
  }
  return true;
}

static bool isConditionOnlyBranchBlock(BasicBlock *BB, Value *&CondValue) {
  CondValue = nullptr;
  if (!BB)
    return false;

  auto *Br = dyn_cast<BranchInst>(BB->getTerminator());
  if (!Br || !Br->isConditional())
    return false;

  CondValue = Br->getCondition();
  SmallPtrSet<Instruction *, 16> condInsts;
  collectCondInputsInBlock(CondValue, BB, condInsts);

  for (auto &I : *BB) {
    if (&I == BB->getTerminator())
      break;
    if (isa<DbgInfoIntrinsic>(&I) || isa<PHINode>(&I))
      continue;

    // Any non-debug/non-phi instruction must contribute directly to the branch
    // condition graph; unrelated work means this is not a pure condition block.
    if (!condInsts.count(&I))
      return false;

    // Allow compare nodes and side-effect-free arithmetic/cast/select used to
    // compute the branch predicate. Reject anything with side effects.
    if (isa<CmpInst>(&I))
      continue;
    if (!isSafeToSpeculativelyExecute(&I))
      return false;
  }
  return CondValue != nullptr;
}

} // namespace

LoopRegion *CBERegion2::getContainingLoopRegion(BasicBlock *BB) {
  LoopRegion *fallback = nullptr;
  CBERegion2 *ancestorR = parentRegion;
  while (ancestorR) {
    if (ancestorR->isaLoopRegion()) {
      auto *lr = static_cast<LoopRegion *>(ancestorR);
      if (!fallback)
        fallback = lr;
      if (BB && lr->getLoop() && lr->getLoop()->contains(BB))
        return lr;
    }
    ancestorR = ancestorR->getParentRegion();
  }
  return fallback;
}

LinearRegion::LinearRegion(BasicBlock *entryBB, CBERegion2 *parentR,
                           LoopInfo *LI, PostDominatorTree *PDT,
                           DominatorTree *DT, CWriter *cwriter, BasicBlock* endBB)
    : CBERegion2{LI, PDT, DT, parentR, entryBB, cwriter} {
  /*for return block*/
  Instruction *term = entryBB->getTerminator();
  if (isa<ReturnInst>(term) || isa<UnreachableInst>(term)) {
    BBs.push_back(entryBB);
    nextEntryBB = nullptr;
    return;
  }

  BasicBlock *nextBB = entryBB;
  Loop *l = LI->getLoopFor(entryBB);
  LoopRegion *lr = getParentLoopRegion();
  BranchInst* entryBranch = dyn_cast<BranchInst>(entryBB->getTerminator());
  // This is a special case of do-while loop
  if(l && entryBB == l->getLoopLatch() && entryBranch->isConditional()) {
    BBs.push_back(entryBB);
    nextEntryBB = l->getExitBlock();
    return;
  }

  while (true) {
    // Do not let a linear region in a loop absorb outer-scope merge blocks.
    // This prevents break-only paths from pulling in shared loop-exit code.
    if (lr && nextBB != entryBB && !lr->getLoop()->contains(nextBB)) {
      errs() << "ANDREW: stopping linear region at loop boundary before "
             << nextBB->getName() << "\n";
      nextEntryBB = nextBB;
      break;
    }

    if (lr)
      lr->removeBBToVisit(nextBB);
    BBs.push_back(nextBB);
    errs() << "CBERegion: including " << nextBB->getName()
           << " in a linear region\n";
    nextBB = nextBB->getSingleSuccessor();
    if(nextBB)
      errs() << "Now looking at " << nextBB->getName() << "\n";

    nextEntryBB = nextBB;
    if (!nextBB || nextBB->getSingleSuccessor() == nullptr)
      break;
    Loop *lNextBB = LI->getLoopFor(nextBB);
    if(lNextBB) {
      errs() << lNextBB->getHeader()->getName() << " is the header\n";
      errs() << lNextBB->getLoopLatch()->getName() << " is the latch\n";
    }
    if (lNextBB && lNextBB->getLoopLatch() == nextBB)
      break;
    if(lNextBB && lNextBB->getHeader() == nextBB)
      break;
    if(nextBB == endBB)
      break;
  }
}

IfElseRegion::IfElseRegion(BasicBlock *entryBB, CBERegion2 *parentR,
                           PostDominatorTree *PDT, DominatorTree *DT,
                           LoopInfo *LI, CWriter *cwriter)
    : CBERegion2{LI, PDT, DT, parentR, entryBB, cwriter} {
  /*fetch branch related infos*/
  this->brBB = entryBB;

  errs() << "ANDREW: entryBB name:" << entryBB->getName() << '\n';
  this->brInst = dyn_cast<BranchInst>(entryBB->getTerminator());
  assert(this->brInst && "terminator is not branch inst 600!\n");
  BranchInst *br = dyn_cast<BranchInst>(entryBB->getTerminator());
  assert(br && "not a branch inst to start if else region\n");
  this->trueStartBB = br->getSuccessor(0);
  this->falseStartBB = br->getSuccessor(1);

  // YEBIN: use LLVM IDom directly
  this->pdBB = PDT->getNode(brBB)->getIDom()->getBlock();
  if(this->pdBB) errs() << this->pdBB->getName() << "\n";

  bool trueBrOnly = false;
  bool falseBrOnly = false;

  // Control flow exits in if-else block
  // Assume two ways of exiting: a return statement as the terminator
  // OR going to the return block (this block must only have a return inst)
  // BE CAREFUL OF: if-else statements that happen at end of function
  // FIXME: this will only work for single level if-else statements
  // Need more sophisticated return checking logic to handle nested statements
  // Nested statements may require a bottom-up approach
  BasicBlock *trueExitBB = isExitingFunction(trueStartBB);
  BasicBlock *falseExitBB = isExitingFunction(falseStartBB);
  bool exitFunctionTrueBr = (trueExitBB != nullptr);
  bool exitFunctionFalseBr = (falseExitBB != nullptr);
  // A branch target that is exactly this if-region's postdom merge is not an
  // early return path; it is normal structured fallthrough toward function end.
  // Also treat "branch -> ... -> pdBB(return)" as merge fallthrough.
  bool trueBranchIsMerge =
      (pdBB && (trueStartBB == pdBB || trueExitBB == pdBB));
  bool falseBranchIsMerge =
      (pdBB && (falseStartBB == pdBB || falseExitBB == pdBB));
  bool trueEarlyExit = exitFunctionTrueBr && !trueBranchIsMerge;
  bool falseEarlyExit = exitFunctionFalseBr && !falseBranchIsMerge;
  // Four possible control flows: both exit, only one exits
  // Both exits is likely at the end of the program - NOT an early return pattern
  if (exitFunctionTrueBr && exitFunctionFalseBr) {
    errs() << "Both branches exit function!! (normal end-of-function if-else, not early return)\n";
    // ANDREW: Fall through to normal if-else handling - this is NOT an early return
    // This happens when we have code like:
    //   if (cond) { x = 1; } else { x = 2; }
    //   return;
    // Both branches lead to the same return, but neither is an "early" return
  }
  // These are easier - the one that exits is an "early exit" and the fall-through continues the rest of the program.
  // Don't use the calculated postdominator!!
  // ANDREW: Only trigger early return when ONE branch exits, not both
  else if (trueEarlyExit && !falseEarlyExit) {
    errs() << "True branch exits function!!\n";
    // ANDREW: Just emit return; statement, don't process subregions
    isFunctionReturn = true;
    negateCond = false;
    nextEntryBB = falseStartBB;
    return;
  }
  else if (falseEarlyExit && !trueEarlyExit) {
    errs() << "False branch exits function!!\n";
    // ANDREW: Just emit return; statement, negate condition
    isFunctionReturn = true;
    negateCond = true;
    nextEntryBB = trueStartBB;
    return;
  }
  // ANDREW: Check if either branch exits the parent loop (break statement)
  else if (getContainingLoopRegion(brBB)) {
    bool exitLoopTrueBr = isExitingLoop(trueStartBB, brBB) &&
                          !isCanonicalLoopExitBranch(brBB, trueStartBB);
    bool exitLoopFalseBr = isExitingLoop(falseStartBB, brBB) &&
                           !isCanonicalLoopExitBranch(brBB, falseStartBB);
    
    if (exitLoopTrueBr && !exitLoopFalseBr) {
      errs() << "ANDREW: True branch exits loop (break)!\n";
      isLoopBreak = true;
      negateCond = false;
      nextEntryBB = falseStartBB;
      // Mark the exit block as visited so we don't process it
      if (auto *lr = getContainingLoopRegion(brBB)) {
        lr->removeBBToVisit(trueStartBB);
        
        // ANDREW: If there are instructions between the branch and the loop exit (e.g. an inner loop),
        // we must process them as subregions so they get printed before the break.
        BasicBlock *loopExit = lr->getNextEntryBB();
        if (trueStartBB != loopExit) {
           // Create subregions for the Then branch until we hit the loop exit
           errs() << "ANDREW: Generating subregions for non-trivial break block " << trueStartBB->getName() << "\n";
           createSubIfElseRegions(trueStartBB, brBB, loopExit, false);
        }
      }
      return;  // Don't create subregions (standard flow), will emit break in printRegionDAG
    } else if (exitLoopFalseBr && !exitLoopTrueBr) {
      errs() << "ANDREW: False branch exits loop (break)!\n";
      isLoopBreak = true;
      negateCond = true;  // Negate condition so we have if(!cond) break;
      nextEntryBB = trueStartBB;
      // Mark the exit block as visited so we don't process it
      if (auto *lr = getContainingLoopRegion(brBB)) {
        lr->removeBBToVisit(falseStartBB);

        // ANDREW: If there are instructions between the branch and the loop exit
        BasicBlock *loopExit = lr->getNextEntryBB();
        if (falseStartBB != loopExit) {
           // Create subregions for the Else branch (stored in thenSubRegions because we negated condition)
           errs() << "ANDREW: Generating subregions for non-trivial break block " << falseStartBB->getName() << "\n";
           // Note: We use 'false' for isElseBranch because we are putting it in the "if (!cond)" block, which acts as the Then block
           createSubIfElseRegions(falseStartBB, brBB, loopExit, false);
        }
      }
      return;
    }
    
    // ANDREW: Check if either branch continues the loop (continue statement)
    bool continueLoopTrueBr = isContinuingLoop(trueStartBB, brBB);
    bool continueLoopFalseBr = isContinuingLoop(falseStartBB, brBB);
    
    if (continueLoopTrueBr && !continueLoopFalseBr) {
      errs() << "ANDREW: True branch continues loop (continue)!\n";
      isLoopContinue = true;
      negateCond = false;
      nextEntryBB = falseStartBB;
      // Mark the continue blocks as visited
      if (auto *lr = getContainingLoopRegion(brBB)) {
        lr->removeBBToVisit(trueStartBB);
        // Also remove backedge blocks that are just branches
        BranchInst *tbr = dyn_cast<BranchInst>(trueStartBB->getTerminator());
        if (tbr && tbr->isUnconditional()) {
          lr->removeBBToVisit(tbr->getSuccessor(0));
        }
      }
      return;
    } else if (continueLoopFalseBr && !continueLoopTrueBr) {
      errs() << "ANDREW: False branch continues loop (continue)!\n";
      isLoopContinue = true;
      negateCond = true;
      nextEntryBB = trueStartBB;
      if (auto *lr = getContainingLoopRegion(brBB)) {
        lr->removeBBToVisit(falseStartBB);
        BranchInst *fbr = dyn_cast<BranchInst>(falseStartBB->getTerminator());
        if (fbr && fbr->isUnconditional()) {
          lr->removeBBToVisit(fbr->getSuccessor(0));
        }
      }
      return;
    }
    // Fall through to normal if-else handling
  }

  // Normal if-else handling
  {

  for (auto &BB : *(brBB->getParent())) {
    if (DT->dominates(trueStartBB, &BB) && PDT->dominates(pdBB, &BB) &&
        pdBB != &BB)
      trueBBs.insert(&BB);
    if (DT->dominates(falseStartBB, &BB) && PDT->dominates(pdBB, &BB) &&
        pdBB != &BB)
      falseBBs.insert(&BB);
  }
  // checking if the other branch section has branches???
  trueBrOnly = noElseRegion(true);
  falseBrOnly = noElseRegion(false);
  bool forcedByShortCircuitTail = false;
  // Short-circuit chains (e.g. a&&b&&c) create nested if-regions where the
  // fail-tail successor is shared with the parent region. In that case, the
  // shared tail must be owned once by the outer region; forcing both branches
  // at each nested level causes recursive re-expansion and can diverge.
  if (parentRegion && parentRegion->isaIfElseRegion()) {
    auto *parentIf = static_cast<IfElseRegion *>(parentRegion);
    if (falseStartBB == parentIf->falseStartBB && trueStartBB != falseStartBB) {
      errs() << "ANDREW: forcing true-branch-only due shared false short-circuit tail in "
             << brBB->getName() << "\n";
      trueBrOnly = true;
      falseBrOnly = false;
      forcedByShortCircuitTail = true;
    } else if (trueStartBB == parentIf->trueStartBB &&
               falseStartBB != trueStartBB) {
      errs() << "ANDREW: forcing false-branch-only due shared true short-circuit tail in "
             << brBB->getName() << "\n";
      trueBrOnly = false;
      falseBrOnly = true;
      forcedByShortCircuitTail = true;
    }
  }
  // -1: neither branch leads to return
  // 0: true branch leads to return
  // 1: false branch leads to return
  // This is weird... only works for very particular control flow
  int returnDominated = dominatedByReturn(brBB);
  errs() << "CBERegion_DEBUG: initial branch-only flags in "
         << brBB->getParent()->getName() << "::" << brBB->getName()
         << " trueBrOnly=" << trueBrOnly
         << " falseBrOnly=" << falseBrOnly
         << " returnDominated=" << returnDominated << "\n";

  // If the merge block carries PHI values from both successors, keep explicit
  // if/else structure to preserve per-branch assignments and avoid hoisting one
  // path outside the conditional.
  bool needsBothBranchesForPhi = false;
  if (pdBB) {
    for (auto &I : *pdBB) {
      PHINode *phi = dyn_cast<PHINode>(&I);
      if (!phi)
        break;
      int trueIdx = phi->getBasicBlockIndex(trueStartBB);
      int falseIdx = phi->getBasicBlockIndex(falseStartBB);
      if (trueIdx >= 0 && falseIdx >= 0) {
        needsBothBranchesForPhi = true;
        break;
      }
    }
  }
  if (!forcedByShortCircuitTail && needsBothBranchesForPhi) {
    errs() << "ANDREW: forcing both branches due to merge PHI dependencies in "
           << brBB->getName() << "\n";
    trueBrOnly = false;
    falseBrOnly = false;
  }

  if (!forcedByShortCircuitTail && !needsBothBranchesForPhi &&
      !trueBrOnly && !falseBrOnly &&
      returnDominated == -1) {
    trueBrOnly = (trueEarlyExit && !falseEarlyExit);
    falseBrOnly = (falseEarlyExit && !trueEarlyExit);
  }

  auto blockHasNonTrivialWork = [](BasicBlock *BB) -> bool {
    if (!BB)
      return false;
    for (auto &I : *BB) {
      if (&I == BB->getTerminator())
        break;
      if (isa<DbgInfoIntrinsic>(&I))
        continue;
      // Treat only side-effecting instructions as "real work" for deciding
      // whether we must keep an explicit else branch.
      if (I.mayHaveSideEffects())
        return true;
    }
    return false;
  };

  auto pathHasNonTrivialWork = [&](BasicBlock *start, BasicBlock *stop) -> bool {
    if (!start || start == stop)
      return false;

    // Explore the whole branch subgraph (until the merge) instead of only
    // unconditional chains. Otherwise we can miss side effects behind a
    // conditional block (e.g. if.end588 -> if.then591 -> printf in FT verify).
    SmallVector<BasicBlock *, 32> worklist;
    SmallPtrSet<BasicBlock *, 32> visited;
    worklist.push_back(start);

    while (!worklist.empty()) {
      BasicBlock *curr = worklist.pop_back_val();
      if (!curr || curr == stop)
        continue;
      if (!visited.insert(curr).second)
        continue;
      if (blockHasNonTrivialWork(curr))
        return true;
      for (BasicBlock *succ : successors(curr)) {
        if (succ && succ != stop)
          worklist.push_back(succ);
      }
    }
    return false;
  };

  // Single-branch optimization is only valid when the opposite path is a
  // pure trampoline to the merge. If it carries work, emit explicit else.
  if (!forcedByShortCircuitTail &&
      trueBrOnly && pathHasNonTrivialWork(falseStartBB, pdBB)) {
    errs() << "ANDREW: forcing both branches because false path has work in "
           << brBB->getName() << "\n";
    trueBrOnly = false;
  }
  if (!forcedByShortCircuitTail &&
      falseBrOnly && pathHasNonTrivialWork(trueStartBB, pdBB)) {
    errs() << "ANDREW: forcing both branches because true path has work in "
           << brBB->getName() << "\n";
    falseBrOnly = false;
  }

  errs() << "CBERegion_DEBUG: final branch-only flags in "
         << brBB->getParent()->getName() << "::" << brBB->getName()
         << " trueBrOnly=" << trueBrOnly
         << " falseBrOnly=" << falseBrOnly << "\n";

  if (trueBrOnly && (returnDominated == -1)) {
    errs() << "SUSAN: marking only true branch\n";
    if (auto lr = getParentLoopRegion())
      for (auto BB : falseBBs)
        lr->removeBBToVisit(BB);
    BasicBlock *trueOnlyStopBB = falseStartBB;
    if (forcedByShortCircuitTail && pdBB)
      trueOnlyStopBB = pdBB;
    errs() << "CBERegion_DEBUG: true-only stop boundary in "
           << brBB->getParent()->getName() << "::" << brBB->getName()
           << " stop=" << (trueOnlyStopBB ? trueOnlyStopBB->getName() : StringRef("<null>"))
           << " falseStart="
           << (falseStartBB ? falseStartBB->getName() : StringRef("<null>"))
           << " pdBB=" << (pdBB ? pdBB->getName() : StringRef("<null>"))
           << " forcedByShortCircuitTail=" << forcedByShortCircuitTail << "\n";
    createSubIfElseRegions(trueStartBB, brBB, trueOnlyStopBB, false);
    nextEntryBB = falseStartBB;
  } else if (falseBrOnly && (returnDominated == -1)) {
    errs() << "SUSAN: marking only false branch\n";
    if (auto lr = getParentLoopRegion())
      for (auto BB : trueBBs)
        lr->removeBBToVisit(BB);
    BasicBlock *falseOnlyStopBB = trueStartBB;
    if (forcedByShortCircuitTail && pdBB)
      falseOnlyStopBB = pdBB;
    errs() << "CBERegion_DEBUG: false-only stop boundary in "
           << brBB->getParent()->getName() << "::" << brBB->getName()
           << " stop=" << (falseOnlyStopBB ? falseOnlyStopBB->getName() : StringRef("<null>"))
           << " trueStart="
           << (trueStartBB ? trueStartBB->getName() : StringRef("<null>"))
           << " pdBB=" << (pdBB ? pdBB->getName() : StringRef("<null>"))
           << " forcedByShortCircuitTail=" << forcedByShortCircuitTail << "\n";
    createSubIfElseRegions(falseStartBB, brBB, falseOnlyStopBB, true);
    nextEntryBB = trueStartBB;
  } else {
    errs() << "SUSAN: marking both branches\n";
    auto nextEntryBB1 =
        createSubIfElseRegions(trueStartBB, brBB, pdBB, false);
    auto nextEntryBB2 =
        createSubIfElseRegions(falseStartBB, brBB, pdBB, true);
    // For full if/else regions, ownership must hand off at the merge exactly once.
    if (pdBB)
      nextEntryBB = pdBB;
    else
      nextEntryBB = nextEntryBB1 ? nextEntryBB1 : nextEntryBB2;
    if (nextEntryBB)
      errs() << "CBERegion: nextEntryBB 121: " << nextEntryBB->getName() << "\n";
    else
      errs() << "CBERegion: nextEntryBB 121: <null>\n";
  }

  useCompoundPredicate = tryBuildCompoundPredicate();
  errs() << "CBERegion_DEBUG: compound predicate for "
         << brBB->getParent()->getName() << "::" << brBB->getName()
         << " enabled=" << useCompoundPredicate << "\n";

  if (parentR && parentR->isaLoopRegion())
    removeIfElseBlockFromLR((LoopRegion *)parentR, brBB);
  errs() << "=================SUSAN: END OF marking region : "
         << br->getParent()->getName() << "==================\n";
  }
}

std::unique_ptr<IfElseRegion::CompoundPredicate>
IfElseRegion::makeLeafPredicate(Value *V, bool Negate) {
  auto Pred = std::make_unique<CompoundPredicate>();
  Pred->op = CompoundPredicateOp::Leaf;
  Pred->leafValue = V;
  Pred->negateLeaf = Negate;
  return Pred;
}

std::unique_ptr<IfElseRegion::CompoundPredicate>
IfElseRegion::makeBinaryPredicate(
    CompoundPredicateOp Op, std::unique_ptr<CompoundPredicate> LHS,
    std::unique_ptr<CompoundPredicate> RHS) {
  auto Pred = std::make_unique<CompoundPredicate>();
  Pred->op = Op;
  Pred->lhs = std::move(LHS);
  Pred->rhs = std::move(RHS);
  return Pred;
}

bool IfElseRegion::tryBuildCompoundPredicate() {
  compoundPredicate.reset();
  flattenThenFromChild = false;
  flattenThenChildEntryBB = nullptr;
  flattenThenUsesChildThenBranch = true;

  if (!brInst || !brInst->isConditional())
    return false;

  Value *parentCond = brInst->getCondition();
  if (!parentCond)
    return false;

  // Pattern A: Parent true-arm starts with a condition-only block that shares
  // the parent's false-tail. This is a short-circuit && pattern.
  Value *childCond = nullptr;
  if (isConditionOnlyBranchBlock(trueStartBB, childCond)) {
    auto *childBr = cast<BranchInst>(trueStartBB->getTerminator());
    bool childTrueToParentFalse = (childBr->getSuccessor(0) == falseStartBB);
    bool childFalseToParentFalse = (childBr->getSuccessor(1) == falseStartBB);
    if (childTrueToParentFalse || childFalseToParentFalse) {
      bool negateChild = childTrueToParentFalse;
      bool childThenIsCombinedThen = childFalseToParentFalse;
      compoundPredicate = makeBinaryPredicate(
          CompoundPredicateOp::And, makeLeafPredicate(parentCond, false),
          makeLeafPredicate(childCond, negateChild));
      // The child condition has been absorbed into the parent condition.
      // When printing the parent true branch, print only the selected child
      // branch body to avoid a redundant nested check.
      flattenThenFromChild = true;
      flattenThenChildEntryBB = trueStartBB;
      flattenThenUsesChildThenBranch = childThenIsCombinedThen;
      errs() << "CBERegion_DEBUG: collapsed short-circuit && at "
             << brBB->getParent()->getName() << "::" << brBB->getName()
             << " child=" << trueStartBB->getName()
             << " sharedFalseTail=" << falseStartBB->getName()
             << " negateChild=" << negateChild
             << " flattenThenUsesChildThenBranch="
             << flattenThenUsesChildThenBranch << "\n";
      return true;
    }
  }

  // Pattern B: Parent false-arm starts with a condition-only block that shares
  // the parent's true-tail. This is a short-circuit || pattern.
  childCond = nullptr;
  if (isConditionOnlyBranchBlock(falseStartBB, childCond)) {
    auto *childBr = cast<BranchInst>(falseStartBB->getTerminator());
    bool childTrueToParentTrue = (childBr->getSuccessor(0) == trueStartBB);
    bool childFalseToParentTrue = (childBr->getSuccessor(1) == trueStartBB);
    if (childTrueToParentTrue || childFalseToParentTrue) {
      bool negateChild = childFalseToParentTrue;
      compoundPredicate = makeBinaryPredicate(
          CompoundPredicateOp::Or, makeLeafPredicate(parentCond, false),
          makeLeafPredicate(childCond, negateChild));
      errs() << "CBERegion_DEBUG: collapsed short-circuit || at "
             << brBB->getParent()->getName() << "::" << brBB->getName()
             << " child=" << falseStartBB->getName()
             << " sharedTrueTail=" << trueStartBB->getName()
             << " negateChild=" << negateChild << "\n";
      return true;
    }
  }

  errs() << "CBERegion_DEBUG: no compound predicate pattern at "
         << brBB->getParent()->getName() << "::" << brBB->getName() << "\n";
  return false;
}

void IfElseRegion::printCompoundPredicate(const CompoundPredicate *Pred) {
  if (!Pred)
    return;

  switch (Pred->op) {
  case CompoundPredicateOp::Leaf:
    if (Pred->negateLeaf)
      cw->Out << "!(";
    cw->writeOperand(Pred->leafValue, cw->ContextCasted);
    if (Pred->negateLeaf)
      cw->Out << ")";
    return;
  case CompoundPredicateOp::And:
  case CompoundPredicateOp::Or:
    cw->Out << "(";
    printCompoundPredicate(Pred->lhs.get());
    cw->Out << (Pred->op == CompoundPredicateOp::And ? " && " : " || ");
    printCompoundPredicate(Pred->rhs.get());
    cw->Out << ")";
    return;
  }
}

void IfElseRegion::printFlattenedChildBranch(IfElseRegion *Child,
                                             bool UseThenBranch) {
  if (!Child)
    return;
  auto &SubRegions =
      UseThenBranch ? Child->thenSubRegions : Child->elseSubRegions;
  for (auto *R : SubRegions)
    R->printRegionDAG();
}

BasicBlock *IfElseRegion::createSubIfElseRegions(BasicBlock *start,
                                                 BasicBlock *brBlock,
                                                 BasicBlock *stopBB,
                                                 bool isElseBranch) {
  if (!start || !brBlock) {
    errs() << "CBERegion_DEBUG: invalid subregion boundary start="
           << (start ? start->getName() : StringRef("<null>"))
           << " brBlock="
           << (brBlock ? brBlock->getName() : StringRef("<null>"))
           << ", aborting subregion walk\n";
    return start;
  }
  LoopRegion *lr = getParentLoopRegion();
  if (lr)
    lr->removeBBToVisit(brBlock);

  BasicBlock *currBB = start;
  std::set<BasicBlock *> visitedBBs;
  unsigned iterationCount = 0;
  const unsigned maxIterations = 10000;
  // TODO: this is a hasty patch
  errs() << start->getName() << " to " << brBlock->getName() << "\n";
  errs() << "stopBB: " << (stopBB ? stopBB->getName() : StringRef("<null>")) << "\n";
  errs() << PDT->dominates(currBB, brBlock) << "\n";
  while (currBB && !PDT->dominates(currBB, brBlock) && currBB != stopBB) {
    if (!visitedBBs.insert(currBB).second) {
      errs() << "ANDREW: detected repeated block in subregion generation: "
             << currBB->getName() << ", stopping to avoid cycling.\n";
      break;
    }
    if (++iterationCount > maxIterations) {
      errs() << "ANDREW: subregion generation exceeded max iterations for "
             << brBlock->getName() << ", stopping defensively.\n";
      break;
    }

    if (lr && currBB == lr->getNextEntryBB()) {
      errs() << "ANDREW: reached loop exit block " << currBB->getName()
             << ", stopping subregion generation.\n";
      break;
    }

    unsigned startDepth = LI->getLoopDepth(start);
    unsigned currDepth = LI->getLoopDepth(currBB);

    if (currDepth < startDepth) {
       // Exited scope - stop
       errs() << "ANDREW: " << currBB->getName() << " exited scope (Depth " << currDepth << " < " << startDepth << "), stopping.\n";
       break;
    }

    // ANDREW: Stop if we've hit a merge point not directly on the break path.
    // Use dominance: The start of the break path must dominate any block on the exclusive path.
    // If start does not dominate currBB, it means currBB is reachable from elsewhere (e.g. normal loop exit).
    if (currBB != start && !DT->dominates(start, currBB)) {
      errs() << "ANDREW: " << currBB->getName() << " is not dominated by " << start->getName() << ", stopping subregion generation.\n";
      break;
    }
    if (!claimedSubregionEntries.insert(currBB).second) {
      errs() << "ANDREW: duplicate subregion entry claim in same if-region: "
             << currBB->getName() << ", stopping sibling re-expansion.\n";
      break;
    }
    CBERegion2 *subR = createSubRegions(this, currBB, stopBB);
    if (!subR) break;
    if (!isElseBranch)
      thenSubRegions.push_back(subR);
    else
      elseSubRegions.push_back(subR);
    if (isa<UnreachableInst>(currBB->getTerminator()))
      break;
    BasicBlock *nextBB = subR->getNextEntryBB();
    if (nextBB == currBB) {
      errs() << "ANDREW: subregion next-entry did not advance at "
             << currBB->getName() << ", stopping to avoid infinite loop.\n";
      break;
    }
    currBB = nextBB;
    if(!currBB) break;
    errs() << "SUSAN: currbb 562: " << currBB->getName() << "\n";
  }
  return currBB;
}

void LoopRegion::createCBERegionDAG(BasicBlock *entryBB) {
  BasicBlock *nextRegionEntryBB = entryBB;
  while (!this->hasNoRemainingBBs()) {
    CBERegion2 *entryR = createSubRegions(this, nextRegionEntryBB);
    if (!entryR) break;
    LoopBodyRegionDAG.push_back(entryR);
    if (nextRegionEntryBB == this->latchBB)
      return;

    nextRegionEntryBB = entryR->getNextEntryBB();
    if (!nextRegionEntryBB) {
      errs() << "Did not detect nextRegion\n";
      break;
    }
    
    // Stop if next block is outside the loop or is the latch (already handled separately)
    if (nextRegionEntryBB == latchBB) {
      errs() << "ANDREW: nextRegionEntryBB is latch, stopping loop body traversal\n";
      break;
    }
    if (!loop->contains(nextRegionEntryBB)) {
      errs() << "ANDREW: nextRegionEntryBB " << nextRegionEntryBB->getName() 
             << " is outside loop, stopping traversal\n";
      break;
    }
    
    errs() << "SUSAN: nextRegionEntryBB " << nextRegionEntryBB->getName();
    errs() << " for region: " << *(this->loop) << "\n";
  }
}

void CBERegion2::createCBERegionDAG(BasicBlock *entryBB, CBERegion2 *parentR,
                                    BasicBlock *endBB) {
  // Reset traversal state at the outermost entry.
  if (CBERegionDAG.empty()) {
    regionBuildVisited.clear();
    activeCreateSubregions.clear();
  }

  if (!entryBB)
    return;
  if (!regionBuildVisited.insert(entryBB).second) {
    errs() << "ANDREW: CBERegion: detected recursive revisit of entry block "
           << entryBB->getName() << ", stopping region DAG recursion.\n";
    return;
  }

  errs() << "YEBIN: in Function " << entryBB->getParent()->getName();
  if (parentR)
    errs() << " with ParentR " << parentR->getEntryBlock()->getName();
  errs() << "\n";
  errs() << "YEBIN: creating CBE Region with " << entryBB->getName() << " to "
         << endBB->getName() << "\n";
  CBERegion2 *entryR = createSubRegions(parentR, entryBB);
  if (!entryR) return;
  CBERegionDAG.push_back(entryR);
  if (entryBB == endBB)
    return;

  BasicBlock *nextRegionEntryBB = entryR->getNextEntryBB();
  if (nextRegionEntryBB) {
    errs() << "SUSAN: nextRegionEntryBB " << nextRegionEntryBB->getName()
           << "\n";
    if (nextRegionEntryBB == entryBB) {
      errs() << "ANDREW: CBERegion: nextRegionEntryBB did not advance from "
             << entryBB->getName() << ", stopping recursion.\n";
      return;
    }
    createCBERegionDAG(nextRegionEntryBB, parentR, endBB);
  }
}

void LinearRegion::print() {
  errs() << "Linear Region with entering block: " << getEntryBlock()->getName()
         << "\n";
  for (auto BB : BBs)
    errs() << BB->getName() << "\n";
}
bool LinearRegion::containsBlock(BasicBlock *BB) {
  for (auto *currBB : BBs)
    if (currBB == BB)
      return true;
  return false;
}
void IfElseRegion::print() {
  errs() << "IfElse Region with entering block: "
         << getEntryBlock()->getParent()->getName()
         << "::" << getEntryBlock()->getName() << "\n";
  errs() << "thenSubRegions : \n";
  for (auto R : thenSubRegions)
    R->print();
  errs() << "thenSubRegions end\n";
  errs() << "elseSubRegions : \n";
  for (auto R : elseSubRegions)
    R->print();
  errs() << "elseSubRegions end\n";
}
bool IfElseRegion::containsBlock(BasicBlock *BB) {
  if (brBB == BB)
    return true;
  for (auto *R : thenSubRegions) {
    if (R && R->containsBlock(BB))
      return true;
  }
  for (auto *R : elseSubRegions) {
    if (R && R->containsBlock(BB))
      return true;
  }
  return false;
}
void LoopRegion::print() {
  errs() << "Loop Region with entering block: " << getEntryBlock()->getName()
         << "\n";
  for (auto R : LoopBodyRegionDAG)
    R->print();
}
bool LoopRegion::containsBlock(BasicBlock *BB) {
  return loop && BB && loop->contains(BB);
}
void CBERegion2::print() {
  errs() << "======================Printing CBERegions================\n";
  errs() << "Top CBERegion\n";
  for (auto R : CBERegionDAG)
    R->print();
}

void LinearRegion::printRegionDAG() {
  errs() << "Linear Region with entering block: " << getEntryBlock()->getName()
         << "\n";
  for (auto BB : BBs) {
    // Skip synthetic loop-exit trampoline blocks. Their PHI initialization is
    // materialized where the structured loop is emitted.
    if (BB->getName().contains("loopexit")) {
      auto *br = dyn_cast<BranchInst>(BB->getTerminator());
      if (br && br->isUnconditional() &&
          BB->getFirstNonPHIOrDbgOrLifetime() == BB->getTerminator()) {
        errs() << "ANDREW: skipping loopexit trampoline block "
               << BB->getName() << "\n";
        continue;
      }
    }
    errs() << "SUSAN: printing bb:" << BB->getName() << "\n";
    cw->printBasicBlock(BB);
  }
}

void IfElseRegion::printRegionDAG() {
  // ANDREW: Handle loop break statements specially
  if (isLoopBreak) {
    auto condInst = brInst->getCondition();
    // print instructions before the branch
    for (auto &I : *brBB) {
      if (&I == condInst) break;
      if (cw->isSkipableInst(&I)) continue;
      cw->printInstruction(&I);
    }
    cw->emitPHIsForPredecessor(brBB);
    
    cw->Out << "  if (";
    if (negateCond) cw->Out << "!(";
    cw->writeOperand(condInst, cw->ContextCasted);
    if (negateCond) cw->Out << ")";
    cw->Out << ") {\n";
    
    // ANDREW: Print logic inside the break block (e.g. inner loops)
    for (auto R : thenSubRegions)
      R->printRegionDAG();
      
    cw->Out << "  break;\n";
    cw->Out << "  }\n";
    return;
  }

  // ANDREW: Handle loop continue statements specially
  if (isLoopContinue) {
    auto condInst = brInst->getCondition();
    // print instructions before the branch
    for (auto &I : *brBB) {
      if (&I == condInst) break;
      if (cw->isSkipableInst(&I)) continue;
      cw->printInstruction(&I);
    }
    cw->emitPHIsForPredecessor(brBB);
    
    cw->Out << "  if (";
    if (negateCond) cw->Out << "!(";
    cw->writeOperand(condInst, cw->ContextCasted);
    if (negateCond) cw->Out << ")";
    cw->Out << ") {\n";
    cw->Out << "  continue;\n";
    cw->Out << "  }\n";
    return;
  }

  // ANDREW: Handle function return statements specially  
  if (isFunctionReturn) {
    auto condInst = brInst->getCondition();
    // print instructions before the branch
    for (auto &I : *brBB) {
      if (&I == condInst) break;
      if (cw->isSkipableInst(&I)) continue;
      cw->printInstruction(&I);
    }
    cw->emitPHIsForPredecessor(brBB);
    
    cw->Out << "  if (";
    if (negateCond) cw->Out << "!(";
    cw->writeOperand(condInst, cw->ContextCasted);
    if (negateCond) cw->Out << ")";
    cw->Out << ") {\n";
    // Emit the actual return value from the exiting branch instead of a typed
    // zero default. This preserves source semantics for early-return patterns.
    BasicBlock *exitingStart = negateCond ? falseStartBB : trueStartBB;
    BasicBlock *retBB = isExitingFunction(exitingStart);
    ReturnInst *retInst =
        retBB ? dyn_cast<ReturnInst>(retBB->getTerminator()) : nullptr;

    if (retInst) {
      if (retInst->getNumOperands() == 0) {
        cw->Out << "  return;\n";
      } else {
        Value *retVal = retInst->getReturnValue();
        if (auto *phi = dyn_cast<PHINode>(retVal)) {
          int incomingIdx = phi->getBasicBlockIndex(exitingStart);
          if (incomingIdx >= 0)
            retVal = phi->getIncomingValue(incomingIdx);
        }
        cw->Out << "  return ";
        cw->writeOperandInternal(retVal);
        cw->Out << ";\n";
      }
    } else {
      // Conservative fallback when the exit path is unreachable or malformed.
      cw->Out << "  return;\n";
    }
    cw->Out << "  }\n";
    return;
  }

  if (!this->parentRegion || this->parentRegion->isaLinearRegion()) {
    auto FuncName = demangleFunctionName(this->entryBlock->getParent()->getName());
    cw->Out << "// INSERT COMMENT IFELSE: " << FuncName << "::" << this->entryBlock->getName() << "\n";
  }

  errs() << "IfElse Region with entering block: "
         << getEntryBlock()->getParent()->getName()
         << "::" << getEntryBlock()->getName() << "\n";
  errs() << "thenSubRegions : \n";
  for (auto R : thenSubRegions)
    R->print();

  auto condInst = brInst->getCondition();
  // print instructions before the branch
  for (auto &I : *brBB) {
    if (&I == condInst)
      break;
    if (cw->isSkipableInst(&I))
      continue;
    cw->printInstruction(&I);
  }
  // Materialize successor PHI incoming values for this predecessor block.
  cw->emitPHIsForPredecessor(brBB);

  // print If branch
  cw->Out << "  if (";
  if (useCompoundPredicate && compoundPredicate) {
    errs() << "CBERegion_DEBUG: printing compound predicate for "
           << brBB->getParent()->getName() << "::" << brBB->getName() << "\n";
    printCompoundPredicate(compoundPredicate.get());
  } else {
    cw->writeOperand(condInst, cw->ContextCasted);
  }
  cw->Out << ") {";
  cw->Out << " // IFELSE MARKER: " << entryBlock->getName() << " IF\n"; 
  for (auto *R : thenSubRegions) {
    if (flattenThenFromChild && R &&
        R->getEntryBlock() == flattenThenChildEntryBB &&
        R->isaIfElseRegion()) {
      auto *Child = static_cast<IfElseRegion *>(R);
      errs() << "CBERegion_DEBUG: flattening absorbed child condition at "
             << brBB->getParent()->getName() << "::" << brBB->getName()
             << " child=" << flattenThenChildEntryBB->getName()
             << " useThen=" << flattenThenUsesChildThenBranch << "\n";
      printFlattenedChildBranch(Child, flattenThenUsesChildThenBranch);
      continue;
    }
    R->printRegionDAG();
  }
  // Materialize merge PHIs for simple direct-edge true branch blocks.
  if (trueStartBB && pdBB) {
    auto *trueTerm = dyn_cast<BranchInst>(trueStartBB->getTerminator());
    if (trueTerm) {
      for (unsigned i = 0; i < trueTerm->getNumSuccessors(); ++i) {
        if (trueTerm->getSuccessor(i) == pdBB) {
          cw->emitPHICopiesForSuccessorEdge(trueStartBB, pdBB, 2);
          break;
        }
      }
    }
  }

  bool falseEdgeNeedsMergeCopies = false;
  if (falseStartBB && pdBB) {
    auto *falseTerm = dyn_cast<BranchInst>(falseStartBB->getTerminator());
    if (falseTerm) {
      for (unsigned i = 0; i < falseTerm->getNumSuccessors(); ++i) {
        if (falseTerm->getSuccessor(i) == pdBB) {
          falseEdgeNeedsMergeCopies = true;
          break;
        }
      }
    }
  }

  // print else branch
  if (!elseSubRegions.empty() || falseEdgeNeedsMergeCopies) {
    errs() << "elseSubRegions : \n";
    cw->Out << "  } else {";
    cw->Out << " // IFELSE MARKER: " << entryBlock->getName() << " ELSE\n";
    for (auto R : elseSubRegions)
      R->printRegionDAG();
    // Materialize merge PHIs for simple direct-edge false branch blocks.
    if (falseEdgeNeedsMergeCopies)
      cw->emitPHICopiesForSuccessorEdge(falseStartBB, pdBB, 2);
  }

  cw->Out << "  }\n";
}


void LoopRegion::printRegionDAG() {
  if (!this->parentRegion || this->parentRegion->isaLinearRegion()) {
    auto FuncName = demangleFunctionName(this->entryBlock->getParent()->getName());
    cw->Out << "// INSERT COMMENT LOOP: " << FuncName << "::" << this->entryBlock->getName() << "\n";
  }
  errs() << "Loop Region with entering block: " << getEntryBlock()->getName()
         << "\n";

  switch(this->loopType) {
    case doWhileLoop:
      errs() << "Print doWhileLoop " << loop->getName() << "...\n";
      printDoWhileLoop();
      return;
    case whileLoop:
      errs() << "Print whileLoop " << loop->getName() << "...\n";
      printWhileLoop();
      return;
    case whileLoopWithContinue:
      errs() << "Print whileLoopWithContinue " << loop->getName() << "...\n";
      printWhileLoopWithContinue();
      return;
    case forLoop:
      errs() << "Print forLoop " << loop->getName() << "...\n";
      break;
    case unknown:
      errs() << "Print unknown loop type " << loop->getName() << "...\n";
      break;
  }

  BasicBlock *header = loop->getHeader();
  bool negateCondition = false;
  Instruction *condInst = cw->findCondInst(loop, negateCondition);

  std::set<Instruction *> printedLiveins;
  std::set<Value *> condRelatedInsts;
  BasicBlock *condBlock = condInst->getParent();
  cw->findCondRelatedInsts(condBlock, condRelatedInsts);
  for (auto condRelatedInst : condRelatedInsts) {
    Instruction *inst = cast<Instruction>(condRelatedInst);
    errs() << "SUSAN: condrelatedinst:" << *inst << "\n";
    if (cw->isIVIncrement(inst) || isa<PHINode>(inst) ||
        isa<BranchInst>(inst) || isa<CmpInst>(inst) ||
        cw->isInlinableInst(*inst) || inst == this->incr)
      continue;
    errs() << "SUSAN: printing condRelatedInst: " << *inst << "\n";
    cw->printInstruction(inst);
  }

  auto headerBr = dyn_cast<BranchInst>(header->getTerminator());
  if (headerBr->getMetadata("tulip.doall.loop.grid.collapse")) {
    // cw->Out << "//INSERT COMMENT: " << header->getName() << "\n";
    cw->Out << "#pragma omp parallel for collapse(2)\n";
  } else if (headerBr->getMetadata("tulip.doall.loop.grid")) {
    bool printCollapse = false;
    for (BasicBlock *BB : loop->getBlocks()) {
      if (BB->getTerminator()->getMetadata("tulip.doall.loop.block")) {
        // cw->Out << "//INSERT COMMENT: " << header->getName() << "\n";
        cw->Out << "#pragma omp parallel for collapse(2)\n";
        printCollapse = true;
        break;
      }
    }
    if (!printCollapse) {
      // cw->Out << "//INSERT COMMENT: " << header->getName() << "\n";
      cw->Out << "#pragma omp parallel for\n";
    }
  } else if (headerBr->getMetadata("noelle.doall.loop")) {
    bool printReduction = false;
    for (BasicBlock *BB : loop->getBlocks()) {
      for (auto &I : *BB) {
        if (I.getMetadata("tulip.reduce.add")) {
          printReduction = true;
          // cw->Out << "//INSERT COMMENT: " << header->getName() << "\n";
          cw->Out << "#pragma omp simd reduction(+:";
          cw->writeOperand(&I);
          cw->Out << ")\n";
        }
      }
    }

    if (!printReduction) {
      // cw->Out << "//INSERT COMMENT: " << header->getName() << "\n";
      cw->Out << "#pragma omp parallel for \n";
    }
  }

  // for (BasicBlock *BB : loop->getBlocks()){
  //   for(auto &I : *BB){
  //     if(I.getMetadata("tulip.reduce.add")){
  //       cw->Out << "#pragma omp simd reduction(+:";
  //       cw->writeOperand(&I);
  //       cw->Out << ")";
  //     }
  //     else if(I.getMetadata("tulip.arr.reduce.add")){
  //       cw->Out << "#pragma omp simd reduction(+:";
  //       GetElementPtrInst *gep = dyn_cast<GetElementPtrInst>(&I);
  //       Value *ptr = gep->getPointerOperand();
  //       cw->Out<<cw->GetValueName(ptr);
  //       PointerType *ptrTy = dyn_cast<PointerType>(ptr->getType());
  //       cw->Out << "[0:";
  //       assert(ptrTy && "CBERegion: not a pointer type? 288\n");
  //       ArrayType* arrTy =
  //       dyn_cast<ArrayType>(ptrTy->getPointerElementType()); assert(arrTy &&
  //       "CBERegion: not an array type? 290\n"); cw->Out <<
  //       arrTy->getNumElements(); cw->Out << "]"; cw->Out << ")";
  //     }
  //   }
  // }

  // Initialize non-IV PHIs from the preheader before entering the loop.
  for (auto &I : *header) {
    if (PHINode *phi = dyn_cast<PHINode>(&I)) {
      if (phi == IV)
        continue;
      if (cw->isExtraIVEquivalentToMainIV(phi)) {
        errs() << "ANDREW: Skipping preheader init for equivalent extra IV: "
               << *phi << "\n";
        continue;
      }
      for (unsigned i = 0; i < phi->getNumIncomingValues(); i++) {
        BasicBlock *incomingBB = phi->getIncomingBlock(i);
        if (!loop->contains(incomingBB)) {
          Value *initVal = phi->getIncomingValue(i);
          if (isa<Constant>(initVal)) {
            errs() << "ANDREW: Skipping constant preheader PHI init: " << *phi << "\n";
            break;
          }
          // Only materialize preheader init for induction-like PHIs.
          // This keeps needed initializers (e.g. nza = rowstr[j]) and
          // avoids redundant reduction carry initializers.
          Value *loopIncomingVal = nullptr;
          for (unsigned j = 0; j < phi->getNumIncomingValues(); j++) {
            if (loop->contains(phi->getIncomingBlock(j))) {
              loopIncomingVal = phi->getIncomingValue(j);
              break;
            }
          }
          if (BinaryOperator *loopBinOp = dyn_cast_or_null<BinaryOperator>(loopIncomingVal)) {
            Value *op0 = loopBinOp->getOperand(0);
            Value *op1 = loopBinOp->getOperand(1);
            bool op0IsPhi = (op0 == phi);
            bool op1IsPhi = (op1 == phi);
            bool op0IsOne =
                (isa<ConstantInt>(op0) && cast<ConstantInt>(op0)->isOne()) ||
                (isa<ConstantFP>(op0) && cast<ConstantFP>(op0)->isExactlyValue(1.0));
            bool op1IsOne =
                (isa<ConstantInt>(op1) && cast<ConstantInt>(op1)->isOne()) ||
                (isa<ConstantFP>(op1) && cast<ConstantFP>(op1)->isExactlyValue(1.0));
            bool isSimpleStep =
                ((loopBinOp->getOpcode() == Instruction::Add ||
                  loopBinOp->getOpcode() == Instruction::FAdd) &&
                 ((op0IsPhi && op1IsOne) || (op1IsPhi && op0IsOne))) ||
                ((loopBinOp->getOpcode() == Instruction::Sub ||
                  loopBinOp->getOpcode() == Instruction::FSub) &&
                 (op0IsPhi && op1IsOne));
            if (!isSimpleStep) {
              errs() << "ANDREW: Skipping non-step preheader PHI init: " << *phi << "\n";
              break;
            }
          }
          // Reduction-carry PHIs frequently appear as floating-point PHI<-PHI
          // transitions across nested loops; avoid emitting noisy preheader
          // rebinds such as "__FIXME__rho_2e_1 = rho".
          if (isa<PHINode>(loopIncomingVal) && phi->getType()->isFloatingPointTy()) {
            errs() << "ANDREW: Skipping floating PHI<-PHI preheader init: " << *phi << "\n";
            break;
          }
          std::string phiName = cw->GetValueName(phi);
          cw->Out << "  " << phiName << " = ";
          if (BinaryOperator *binOp = dyn_cast<BinaryOperator>(initVal)) {
            cw->writeOperandInternal(binOp->getOperand(0));
            switch (binOp->getOpcode()) {
              case Instruction::Add: cw->Out << " + "; break;
              case Instruction::FAdd: cw->Out << " + "; break;
              case Instruction::Sub: cw->Out << " - "; break;
              case Instruction::FSub: cw->Out << " - "; break;
              case Instruction::Mul: cw->Out << " * "; break;
              case Instruction::FMul: cw->Out << " * "; break;
              case Instruction::UDiv:
              case Instruction::SDiv:
              case Instruction::FDiv: cw->Out << " / "; break;
              case Instruction::URem:
              case Instruction::SRem:
              case Instruction::FRem: cw->Out << " % "; break;
              default: cw->Out << " /* unknown op */ "; break;
            }
            cw->writeOperandInternal(binOp->getOperand(1));
          } else {
            cw->writeOperandInternal(initVal);
          }
          cw->Out << ";\n";
          break;
        }
      }
    }
  }

  // Materialize constant PHI values that come from synthetic loopexit blocks
  // before printing the structured for-loop body.
  if (nextEntryBB && nextEntryBB->getName().contains("loopexit")) {
    if (auto *exitBr = dyn_cast<BranchInst>(nextEntryBB->getTerminator())) {
      if (exitBr->isUnconditional() &&
          nextEntryBB->getFirstNonPHIOrDbgOrLifetime() ==
              nextEntryBB->getTerminator()) {
        BasicBlock *mergeBB = exitBr->getSuccessor(0);
        for (auto &MI : *mergeBB) {
          auto *phi = dyn_cast<PHINode>(&MI);
          if (!phi)
            break;
          int incomingIdx = phi->getBasicBlockIndex(nextEntryBB);
          if (incomingIdx < 0)
            continue;
          Value *incomingVal = phi->getIncomingValue(incomingIdx);
          if (!isa<Constant>(incomingVal))
            continue;
          std::string phiName = cw->GetValueName(phi);
          cw->Out << "  " << phiName << " = ";
          cw->writeOperandInternal(incomingVal);
          cw->Out << ";\n";
          errs() << "ANDREW: pre-initializing loopexit PHI " << *phi << "\n";
        }
      }
    }
  }

  cw->Out << "for(";

  // initiation
  // Only emit an inline IV declaration when it is safe.
  // - If the IV name differs from the lower-bound name, declaring in-header is
  //   naturally safe.
  // - If the lower bound is an instruction/expression, we may still need an
  //   in-header declaration to avoid "for(i = ...)" with an undeclared i.
  //   However, do NOT redeclare when the IV is used outside this loop, because
  //   that shadows an outer-scope IV and can leave the outer value uninitialized
  //   after the loop (e.g. use-after-loop of k in sparse()).
  std::string ivName = cw->GetValueName(IV);
  std::string lbName = cw->GetValueName(lb);
  bool lbIsInstruction = isa<Instruction>(lb);
  bool ivUsedOutsideLoop = false;
  for (User *U : IV->users()) {
    if (Instruction *UI = dyn_cast<Instruction>(U))
      if (!loop->contains(UI->getParent())) {
        ivUsedOutsideLoop = true;
        break;
      }
  }

  // Never redeclare the IV in the for-header if it escapes the loop; that
  // must reuse the function-scope variable.
  bool shouldDeclareInHeader =
      !ivUsedOutsideLoop && (ivName != lbName || lbIsInstruction);
  if (shouldDeclareInHeader) {
    cw->printTypeName(cw->Out, IV->getType(), true);
    cw->Out << " ";
  }
  cw->Out << cw->GetValueName(IV, true) << " = ";
  if (Instruction *lbInst = dyn_cast<Instruction>(lb))
    cw->writeInstComputationInline(*lbInst);
  else
    cw->writeOperand(lb);
  cw->Out << "; ";

  // exit condition
  // Use writeOperandInternal instead of GetValueName to properly handle
  // conversion instructions and get the correct variable name
  // ANDREW CASTING
  if (ICmpInst *icmp = dyn_cast<ICmpInst>(condInst)) {
    cw->writeOperandWithCast(condInst->getOperand(0), *icmp);
    if (!negateCondition && (icmp->getPredicate() == ICmpInst::ICMP_NE))
      cw->Out << " < ";
    else if (negateCondition && (icmp->getPredicate() == ICmpInst::ICMP_EQ))
      cw->Out << " < ";
    else
      cw->printCmpOperator(icmp, negateCondition);
    // ANDREW CASTING
    cw->writeOperandWithCast(condInst->getOperand(1), *icmp);
  } else if (FCmpInst *fcmp = dyn_cast<FCmpInst>(condInst)) {
    // ANDREW CASTING
    cw->writeOperandInternal(condInst->getOperand(0));
    // Handle float comparisons
    CmpInst::Predicate pred = fcmp->getPredicate();
    if (negateCondition) {
      // Negate the predicate
      switch (pred) {
        case FCmpInst::FCMP_OEQ: pred = FCmpInst::FCMP_ONE; break;
        case FCmpInst::FCMP_ONE: pred = FCmpInst::FCMP_OEQ; break;
        case FCmpInst::FCMP_OGT: pred = FCmpInst::FCMP_OLE; break;
        case FCmpInst::FCMP_OGE: pred = FCmpInst::FCMP_OLT; break;
        case FCmpInst::FCMP_OLT: pred = FCmpInst::FCMP_OGE; break;
        case FCmpInst::FCMP_OLE: pred = FCmpInst::FCMP_OGT; break;
        default: break;
      }
    }
    switch (pred) {
      case FCmpInst::FCMP_OEQ: cw->Out << " == "; break;
      case FCmpInst::FCMP_ONE: cw->Out << " != "; break;
      case FCmpInst::FCMP_OGT: cw->Out << " > "; break;
      case FCmpInst::FCMP_OGE: cw->Out << " >= "; break;
      case FCmpInst::FCMP_OLT: cw->Out << " < "; break;
      case FCmpInst::FCMP_OLE: cw->Out << " <= "; break;
      case FCmpInst::FCMP_UEQ: cw->Out << " == "; break;
      case FCmpInst::FCMP_UNE: cw->Out << " != "; break;
      case FCmpInst::FCMP_UGT: cw->Out << " > "; break;
      case FCmpInst::FCMP_UGE: cw->Out << " >= "; break;
      case FCmpInst::FCMP_ULT: cw->Out << " < "; break;
      case FCmpInst::FCMP_ULE: cw->Out << " <= "; break;
      default:
        llvm_unreachable("Unhandled FCmpInst predicate in loop condition");
    }
    cw->writeOperandInternal(condInst->getOperand(1));
  }
  cw->Out << "; ";

  // increment
  cw->printInstruction(cast<Instruction>(incr), false);
  cw->Out << ") {\n";

  // print loop body
  for (auto R : LoopBodyRegionDAG)
    R->printRegionDAG();

  bool latchCoveredByBodyRegion = false;
  for (auto *R : LoopBodyRegionDAG) {
    if (R && R->containsBlock(latchBB)) {
      latchCoveredByBodyRegion = true;
      break;
    }
  }
  // print extra instructions in a latch other than incr and br
  if (!latchCoveredByBodyRegion) {
    errs() << "CBERegion: printing latchBB " << latchBB->getName() << "\n";
    for (auto &I : *latchBB) {
      errs() << "CBERegion: I 316: " << I << "\n";
      if (!cw->isSkipableInst(&I) && incr != &I &&
          latchBB->getTerminator() != &I)
        cw->printInstruction(&I);
    }
  } else {
    errs() << "ANDREW: latch already covered by body regions, skipping "
           << latchBB->getName() << "\n";
  }
  

  // ANDREW: Emit PHI node updates for loop-carried variables (other than IV)
  // This handles cases like: kk = ik; at the end of each iteration
  // Note: header is already declared above
  for (auto &I : *header) {
    if (PHINode *phi = dyn_cast<PHINode>(&I)) {
      // Skip the induction variable, it's handled by the for() update
      if (phi == IV) continue;

      if (cw->isExtraIVEquivalentToMainIV(phi)) {
        errs() << "ANDREW: Skipping equivalent extra IV update (remapped to main IV): "
               << *phi << "\n";
        continue;
      }
      
      // Find the value coming from inside the loop (not the initial value)
      for (unsigned i = 0; i < phi->getNumIncomingValues(); i++) {
        BasicBlock *incomingBB = phi->getIncomingBlock(i);
        if (loop->contains(incomingBB)) {
          Value *incomingVal = phi->getIncomingValue(i);
          if (incomingVal == phi) {
            errs() << "ANDREW: Skipping self-assignment for PHI by identity: " << *phi << "\n";
            break;
          }
          std::string phiName = cw->GetValueName(phi);
          if (Instruction *incomingInst = dyn_cast<Instruction>(incomingVal)) {
            if (cw->GetValueName(incomingVal) == phiName &&
                !cw->isIVIncrement(incomingInst) &&
                !isa<BinaryOperator>(incomingInst)) {
              errs() << "ANDREW: Skipping duplicate PHI update by stable-name match: "
                     << *phi << "\n";
              break;
            }
          }
          if (BinaryOperator *binOp = dyn_cast<BinaryOperator>(incomingVal)) {
            Value *op0 = binOp->getOperand(0);
            Value *op1 = binOp->getOperand(1);
            bool op0IsPhi = (op0 == phi);
            bool op1IsPhi = (op1 == phi);
            bool op0IsOne =
                (isa<ConstantInt>(op0) && cast<ConstantInt>(op0)->isOne()) ||
                (isa<ConstantFP>(op0) && cast<ConstantFP>(op0)->isExactlyValue(1.0));
            bool op1IsOne =
                (isa<ConstantInt>(op1) && cast<ConstantInt>(op1)->isOne()) ||
                (isa<ConstantFP>(op1) && cast<ConstantFP>(op1)->isExactlyValue(1.0));
            bool isSimpleStep =
                ((binOp->getOpcode() == Instruction::Add ||
                  binOp->getOpcode() == Instruction::FAdd) &&
                 ((op0IsPhi && op1IsOne) || (op1IsPhi && op0IsOne))) ||
                ((binOp->getOpcode() == Instruction::Sub ||
                  binOp->getOpcode() == Instruction::FSub) &&
                 (op0IsPhi && op1IsOne));
            // Skip duplicate reduction-style PHI updates (already emitted in body),
            // but keep simple step updates like nza = nza + 1.
            if (!isSimpleStep) {
              std::string incomingName = cw->GetValueName(incomingVal);
              if (incomingName != phiName) {
                // Keep carry propagation (e.g. rho0 = rho) by assigning from the
                // incoming recurrence value, not from op0 (which may still be phi).
                cw->Out << "  " << phiName << " = ";
                cw->writeOperandInternal(incomingVal);
                cw->Out << ";\n";
                errs() << "ANDREW: Emitting carry-only PHI update for reduction: "
                       << *phi << " = " << *incomingVal << "\n";
              } else {
                errs() << "ANDREW: Skipping duplicate reduction PHI update: " << *phi
                       << " = " << *incomingVal << "\n";
              }
              break;
            }
          }
          // Emit: phi_name = incoming_value;
          // Use writeOperandInternal to bypass InstsToReplaceByPhi coalescing
          cw->Out << "  " << phiName << " = ";
          if (BinaryOperator *binOp = dyn_cast<BinaryOperator>(incomingVal)) {
            cw->writeOperandInternal(binOp->getOperand(0));
            switch (binOp->getOpcode()) {
              case Instruction::Add: cw->Out << " + "; break;
              case Instruction::FAdd: cw->Out << " + "; break;
              case Instruction::Sub: cw->Out << " - "; break;
              case Instruction::FSub: cw->Out << " - "; break;
              case Instruction::Mul: cw->Out << " * "; break;
              case Instruction::FMul: cw->Out << " * "; break;
              case Instruction::UDiv:
              case Instruction::SDiv:
              case Instruction::FDiv: cw->Out << " / "; break;
              case Instruction::URem:
              case Instruction::SRem:
              case Instruction::FRem: cw->Out << " % "; break;
              default: cw->Out << " /* unknown op */ "; break;
            }
            cw->writeOperandInternal(binOp->getOperand(1));
          } else {
            cw->writeOperandInternal(incomingVal);
          }
          cw->Out << ";\n";
          errs() << "ANDREW: Emitting PHI update: " << *phi << " = " << *incomingVal << "\n";
          break;
        }
      }
    }
  }

  cw->Out << "}\n";
}

void LoopRegion::printDoWhileLoop() {
  errs() << "PRINTING DOWHILE\n";
  for(auto R: LoopBodyRegionDAG) {
    R->print();
  }
  BasicBlock *header = loop->getHeader();
  bool negateCondition = false;
  Instruction *condInst = cw->findCondInst(loop, negateCondition);
  
  // Handle case where we can't find a condition instruction
  if (!condInst) {
    errs() << "Warning: condInst is null in printDoWhileLoop, emitting simple loop\n";
    cw->Out << "do {\n";
    cw->printBasicBlock(loop->getHeader());
    for (auto R : LoopBodyRegionDAG)
      R->printRegionDAG();
    cw->Out << "} while(1); // TODO: fix loop condition\n";
    return;
  }
  
  errs() << *condInst << "\n";

  std::set<Instruction *> printedLiveins;
  // TODO: move to later?
  // cond block should be latch
  std::set<Value *> condRelatedInsts;
  BasicBlock *condBlock = condInst->getParent();
  cw->findCondRelatedInsts(condBlock, condRelatedInsts);
  for (auto condRelatedInst : condRelatedInsts) {
    Instruction *inst = cast<Instruction>(condRelatedInst);
    if (isa<PHINode>(inst) || isa<BranchInst>(inst) || isa<CmpInst>(inst) ||
        cw->isInlinableInst(*inst)) {
      errs() << "YEBIN: not printing condrelatedinst:" << *inst << "\n";
      continue;
    }
    errs() << "YEBIN: printing condRelatedInst: " << *inst << "\n";
    cw->printInstruction(inst);
  }

  cw->Out << "do {\n";
  // print things in header; this is not in another region
  cw->printBasicBlock(loop->getHeader());

  // print loop body
  for (auto R : LoopBodyRegionDAG)
    R->printRegionDAG();

  // print extra instructions in a latch other than incr and br
  //errs() << "CBERegion: printing latchBB " << latchBB->getName() << "\n";
  //for (auto &I : *latchBB) {
  //  errs() << "CBERegion: I 316: " << I << "\n";
  //  if (!cw->isSkipableInst(&I) && incr != &I && latchBB->getTerminator() != &I)
  //    cw->printInstruction(&I);
  //}

  // Emit PHI node updates for loop-carried variables
  // This handles cases like: count = count + 1; at the end of each iteration
  for (auto &I : *header) {
    if (PHINode *phi = dyn_cast<PHINode>(&I)) {
      // Find the value coming from inside the loop (not the initial value)
      for (unsigned i = 0; i < phi->getNumIncomingValues(); i++) {
        BasicBlock *incomingBB = phi->getIncomingBlock(i);
        if (loop->contains(incomingBB)) {
          Value *incomingVal = phi->getIncomingValue(i);
          // Don't emit self-assignments (phi = phi)
          if (incomingVal == phi) continue;
          std::string phiName = cw->GetValueName(phi);
          if (Instruction *incomingInst = dyn_cast<Instruction>(incomingVal)) {
            if (cw->GetValueName(incomingVal) == phiName &&
                !cw->isIVIncrement(incomingInst) &&
                !isa<BinaryOperator>(incomingInst)) {
              errs() << "ANDREW: Skipping duplicate do-while PHI update by stable-name match: "
                     << *phi << "\n";
              break;
            }
          }
          if (BinaryOperator *binOp = dyn_cast<BinaryOperator>(incomingVal)) {
            Value *op0 = binOp->getOperand(0);
            Value *op1 = binOp->getOperand(1);
            bool op0IsPhi = (op0 == phi);
            bool op1IsPhi = (op1 == phi);
            bool op0IsOne =
                (isa<ConstantInt>(op0) && cast<ConstantInt>(op0)->isOne()) ||
                (isa<ConstantFP>(op0) && cast<ConstantFP>(op0)->isExactlyValue(1.0));
            bool op1IsOne =
                (isa<ConstantInt>(op1) && cast<ConstantInt>(op1)->isOne()) ||
                (isa<ConstantFP>(op1) && cast<ConstantFP>(op1)->isExactlyValue(1.0));
            bool isSimpleStep =
                ((binOp->getOpcode() == Instruction::Add ||
                  binOp->getOpcode() == Instruction::FAdd) &&
                 ((op0IsPhi && op1IsOne) || (op1IsPhi && op0IsOne))) ||
                ((binOp->getOpcode() == Instruction::Sub ||
                  binOp->getOpcode() == Instruction::FSub) &&
                 (op0IsPhi && op1IsOne));
            if (!isSimpleStep) {
              errs() << "ANDREW: Skipping duplicate do-while reduction PHI update: " << *phi
                     << " = " << *incomingVal << "\n";
              break;
            }
          }
          
          // Emit: phi_name = incoming_expression;
          cw->Out << "  " << cw->GetValueName(phi) << " = ";
          
          // If the incoming value is a BinaryOperator (add, sub, etc.),
          // expand it inline instead of using a potentially FIXME variable name
          if (BinaryOperator *binOp = dyn_cast<BinaryOperator>(incomingVal)) {
            // Write: op0 OPERATOR op1
            cw->writeOperandInternal(binOp->getOperand(0));
            switch (binOp->getOpcode()) {
              case Instruction::Add: cw->Out << " + "; break;
              case Instruction::FAdd: cw->Out << " + "; break;
              case Instruction::Sub: cw->Out << " - "; break;
              case Instruction::FSub: cw->Out << " - "; break;
              case Instruction::Mul: cw->Out << " * "; break;
              case Instruction::FMul: cw->Out << " * "; break;
              case Instruction::UDiv:
              case Instruction::SDiv:
              case Instruction::FDiv: cw->Out << " / "; break;
              case Instruction::URem:
              case Instruction::SRem:
              case Instruction::FRem: cw->Out << " % "; break;
              default: cw->Out << " /* unknown op */ "; break;
            }
            Value *rhs = binOp->getOperand(1);
            bool rhsNeedsParensForNonAssoc =
                (binOp->getOpcode() == Instruction::Sub ||
                 binOp->getOpcode() == Instruction::FSub ||
                 binOp->getOpcode() == Instruction::UDiv ||
                 binOp->getOpcode() == Instruction::SDiv ||
                 binOp->getOpcode() == Instruction::FDiv ||
                 binOp->getOpcode() == Instruction::URem ||
                 binOp->getOpcode() == Instruction::SRem ||
                 binOp->getOpcode() == Instruction::FRem ||
                 binOp->getOpcode() == Instruction::Shl ||
                 binOp->getOpcode() == Instruction::LShr ||
                 binOp->getOpcode() == Instruction::AShr) &&
                isa<BinaryOperator>(rhs);
            if (rhsNeedsParensForNonAssoc)
              cw->Out << "(";
            cw->writeOperandInternal(rhs);
            if (rhsNeedsParensForNonAssoc)
              cw->Out << ")";
            errs() << "ANDREW: Expanded BinaryOp for PHI update: " << *binOp << "\n";
          } else {
            // For non-BinaryOperator, use standard operand writing
            cw->writeOperandInternal(incomingVal);
          }
          
          cw->Out << ";\n";
          errs() << "ANDREW: Emitting do-while PHI update: " << *phi << " = " << *incomingVal << "\n";
          break;
        }
      }
    }
  }

  cw->Out << "} while(";

  // Helper lambda to write operand, checking if it should be replaced by a PHI name
  auto writeOperandOrPHI = [&](Value *op, ICmpInst *icmp = nullptr) {
    bool emitCast = icmp && cw->needsCast(op, *icmp);
    if (emitCast) {
      Type *OpTy = op->getType();
      if (OpTy->isPointerTy())
        OpTy = cw->TD->getIntPtrType(op->getContext());
      cw->Out << "((";
      cw->printSimpleType(cw->Out, OpTy, icmp->isSigned());
      cw->Out << ")";
    }
    // Check if this operand is the incoming value to a PHI node in the header
    for (auto &I : *header) {
      if (PHINode *phi = dyn_cast<PHINode>(&I)) {
        for (unsigned i = 0; i < phi->getNumIncomingValues(); i++) {
          BasicBlock *incomingBB = phi->getIncomingBlock(i);
          if (loop->contains(incomingBB) && phi->getIncomingValue(i) == op) {
            // This operand feeds into a PHI node - use the PHI name instead
            cw->Out << cw->GetValueName(phi);
            if (emitCast)
              cw->Out << ")";
            return;
          }
        }
      }
    }
    // Not a PHI incoming value, write normally
    cw->writeOperandInternal(op);
    if (emitCast)
      cw->Out << ")";
  };

  //exit condition
  CmpInst *cmp = dyn_cast<CmpInst>(condInst);
  // not the result of a comparison; single value
  if(!cmp)
    cw->writeOperand(condInst, cw->ContextCasted);
  else {
    // Handle comparison instructions (e.g., nn1 < n)
    ICmpInst *icmp = dyn_cast<ICmpInst>(condInst);
    writeOperandOrPHI(condInst->getOperand(0), icmp);
    if (icmp) {
      if (!negateCondition && (icmp->getPredicate() == ICmpInst::ICMP_NE))
        cw->Out << " < ";
      else if (negateCondition && (icmp->getPredicate() == ICmpInst::ICMP_EQ))
        cw->Out << " < ";
      else
        cw->printCmpOperator(icmp, negateCondition);
    } else if (FCmpInst *fcmp = dyn_cast<FCmpInst>(condInst)) {
      // Handle float comparisons
      CmpInst::Predicate pred = fcmp->getPredicate();
      if (negateCondition) {
        switch (pred) {
          case FCmpInst::FCMP_OEQ: pred = FCmpInst::FCMP_ONE; break;
          case FCmpInst::FCMP_ONE: pred = FCmpInst::FCMP_OEQ; break;
          case FCmpInst::FCMP_OGT: pred = FCmpInst::FCMP_OLE; break;
          case FCmpInst::FCMP_OGE: pred = FCmpInst::FCMP_OLT; break;
          case FCmpInst::FCMP_OLT: pred = FCmpInst::FCMP_OGE; break;
          case FCmpInst::FCMP_OLE: pred = FCmpInst::FCMP_OGT; break;
          default: break;
        }
      }
      switch (pred) {
        case FCmpInst::FCMP_OEQ: cw->Out << " == "; break;
        case FCmpInst::FCMP_ONE: cw->Out << " != "; break;
        case FCmpInst::FCMP_OGT: cw->Out << " > "; break;
        case FCmpInst::FCMP_OGE: cw->Out << " >= "; break;
        case FCmpInst::FCMP_OLT: cw->Out << " < "; break;
        case FCmpInst::FCMP_OLE: cw->Out << " <= "; break;
        case FCmpInst::FCMP_UEQ: cw->Out << " == "; break;
        case FCmpInst::FCMP_UNE: cw->Out << " != "; break;
        case FCmpInst::FCMP_UGT: cw->Out << " > "; break;
        case FCmpInst::FCMP_UGE: cw->Out << " >= "; break;
        case FCmpInst::FCMP_ULT: cw->Out << " < "; break;
        case FCmpInst::FCMP_ULE: cw->Out << " <= "; break;
        default:
          llvm_unreachable("Unhandled FCmpInst predicate in do-while loop condition");
      }
    }
    writeOperandOrPHI(condInst->getOperand(1), icmp);
  }

  cw->Out << ");\n";
}

// FIXME: add print for while loop
void LoopRegion::printWhileLoop() {
  errs() << "ANDREW PRINTING WHILE\n";

  BasicBlock *header = loop->getHeader();
  bool negateCondition = false;
  Instruction *condInst = cw->findCondInst(loop, negateCondition);
  BasicBlock *condBlock = nullptr;
  
  // Handle case where we can't find a condition instruction
  if (!condInst) {
    errs() << "ANDREW: condInst is null in printWhileLoop, trying header fallback\n";
    if (BranchInst *headerBr = dyn_cast<BranchInst>(header->getTerminator())) {
      if (!headerBr->isConditional() && headerBr->getNumSuccessors() == 1) {
        BasicBlock *fallbackCondBlock = headerBr->getSuccessor(0);
        if (BranchInst *fallbackCondBr =
                dyn_cast<BranchInst>(fallbackCondBlock->getTerminator())) {
          if (fallbackCondBr->isConditional()) {
            Value *fallbackCond = fallbackCondBr->getCondition();
            condInst = dyn_cast<Instruction>(fallbackCond);
            if (condInst) {
              condBlock = fallbackCondBlock;
              errs() << "ANDREW: recovered condInst from header successor block "
                     << condBlock->getName() << "\n";
            }
          }
        }
      }
    }
    if (!condInst) {
      errs() << "ANDREW: condInst fallback failed in printWhileLoop, emitting simple loop\n";
      cw->Out << "while(1) { // TODO: fix loop condition\n";
      cw->printBasicBlock(loop->getHeader());
      for (auto R : LoopBodyRegionDAG)
        R->printRegionDAG();
      cw->Out << "}\n";
      return;
    }
  }
  
  errs() << *condInst << "\n";

  std::set<Instruction *> printedLiveins;
  // TODO: move to later?
  // cond block should be latch
  std::set<Value *> condRelatedInsts;
  if (!condBlock)
    condBlock = condInst->getParent();
  cw->findCondRelatedInsts(condBlock, condRelatedInsts);
  for (auto condRelatedInst : condRelatedInsts) {
    Instruction *inst = cast<Instruction>(condRelatedInst);
    if (isa<PHINode>(inst) || isa<BranchInst>(inst) || isa<CmpInst>(inst) ||
        cw->isInlinableInst(*inst)) {
      errs() << "YEBIN: not printing condrelatedinst:" << *inst << "\n";
      continue;
    }
    errs() << "YEBIN: printing condRelatedInst: " << *inst << "\n";
    cw->printInstruction(inst);
  }
  // print things in header; this is not in another region
  cw->printBasicBlock(loop->getHeader());

  cw->Out << "while (";
  //exit condition
  CmpInst *cmp = dyn_cast<CmpInst>(condInst);
  // not the result of a comparison; single value
  if(!cmp)
    cw->writeOperand(condInst, cw->ContextCasted);
  else {
    if (ICmpInst *icmp = dyn_cast<ICmpInst>(condInst)) {
      cw->writeOperandWithCast(condInst->getOperand(0), *icmp);
      if (!negateCondition && (icmp->getPredicate() == ICmpInst::ICMP_NE))
        cw->Out << " < ";
      else if (negateCondition && (icmp->getPredicate() == ICmpInst::ICMP_EQ))
        cw->Out << " < ";
      else
        cw->printCmpOperator(icmp, negateCondition);
      cw->writeOperandWithCast(condInst->getOperand(1), *icmp);
    } else {
      cw->writeOperandInternal(condInst->getOperand(0));
      // fcmp not handled here, potentially buggy for float while loops
      cw->writeOperandInternal(condInst->getOperand(1));
    }
  }
  cw->Out << ") {\n";

  // print loop body
  for (auto R : LoopBodyRegionDAG) {
    R->printRegionDAG();
  }

  // print extra instructions in a latch other than incr and br
  errs() << "CBERegion: printing latchBB " << latchBB->getName() << "\n";
  for (auto &I : *latchBB) {
    errs() << "CBERegion: I 316: " << I << "\n";
    if (!cw->isSkipableInst(&I) && incr != &I && latchBB->getTerminator() != &I)
      cw->printInstruction(&I);
  }

  // Emit PHI node updates for loop-carried variables (other than IV)
  // This handles cases like: kk = ik; at the end of each iteration
  // Note: header is already declared above
  for (auto &I : *header) {
    if (PHINode *phi = dyn_cast<PHINode>(&I)) {
      
      // Find the value coming from inside the loop (not the initial value)
      for (unsigned i = 0; i < phi->getNumIncomingValues(); i++) {
        BasicBlock *incomingBB = phi->getIncomingBlock(i);
        if (loop->contains(incomingBB)) {
          Value *incomingVal = phi->getIncomingValue(i);
          if (incomingVal == phi) {
            errs() << "ANDREW: Skipping self-assignment for while PHI by identity: " << *phi << "\n";
            break;
          }
          std::string phiName = cw->GetValueName(phi);
          if (Instruction *incomingInst = dyn_cast<Instruction>(incomingVal)) {
            if (cw->GetValueName(incomingVal) == phiName &&
                !cw->isIVIncrement(incomingInst) &&
                !isa<BinaryOperator>(incomingInst)) {
              errs() << "ANDREW: Skipping duplicate while PHI update by stable-name match: "
                     << *phi << "\n";
              break;
            }
          }
          if (BinaryOperator *binOp = dyn_cast<BinaryOperator>(incomingVal)) {
            Value *op0 = binOp->getOperand(0);
            Value *op1 = binOp->getOperand(1);
            bool op0IsPhi = (op0 == phi);
            bool op1IsPhi = (op1 == phi);
            bool op0IsOne =
                (isa<ConstantInt>(op0) && cast<ConstantInt>(op0)->isOne()) ||
                (isa<ConstantFP>(op0) && cast<ConstantFP>(op0)->isExactlyValue(1.0));
            bool op1IsOne =
                (isa<ConstantInt>(op1) && cast<ConstantInt>(op1)->isOne()) ||
                (isa<ConstantFP>(op1) && cast<ConstantFP>(op1)->isExactlyValue(1.0));
            bool isSimpleStep =
                ((binOp->getOpcode() == Instruction::Add ||
                  binOp->getOpcode() == Instruction::FAdd) &&
                 ((op0IsPhi && op1IsOne) || (op1IsPhi && op0IsOne))) ||
                ((binOp->getOpcode() == Instruction::Sub ||
                  binOp->getOpcode() == Instruction::FSub) &&
                 (op0IsPhi && op1IsOne));
            if (!isSimpleStep) {
              errs() << "ANDREW: Skipping duplicate while reduction PHI update: " << *phi
                     << " = " << *incomingVal << "\n";
              break;
            }
            // Emit explicit step update to avoid name-coalescing turning
            // "x = x + 1" into "x = x" for IV-like carries.
            cw->Out << "  " << phiName << " = " << phiName;
            if (binOp->getOpcode() == Instruction::Sub ||
                binOp->getOpcode() == Instruction::FSub)
              cw->Out << " - 1;\n";
            else
              cw->Out << " + 1;\n";
            errs() << "ANDREW: Emitting explicit while PHI step: " << phiName
                   << "\n";
            break;
          }
          // Emit: phi_name = incoming_value;
          // Use writeOperandInternal to bypass InstsToReplaceByPhi coalescing
          cw->Out << "  " << phiName << " = ";
          cw->writeOperandInternal(incomingVal);
          cw->Out << ";\n";
          errs() << "ANDREW: Emitting PHI update: " << *phi << " = " << *incomingVal << "\n";
          break;
        }
      }
    }
  }

  cw->Out << "}\n";
}

void LoopRegion::printWhileLoopWithContinue() {
  errs() << "ANDREW:PRINTING WHILE WITH CONTINUE\n";
  
  BasicBlock *header = loop->getHeader();
  
  // For whileLoopWithContinue, the header has an unconditional branch to the condition block
  BranchInst *headerBr = dyn_cast<BranchInst>(header->getTerminator());
  if (!headerBr || headerBr->isConditional()) {
    errs() << "ERROR: Expected unconditional branch in whileLoopWithContinue header\n";
    printWhileLoop();
    return;
  }
  
  BasicBlock *condBlock = headerBr->getSuccessor(0);
  errs() << "ANDREW: condBlock is " << condBlock->getName() << "\n";
  
  // Get the condition from the condBlock
  BranchInst *condBr = dyn_cast<BranchInst>(condBlock->getTerminator());
  if (!condBr || !condBr->isConditional()) {
    errs() << "ANDREW: Expected conditional branch in condition block, falling back to printWhileLoop\n";
    printWhileLoop();
    return;
  }
  
  // Find the condition instruction and operands
  Value *condValue = condBr->getCondition();
  CmpInst *condInst = dyn_cast<CmpInst>(condValue);
  
  // Determine which successor is the body and which is the exit
  BasicBlock *bodySucc = nullptr;
  BasicBlock *exitSucc = nullptr;
  bool negateCondition = false;
  
  BasicBlock *succ0 = condBr->getSuccessor(0);
  BasicBlock *succ1 = condBr->getSuccessor(1);
  
  if (!loop->contains(succ0)) {
    exitSucc = succ0;
    bodySucc = succ1;
    negateCondition = true;  // Branch to exit on true, so negate for while condition
  } else if (!loop->contains(succ1)) {
    exitSucc = succ1;
    bodySucc = succ0;
    negateCondition = false;  // Branch to body on true
  } else {
    bool succ0LeadsToExit = reachesLoopExitFrom(succ0, nextEntryBB, loop);
    bool succ1LeadsToExit = reachesLoopExitFrom(succ1, nextEntryBB, loop);
    if (succ0LeadsToExit != succ1LeadsToExit) {
      exitSucc = succ0LeadsToExit ? succ0 : succ1;
      bodySucc = succ0LeadsToExit ? succ1 : succ0;
      negateCondition = (exitSucc == succ0);
    } else if (succ0 == nextEntryBB) {
      exitSucc = succ0;
      bodySucc = succ1;
      negateCondition = true;
    } else if (succ1 == nextEntryBB) {
      exitSucc = succ1;
      bodySucc = succ0;
      negateCondition = false;
    } else {
      errs() << "ANDREW: ambiguous condition successors in whileLoopWithContinue at "
             << condBlock->getName() << ", defaulting to succ1 as exit\n";
      exitSucc = succ1;
      bodySucc = succ0;
      negateCondition = false;
    }
  }
  
  errs() << "ANDREW: bodySucc=" << bodySucc->getName() 
         << ", exitSucc=" << (exitSucc ? exitSucc->getName() : "null")
         << ", negate=" << negateCondition << "\n";
  
  // Print PHI node initializations from header (from loop preheader)
  for (auto &I : *header) {
    if (PHINode *phi = dyn_cast<PHINode>(&I)) {
      // Find incoming value from outside the loop
      for (unsigned i = 0; i < phi->getNumIncomingValues(); i++) {
        BasicBlock *incomingBB = phi->getIncomingBlock(i);
        if (!loop->contains(incomingBB)) {
          Value *initVal = phi->getIncomingValue(i);
          cw->Out << "  " << cw->GetValueName(phi) << " = ";
          if (ConstantInt *CI = dyn_cast<ConstantInt>(initVal)) {
            cw->Out << CI->getSExtValue();
          } else {
            cw->writeOperandInternal(initVal);
          }
          cw->Out << ";\n";
          break;
        }
      }
    }
  }
  
  // Print while condition using condBlock's condition
  cw->Out << "// INSERT COMMENT LOOP: " << header->getParent()->getName() 
          << "::" << loop->getName() << "\n";
  cw->Out << "while (";
  if (condInst) {
    if (ICmpInst *icmp = dyn_cast<ICmpInst>(condInst)) {
      // Keep icmp operand signedness consistent with predicate semantics.
      cw->writeOperandWithCast(condInst->getOperand(0), *icmp);
      cw->printCmpOperator(icmp, negateCondition);
      cw->writeOperandWithCast(condInst->getOperand(1), *icmp);
    } else {
      cw->writeOperand(condInst, cw->ContextCasted);
    }
  } else {
    cw->writeOperand(condValue, cw->ContextCasted);
  }
  cw->Out << ") {\n";
  
  // Print loop body using the child regions (same approach as for loops)
  // LoopBodyRegionDAG was populated by createCBERegionDAG(startBB) in the constructor
  // The child regions already contain the full control flow including:
  // - if-else structures for continue checks
  // - nested loops
  // - function calls
  // - all other statements
  errs() << "ANDREW: Printing LoopBodyRegionDAG with " << LoopBodyRegionDAG.size() << " regions\n";
  for (auto R : LoopBodyRegionDAG)
    R->printRegionDAG();
  
  // print extra instructions in a latch other than incr and br
  errs() << "CBERegion: printing latchBB " << latchBB->getName() << "\n";
  for (auto &I : *latchBB) {
    errs() << "CBERegion: I 316: " << I << "\n";
    if (!cw->isSkipableInst(&I) && incr != &I && latchBB->getTerminator() != &I)
      cw->printInstruction(&I);
  }

  // Emit PHI node updates for loop-carried variables (other than IV)
  // This handles cases like: kk = ik; at the end of each iteration
  // Note: header is already declared above
  for (auto &I : *header) {
    if (PHINode *phi = dyn_cast<PHINode>(&I)) {
      
      // Find the value coming from inside the loop (not the initial value)
      for (unsigned i = 0; i < phi->getNumIncomingValues(); i++) {
        BasicBlock *incomingBB = phi->getIncomingBlock(i);
        if (loop->contains(incomingBB)) {
          Value *incomingVal = phi->getIncomingValue(i);
          if (incomingVal == phi) {
            errs() << "ANDREW: Skipping self-assignment for whileWithContinue PHI by identity: " << *phi << "\n";
            break;
          }
          std::string phiName = cw->GetValueName(phi);
          if (Instruction *incomingInst = dyn_cast<Instruction>(incomingVal)) {
            if (cw->GetValueName(incomingVal) == phiName &&
                !cw->isIVIncrement(incomingInst) &&
                !isa<BinaryOperator>(incomingInst)) {
              errs() << "ANDREW: Skipping duplicate whileWithContinue PHI update by stable-name match: "
                     << *phi << "\n";
              break;
            }
          }
          if (BinaryOperator *binOp = dyn_cast<BinaryOperator>(incomingVal)) {
            Value *op0 = binOp->getOperand(0);
            Value *op1 = binOp->getOperand(1);
            bool op0IsPhi = (op0 == phi);
            bool op1IsPhi = (op1 == phi);
            bool op0IsOne =
                (isa<ConstantInt>(op0) && cast<ConstantInt>(op0)->isOne()) ||
                (isa<ConstantFP>(op0) && cast<ConstantFP>(op0)->isExactlyValue(1.0));
            bool op1IsOne =
                (isa<ConstantInt>(op1) && cast<ConstantInt>(op1)->isOne()) ||
                (isa<ConstantFP>(op1) && cast<ConstantFP>(op1)->isExactlyValue(1.0));
            bool isSimpleStep =
                ((binOp->getOpcode() == Instruction::Add ||
                  binOp->getOpcode() == Instruction::FAdd) &&
                 ((op0IsPhi && op1IsOne) || (op1IsPhi && op0IsOne))) ||
                ((binOp->getOpcode() == Instruction::Sub ||
                  binOp->getOpcode() == Instruction::FSub) &&
                 (op0IsPhi && op1IsOne));
            if (!isSimpleStep) {
              errs() << "ANDREW: Skipping duplicate whileWithContinue reduction PHI update: " << *phi
                     << " = " << *incomingVal << "\n";
              break;
            }
            // Emit explicit step update to avoid phi-replacement collapse
            // turning "x = x + 1" into "x = x".
            cw->Out << "  " << phiName << " = " << phiName;
            if (binOp->getOpcode() == Instruction::Sub ||
                binOp->getOpcode() == Instruction::FSub)
              cw->Out << " - 1;\n";
            else
              cw->Out << " + 1;\n";
            errs() << "ANDREW: Emitting explicit whileWithContinue PHI step: "
                   << phiName << "\n";
            break;
          }
          // Emit: phi_name = incoming_value;
          // Use writeOperandInternal to bypass InstsToReplaceByPhi coalescing
          cw->Out << "  " << phiName << " = ";
          cw->writeOperandInternal(incomingVal);
          cw->Out << ";\n";
          errs() << "ANDREW: Emitting PHI update: " << *phi << " = " << *incomingVal << "\n";
          break;
        }
      }
    }
  }
  
  cw->Out << "}\n";
}

void CBERegion2::printRegionDAG() {
  for (auto R : CBERegionDAG) {
    R->printRegionDAG();
  }
}
;
void IfElseRegion::removeIfElseBlockFromLR(LoopRegion *lr, BasicBlock *brBB) {
  for (auto &BB : *(brBB->getParent())) {
    if (DT->dominates(brBB, &BB) && PDT->dominates(pdBB, &BB) && pdBB != &BB) {
      lr->removeBBToVisit(&BB);
    }
  }
}

LoopRegion::LoopRegion(BasicBlock *entryBB, LoopInfo *LI,
                       PostDominatorTree *PDT, DominatorTree *DT,
                       CBERegion2 *parentR, CWriter *cwriter)
    : CBERegion2{LI, PDT, DT, parentR, entryBB, cwriter} {
  // latch BB isn't considered a loop body;
  errs() << "\ncreating loop region for entryBB: " << entryBB->getName()
         << "\n";

  parentRegion = parentR;
  loop = LI->getLoopFor(entryBB);
  assert(loop && "cannot find loop for a loop region\n");

  latchBB = loop->getLoopLatch();
  if (latchBB)
    errs() << "ANDREW: latchBB is " << latchBB->getName() << "\n";
  else
    errs() << "ANDREW: latchBB is NULL (loop has multiple latches)\n";
  errs() << "YEBIN For Loop " << loop->getHeader()->getParent()->getName()
         << "::" << loop->getName() << "\n";
  this->loopType = cw->getLoopType(loop);
  errs() << "YEBIN: LOOP TYPE " << loopType << "\n";
  this->nestlevel = LI->getLoopDepth(entryBB);
  conditionBlock = nullptr;  // Initialize, will be set for whileLoopWithContinue

  nextEntryBB = cw->getSingleExitBlock(loop);
  errs() << "Unique Exit Block " << nextEntryBB->getName() << "\n";
  errs() << *nextEntryBB << "\n";
  assert(nextEntryBB && "loop doesn't have unique exit block\n");

  BasicBlock *startBB = entryBB;
  BasicBlock *succ0, *succ1;
  bool negateCondition = false;
  switch(this->loopType) {
    case doWhileLoop:
      errs() << "Found doWhileLoop " << loop->getName() << "\n";
      // For do-while loops, the header contains the first part of the body.
      // The header is printed separately in printDoWhileLoop, so startBB
      // should be the first successor of the header that needs region analysis.
      // Handle both conditional and unconditional branches at the header.
      if (BranchInst *BI = dyn_cast<BranchInst>(entryBB->getTerminator())) {
        if (BI->isUnconditional()) {
          startBB = BI->getSuccessor(0);
        } else {
          // Conditional branch - pick first successor that's in the loop and not latch
          succ0 = BI->getSuccessor(0);
          succ1 = BI->getSuccessor(1);
          if (loop->contains(succ0) && succ0 != latchBB)
            startBB = succ0;
          else if (loop->contains(succ1) && succ1 != latchBB)
            startBB = succ1;
          else
            startBB = entryBB; // fallback - header is the body
        }
      } else {
        startBB = entryBB; // No branch - unusual, use header
      }
      errs() << "DoWhile StartBB: " << startBB->getName() << "\n";
      break;
    case whileLoopWithContinue: {
      errs() << "Found whileLoopWithContinue " << loop->getName() << "\n";
      // Header has unconditional branch to condition block
      BranchInst *hBr = dyn_cast<BranchInst>(entryBB->getTerminator());
      BasicBlock *condBlock = hBr->getSuccessor(0);
      conditionBlock = condBlock;  // Store for later exclusion from body regions
      BranchInst *condBr = dyn_cast<BranchInst>(condBlock->getTerminator());
      
      // Find body successor (the one that stays in the loop)
      succ0 = condBr->getSuccessor(0);
      succ1 = condBr->getSuccessor(1);
      if (!loop->contains(succ0)) {
        startBB = succ1;  // succ0 exits, succ1 is body
      } else if (!loop->contains(succ1)) {
        startBB = succ0;  // succ1 exits, succ0 is body
      } else {
        bool succ0LeadsToExit = reachesLoopExitFrom(succ0, nextEntryBB, loop);
        bool succ1LeadsToExit = reachesLoopExitFrom(succ1, nextEntryBB, loop);
        if (succ0LeadsToExit != succ1LeadsToExit) {
          startBB = succ0LeadsToExit ? succ1 : succ0;
        } else if (succ0 == nextEntryBB) {
          startBB = succ1;
        } else if (succ1 == nextEntryBB) {
          startBB = succ0;
        } else {
          errs() << "ANDREW: ambiguous whileLoopWithContinue body successor at "
                 << condBlock->getName() << ", defaulting to succ0\n";
          startBB = succ0;
        }
      }
      errs() << "WhileWithContinue StartBB: " << startBB->getName() << "\n";
      errs() << "WhileWithContinue ConditionBlock: " << conditionBlock->getName() << "\n";
      break;
    }
    case whileLoop:
      errs() << "Found whileLoop " << loop->getName() << "\n";
      //TODO: deal with while loops
      succ0 = dyn_cast<BranchInst>(entryBB->getTerminator())->getSuccessor(0);
      succ1 = dyn_cast<BranchInst>(entryBB->getTerminator())->getSuccessor(1);
      if (succ0 == nextEntryBB)
        startBB = succ1;
      else if (succ1 == nextEntryBB)
        startBB = succ0;
      else
        assert(0 && "exit block is not from header!\n");
      errs() << "StartBB: " << startBB->getName() << "\n";    
      break;
    case forLoop:
      this->IV = cw->getInductionVariable(loop);
      this->IVInc = cw->getIVIncrement(loop, IV);
      if (LI->getLoopFor(IV->getIncomingBlock(0)) != loop)
        this->lb = IV->getIncomingValue(0);
      else if ((LI->getLoopFor(IV->getIncomingBlock(0)) == loop))
        this->incr = IV->getIncomingValue(0);
      if (LI->getLoopFor(IV->getIncomingBlock(1)) != loop)
        this->lb = IV->getIncomingValue(1);
      else if ((LI->getLoopFor(IV->getIncomingBlock(1)) == loop))
        this->incr = IV->getIncomingValue(1);
      this->ub = cw->findCondInst(loop, negateCondition)->getOperand(1);
      succ0 = dyn_cast<BranchInst>(entryBB->getTerminator())->getSuccessor(0);
      succ1 = dyn_cast<BranchInst>(entryBB->getTerminator())->getSuccessor(1);

      if (succ0 == nextEntryBB)
        startBB = succ1;
      else if (succ1 == nextEntryBB)
        startBB = succ0;
      else
        assert(0 && "exit block is not from header!\n");
      break;
    default:
      assert(0 && "Not a valid loop type! Do you have gotos?\n");
  }
  

  auto loopBBs = loop->getBlocks();
  LoopRegion *lr = getParentLoopRegion();
  for (auto BB : loopBBs) {
    if (lr)
      lr->removeBBToVisit(BB);
    // For whileLoopWithContinue, exclude the conditionBlock to prevent
    // it from being processed as a nested inner loop
    if (BB != entryBB && BB != latchBB && BB != conditionBlock)
      addBBToVisit(BB);
  }

  errs() << "YEBIN CBERegion: startBB 393: " << startBB->getParent()->getName()
         << "::" << startBB->getName() << "\n";
  createCBERegionDAG(startBB);
}

CBERegion2 *CBERegion2::createSubRegions(CBERegion2 *parentR,
                                         BasicBlock *entryBB, BasicBlock *endBB) {
  CBERegion2 *R = nullptr;
  if (!entryBB)
    return nullptr;
  // Backedge detection: if entryBB is the header of any ancestor loop region, stop recursion.
  CBERegion2 *ancestor = parentR;
  while (ancestor) {
    if (ancestor->isaLoopRegion()) {
      LoopRegion *LR = static_cast<LoopRegion*>(ancestor);

      // If the entryBB is the header of an ancestor loop region, stop traversal.
      if (LR->getLoop()->getHeader() == entryBB) {
        errs() << "ANDREW: CBERegion: detected backedge to header of loop " << entryBB->getName() << ", stopping traversal\n";
        return nullptr;
      }
      
      // ANDREW: Also check if entryBB is the condition block of a whileLoopWithContinue
      // This prevents re-processing the condition block as a nested loop
      if (LR->getConditionBlock() && LR->getConditionBlock() == entryBB) {
        errs() << "ANDREW: CBERegion: detected backedge to condition block " << entryBB->getName() << ", stopping traversal\n";
        return nullptr;
      }
    }
    ancestor = ancestor->getParentRegion();
  }

  if (!parentR) {
    errs() << "YEBIN: new topmost region\n";
  } else {
    auto parentBB = parentR->getEntryBlock();
    if (parentR->isaLoopRegion())
      errs() << "YEBIN: new region in loop " << parentBB->getParent()->getName()
             << "::" << parentBB->getName() << "\n";
    else if (parentR->isaLinearRegion())
      errs() << "YEBIN: new region in linear "
             << parentBB->getParent()->getName() << "::" << parentBB->getName()
             << "\n";
    else if (parentR->isaIfElseRegion())
      errs() << "YEBIN: new region in ifelse "
             << parentBB->getParent()->getName() << "::" << parentBB->getName()
             << "\n";
  }

  // Cycle guard: block only recursive re-entry of the same (parent region, entry).
  // Unlike global deduplication, this only suppresses currently-active recursion paths.
  BasicBlock *parentEntryBB = parentR ? parentR->getEntryBlock() : nullptr;
  std::pair<BasicBlock*, BasicBlock*> activeKey = {parentEntryBB, entryBB};
  bool activeInserted = false;
  if (parentR) {
    activeInserted = activeCreateSubregions.insert(activeKey).second;
    if (!activeInserted) {
      errs() << "ANDREW: CBERegion: detected active recursive subregion request from "
             << parentEntryBB->getName() << " to " << entryBB->getName()
             << ", pruning cywcle\n";
      return nullptr;
    }
  }

  switch (whichRegion(entryBB, LI)) {
  case 0: {
    errs() << "SUSAN: block is a linear region! " << entryBB->getName() << "\n";
    R = new LinearRegion(entryBB, parentR, LI, PDT, DT, this->cw, endBB);
    break;
  }
  case 1: {
    errs() << "SUSAN: block is an if-else region! " << entryBB->getName()
           << "\n";
    R = new IfElseRegion(entryBB, parentR, PDT, DT, LI, this->cw);
    break;
  }
  case 2: {
    errs() << "SUSAN: block is a loop region! " << entryBB->getName() << "\n";
    R = new LoopRegion(entryBB, LI, PDT, DT, parentR, this->cw);
    break;
  }
  }
  if (activeInserted)
    activeCreateSubregions.erase(activeKey);
  return R;
}

} // namespace llvm_cbe
