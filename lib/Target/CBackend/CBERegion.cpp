#include "CBackend.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/CodeGen/TargetLowering.h"
#include "llvm/IR/BasicBlock.h"
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

  bool trueBrOnly;
  bool falseBrOnly;

  // Control flow exits in if-else block
  // Assume two ways of exiting: a return statement as the terminator
  // OR going to the return block (this block must only have a return inst)
  // BE CAREFUL OF: if-else statements that happen at end of function
  // FIXME: this will only work for single level if-else statements
  // Need more sophisticated return checking logic to handle nested statements
  // Nested statements may require a bottom-up approach
  bool exitFunctionTrueBr = isExitingFunction(trueStartBB);
  bool exitFunctionFalseBr = isExitingFunction(falseStartBB);
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
  else if (exitFunctionTrueBr && !exitFunctionFalseBr) {
    errs() << "True branch exits function!!\n";
    // ANDREW: Just emit return; statement, don't process subregions
    isFunctionReturn = true;
    negateCond = false;
    nextEntryBB = falseStartBB;
    return;
  }
  else if (exitFunctionFalseBr && !exitFunctionTrueBr) {
    errs() << "False branch exits function!!\n";
    // ANDREW: Just emit return; statement, negate condition
    isFunctionReturn = true;
    negateCond = true;
    nextEntryBB = trueStartBB;
    return;
  }
  // ANDREW: Check if either branch exits the parent loop (break statement)
  else if (getParentLoopRegion()) {
    bool exitLoopTrueBr = isExitingLoop(trueStartBB);
    bool exitLoopFalseBr = isExitingLoop(falseStartBB);
    
    if (exitLoopTrueBr && !exitLoopFalseBr) {
      errs() << "ANDREW: True branch exits loop (break)!\n";
      isLoopBreak = true;
      negateCond = false;
      nextEntryBB = falseStartBB;
      // Mark the exit block as visited so we don't process it
      if (auto *lr = getParentLoopRegion()) {
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
      if (auto *lr = getParentLoopRegion()) {
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
    bool continueLoopTrueBr = isContinuingLoop(trueStartBB);
    bool continueLoopFalseBr = isContinuingLoop(falseStartBB);
    
    if (continueLoopTrueBr && !continueLoopFalseBr) {
      errs() << "ANDREW: True branch continues loop (continue)!\n";
      isLoopContinue = true;
      negateCond = false;
      nextEntryBB = falseStartBB;
      // Mark the continue blocks as visited
      if (auto *lr = getParentLoopRegion()) {
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
      if (auto *lr = getParentLoopRegion()) {
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
  //bool trueBrOnly = noElseRegion(true);
  //bool falseBrOnly = noElseRegion(false);
  // -1: neither branch leads to return
  // 0: true branch leads to return
  // 1: false branch leads to return
  // This is weird... only works for very particular control flow
  int returnDominated = dominatedByReturn(brBB);
  if (!trueBrOnly && !falseBrOnly && returnDominated == -1) {
    trueBrOnly = (exitFunctionTrueBr && !exitFunctionFalseBr);
    falseBrOnly = (exitFunctionFalseBr && !exitFunctionTrueBr);
  }

  if (trueBrOnly && (returnDominated == -1)) {
    errs() << "SUSAN: marking only true branch\n";
    if (auto lr = getParentLoopRegion())
      for (auto BB : falseBBs)
        lr->removeBBToVisit(BB);
    createSubIfElseRegions(trueStartBB, brBB, falseStartBB, false);
    nextEntryBB = falseStartBB;
  } else if (falseBrOnly && (returnDominated == -1)) {
    errs() << "SUSAN: marking only false branch\n";
    if (auto lr = getParentLoopRegion())
      for (auto BB : trueBBs)
        lr->removeBBToVisit(BB);
    createSubIfElseRegions(falseStartBB, brBB, trueStartBB, true);
    nextEntryBB = trueStartBB;
  } else {
    errs() << "SUSAN: marking both branches\n";
    auto nextEntryBB1 =
        createSubIfElseRegions(trueStartBB, brBB, falseStartBB, false);
    auto nextEntryBB2 =
        createSubIfElseRegions(falseStartBB, brBB, trueStartBB, true);
    nextEntryBB = nextEntryBB1 ? nextEntryBB1 : nextEntryBB2;
    errs() << "CBERegion: nextEntryBB 121: " << nextEntryBB->getName() << "\n";
  }

  if (parentR && parentR->isaLoopRegion())
    removeIfElseBlockFromLR((LoopRegion *)parentR, brBB);
  errs() << "=================SUSAN: END OF marking region : "
         << br->getParent()->getName() << "==================\n";
  }
}

BasicBlock *IfElseRegion::createSubIfElseRegions(BasicBlock *start,
                                                 BasicBlock *brBlock,
                                                 BasicBlock *otherStart,
                                                 bool isElseBranch) {
  LoopRegion *lr = getParentLoopRegion();
  if (lr)
    lr->removeBBToVisit(brBlock);

  BasicBlock *currBB = start;
  // TODO: this is a hasty patch
  errs() << start->getName() << " to " << brBlock->getName() << "\n";
  errs() << "otherStart: " << otherStart->getName() << "\n";
  errs() << PDT->dominates(currBB, brBlock) << "\n";
  while (!PDT->dominates(currBB, brBlock) && currBB != otherStart) {
    CBERegion2 *subR = createSubRegions(this, currBB, otherStart);
    if (!subR) break;
    if (!isElseBranch)
      thenSubRegions.push_back(subR);
    else
      elseSubRegions.push_back(subR);
    if (isa<UnreachableInst>(currBB->getTerminator()))
      break;
    currBB = subR->getNextEntryBB();
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
    if (entryBB == this->latchBB)
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
    createCBERegionDAG(nextRegionEntryBB, parentR, endBB);
  }
}

void LinearRegion::print() {
  errs() << "Linear Region with entering block: " << getEntryBlock()->getName()
         << "\n";
  for (auto BB : BBs)
    errs() << BB->getName() << "\n";
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
void LoopRegion::print() {
  errs() << "Loop Region with entering block: " << getEntryBlock()->getName()
         << "\n";
  for (auto R : LoopBodyRegionDAG)
    R->print();
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
    
    cw->Out << "  if (";
    if (negateCond) cw->Out << "!(";
    cw->writeOperand(condInst, cw->ContextCasted);
    if (negateCond) cw->Out << ")";
    cw->Out << ") {\n";
    // Handle return type properly - emit appropriate return value
    Function *F = brBB->getParent();
    Type *retTy = F->getReturnType();
    if (retTy->isVoidTy()) {
      cw->Out << "  return;\n";
    } else if (retTy->isIntegerTy()) {
      cw->Out << "  return 0;\n";
    } else if (retTy->isFloatingPointTy()) {
      cw->Out << "  return 0.0;\n";
    } else if (retTy->isPointerTy()) {
      cw->Out << "  return NULL;\n";
    } else {
      // Fallback for other types
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

  // print If branch
  cw->Out << "  if (";
  cw->writeOperand(condInst, cw->ContextCasted);
  cw->Out << ") {";
  cw->Out << " // IFELSE MARKER: " << entryBlock->getName() << " IF\n"; 
  for (auto R : thenSubRegions)
    R->printRegionDAG();

  // print else branch
  if (!elseSubRegions.empty()) {
    errs() << "elseSubRegions : \n";
    cw->Out << "  } else {";
    cw->Out << " // IFELSE MARKER: " << entryBlock->getName() << " ELSE\n";
    for (auto R : elseSubRegions)
      R->printRegionDAG();
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

  cw->Out << "for(";

  // initiation
  cw->printTypeName(cw->Out, IV->getType(), true);
  cw->Out << " ";
  cw->Out << cw->GetValueName(IV, true) << " = ";
  if (Instruction *lbInst = dyn_cast<Instruction>(lb))
    cw->writeInstComputationInline(*lbInst);
  else
    cw->writeOperand(lb);
  cw->Out << "; ";

  // exit condition
  // Use writeOperandInternal instead of GetValueName to properly handle
  // conversion instructions and get the correct variable name
  if (ICmpInst *icmp = dyn_cast<ICmpInst>(condInst)) {
    cw->writeOperandWithCast(condInst->getOperand(0), *icmp);
    if (!negateCondition && (icmp->getPredicate() == ICmpInst::ICMP_NE))
      cw->Out << " < ";
    else if (negateCondition && (icmp->getPredicate() == ICmpInst::ICMP_EQ))
      cw->Out << " < ";
    else
      cw->printCmpOperator(icmp, negateCondition);
    cw->writeOperandWithCast(condInst->getOperand(1), *icmp);
  } else if (FCmpInst *fcmp = dyn_cast<FCmpInst>(condInst)) {
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

  // print extra instructions in a latch other than incr and br
  errs() << "CBERegion: printing latchBB " << latchBB->getName() << "\n";
  for (auto &I : *latchBB) {
    errs() << "CBERegion: I 316: " << I << "\n";
    if (!cw->isSkipableInst(&I) && incr != &I && latchBB->getTerminator() != &I)
      cw->printInstruction(&I);
  }
  

  // ANDREW: Emit PHI node updates for loop-carried variables (other than IV)
  // This handles cases like: kk = ik; at the end of each iteration
  // Note: header is already declared above
  for (auto &I : *header) {
    if (PHINode *phi = dyn_cast<PHINode>(&I)) {
      // Skip the induction variable, it's handled by the for() update
      if (phi == IV) continue;
      
      // Find the value coming from inside the loop (not the initial value)
      for (unsigned i = 0; i < phi->getNumIncomingValues(); i++) {
        BasicBlock *incomingBB = phi->getIncomingBlock(i);
        if (loop->contains(incomingBB)) {
          Value *incomingVal = phi->getIncomingValue(i);
          // Skip self-assignments (when incoming value resolves to same name as PHI)
          std::string phiName = cw->GetValueName(phi);
          std::string incomingName = cw->GetValueName(incomingVal);
          if (phiName == incomingName) {
            errs() << "ANDREW: Skipping self-assignment for PHI: " << *phi << "\n";
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
          // Skip self-assignments (when incoming value resolves to same name as PHI)
          std::string phiName = cw->GetValueName(phi);
          std::string incomingName = cw->GetValueName(incomingVal);
          if (phiName == incomingName) {
            errs() << "ANDREW: Skipping self-assignment for do-while PHI: " << *phi << "\n";
            break;
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
              case Instruction::Sub: cw->Out << " - "; break;
              case Instruction::Mul: cw->Out << " * "; break;
              case Instruction::UDiv:
              case Instruction::SDiv: cw->Out << " / "; break;
              case Instruction::URem:
              case Instruction::SRem: cw->Out << " % "; break;
              default: cw->Out << " /* unknown op */ "; break;
            }
            cw->writeOperandInternal(binOp->getOperand(1));
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
  
  // Handle case where we can't find a condition instruction
  if (!condInst) {
    errs() << "Warning: condInst is null in printWhileLoop, emitting simple loop\n";
    cw->Out << "while(1) { // TODO: fix loop condition\n";
    cw->printBasicBlock(loop->getHeader());
    for (auto R : LoopBodyRegionDAG)
      R->printRegionDAG();
    cw->Out << "}\n";
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
    if ((cw->isIVIncrement(&I)) && latchBB->getTerminator() != &I)
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
          // Skip self-assignments (when incoming value resolves to same name as PHI)
          std::string phiName = cw->GetValueName(phi);
          std::string incomingName = cw->GetValueName(incomingVal);
          if (phiName == incomingName) {
            errs() << "ANDREW: Skipping self-assignment for while PHI: " << *phi << "\n";
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
    errs() << "ERROR: Expected conditional branch in condition block\n";
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
    if (succ0 == nextEntryBB) {
      exitSucc = succ0;
      bodySucc = succ1;
      negateCondition = true;
    } else {
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
      // Get the proper variable name for the loop condition
      Value *op0 = condInst->getOperand(0);
      // If op0 is a PHI from condBlock, use the proper name
      cw->Out << cw->GetValueName(op0);
      cw->printCmpOperator(icmp, negateCondition);
      cw->writeOperandInternal(condInst->getOperand(1));
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
    if ((cw->isIVIncrement(&I)) && latchBB->getTerminator() != &I)
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
          // Skip self-assignments (when incoming value resolves to same name as PHI)
          std::string phiName = cw->GetValueName(phi);
          std::string incomingName = cw->GetValueName(incomingVal);
          if (phiName == incomingName) {
            errs() << "ANDREW: Skipping self-assignment for whileWithContinue PHI: " << *phi << "\n";
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
      } else if (succ0 == nextEntryBB) {
        startBB = succ1;
      } else {
        startBB = succ0;
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
  return R;
}

} // namespace llvm_cbe
