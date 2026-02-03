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
  // Both exits is likely at the end of the program
  if (exitFunctionTrueBr && exitFunctionFalseBr) {
    errs() << "Both branches exit function!!\n";
  }
  // These are easier - the one that exits is an "early exit" and the fall-through continues the rest of the program.
  // Don't use the calculated postdominator!!
  if (exitFunctionTrueBr) {
    errs() << "True branch exits function!!\n";
    trueBrOnly = true; 
    // currently only works if the exit is in the start BB
    //trueBBs.insert(trueStartBB);
    // Goes to return block
    if (auto *lr = getParentLoopRegion())
      for (auto *BB : falseBBs)
        lr->removeBBToVisit(BB);
    createSubIfElseRegions(trueStartBB, brBB, falseStartBB, false);
    nextEntryBB = falseStartBB;
  }
  else if (exitFunctionFalseBr) {
    errs() << "False branch exits function!!\n";
    falseBrOnly = true;
    if (auto *lr = getParentLoopRegion())
      for (auto *BB : trueBBs)
        lr->removeBBToVisit(BB);
    createSubIfElseRegions(falseStartBB, brBB, trueStartBB, true);
    nextEntryBB = trueStartBB;
  }
  else {

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
    if (!nextRegionEntryBB)
      errs() << "Did not detect nextRegion\n";
    if (nextRegionEntryBB) {
      errs() << "SUSAN: nextRegionEntryBB " << nextRegionEntryBB->getName();
      errs() << " for region: " << *(this->loop) << "\n";
      // createCBERegionDAG(nextRegionEntryBB);
    }
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
    case forLoop:
      errs() << "Print forLoop " << loop->getName() << "...\n";
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
  cw->writeOperandInternal(condInst->getOperand(0));
  if (ICmpInst *icmp = dyn_cast<ICmpInst>(condInst)) {
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
  }
  cw->writeOperandInternal(condInst->getOperand(1));
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

  cw->Out << "} while(";

  //exit condition
  CmpInst *cmp = dyn_cast<CmpInst>(condInst);
  // not the result of a comparison; single value
  if(!cmp)
    cw->writeOperand(condInst, cw->ContextCasted);
  //cw->Out << cw->GetValueName(condInst->getOperand(0));
  //if (ICmpInst *icmp = dyn_cast<ICmpInst>(condInst)) {
  //  if (!negateCondition && (icmp->getPredicate() == ICmpInst::ICMP_NE))
  //    cw->Out << " < ";
  //  else if (negateCondition && (icmp->getPredicate() == ICmpInst::ICMP_EQ))
  //    cw->Out << " < ";
  //  else
  //    cw->printCmpOperator(icmp, negateCondition);
  //}
  //cw->writeOperandInternal(condInst->getOperand(1));

  cw->Out << ");\n";
}

// FIXME: add print for while loop
void LoopRegion::printWhileLoop() {
  errs() << "PRINTING DOWHILE\n";

  BasicBlock *header = loop->getHeader();
  bool negateCondition = false;
  Instruction *condInst = cw->findCondInst(loop, negateCondition);
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
    cw->Out << cw->GetValueName(condInst->getOperand(0));
    if (ICmpInst *icmp = dyn_cast<ICmpInst>(condInst)) {
      if (!negateCondition && (icmp->getPredicate() == ICmpInst::ICMP_NE))
        cw->Out << " < ";
      else if (negateCondition && (icmp->getPredicate() == ICmpInst::ICMP_EQ))
        cw->Out << " < ";
      else
        cw->printCmpOperator(icmp, negateCondition);
    }
    cw->writeOperandInternal(condInst->getOperand(1));
  }
  cw->Out << ") {\n";

  // print loop body
  for (auto R : LoopBodyRegionDAG) {
    R->printRegionDAG();
  }

  // print extra instructions in a latch other than incr and br
  //errs() << "CBERegion: printing latchBB " << latchBB->getName() << "\n";
  //for (auto &I : *latchBB) {
  //  errs() << "CBERegion: I 316: " << I << "\n";
  //  if (!cw->isSkipableInst(&I) && incr != &I && latchBB->getTerminator() != &I)
  //    cw->printInstruction(&I);
  //}

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
  errs() << "YEBIN For Loop " << loop->getHeader()->getParent()->getName()
         << "::" << loop->getName() << "\n";
  this->loopType = cw->getLoopType(loop);
  errs() << "YEBIN: LOOP TYPE " << loopType << "\n";
  this->nestlevel = LI->getLoopDepth(entryBB);

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
      startBB = entryBB->getUniqueSuccessor();
      assert(startBB && "Cannot find unique sucessor of header!\n");
      break;
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
    if (BB != entryBB && BB != latchBB)
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
