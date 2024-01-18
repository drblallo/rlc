
#include <cassert>
#include <cstdint>
#include <string>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "rlc/dialect/ActionArgumentAnalysis.hpp"
#include "rlc/dialect/Dialect.h"
#include "rlc/dialect/Operations.hpp"
#include "rlc/dialect/SymbolTable.h"
#include "rlc/dialect/Types.hpp"

static mlir::Value findFunction(mlir::ModuleOp module, llvm::StringRef functionName) {
    for (auto op : module.getOps<mlir::rlc::FunctionOp>())
        if(op.getUnmangledName().equals(functionName))
            return op.getResult();

    assert(0 && "failed to find the function");
    return nullptr;
}

/*
    let actionEntity = action()
    TODO handle action function arguments.
*/
static mlir::rlc::DeclarationStatement emitActionEntityDeclaration(
        mlir::rlc::ActionFunction action,
        mlir::rlc::FunctionOp fuzzActionFunction,
        mlir::OpBuilder builder
) {
    auto ip = builder.saveInsertionPoint();
    auto declaration = builder.create<mlir::rlc::DeclarationStatement>(
        fuzzActionFunction->getLoc(),
        action.getEntityType(),
        llvm::StringRef("actionEntity"));
    builder.createBlock(&declaration.getBody());

    auto call = builder.create<mlir::rlc::CallOp>(
        fuzzActionFunction->getLoc(),
        // the first result of the ActionFunction op is the function that initializes the entity.
        action->getResults().front(),
        true,
        mlir::ValueRange({}) // TODO Assuming the action has no args for now.
    );
    builder.create<mlir::rlc::Yield>(fuzzActionFunction->getLoc(), call.getResult(0));
    builder.restoreInsertionPoint(ip);
    return declaration;
}

/*
    not is_done_action(actionEntity) and not stop and isInputLongEnough
*/
static void emitLoopCondition(
    mlir::rlc::ActionFunction action,
    mlir::Region *condition,
    mlir::Value actionEntity,
    mlir::Value stopFlag,
    mlir::OpBuilder builder
) {
    
    auto ip = builder.saveInsertionPoint(); 

    auto isInputLongEnough = findFunction(action->getParentOfType<mlir::ModuleOp>(), "fuzzer_is_input_long_enough");    
    builder.createBlock(condition);
    auto actionIsDone = builder.create<mlir::rlc::CallOp>(
        action->getLoc(),
        action->getResult(1), // the second result of the actionFunction is the isDoneFunction.
        true,
        mlir::ValueRange({actionEntity})
    );
    auto longEnough = builder.create<mlir::rlc::CallOp>(
        action->getLoc(),
        isInputLongEnough,
        false,
        mlir::ValueRange({})
    );
    auto neg = builder.create<mlir::rlc::NotOp>(action->getLoc(), actionIsDone.getResult(0));
    auto neg2 = builder.create<mlir::rlc::NotOp>(action->getLoc(), stopFlag);
    auto conj = builder.create<mlir::rlc::AndOp>(action->getLoc(), neg, neg2);
    auto conj2 = builder.create<mlir::rlc::AndOp>(action.getLoc(), conj.getResult(), longEnough->getResult(0));
    builder.create<mlir::rlc::Yield>(action->getLoc(), conj2.getResult());
    builder.restoreInsertionPoint(ip);
}

/*
    Emits:

        let availableSubactions = Vector<Int>
        if(subAction0.resumptionIndex == actionEntity.resumptionIndex)
            availableSubactions.push(0)
        if(subAction1.resumptionIndex == actionEntity.resumptionIndex)
            availableSubactions.push(1)
        ...
        let index = getInput(availableSubactions.size)
        let chosenAction = availableSubactions.get(index)

    We use the helper functions init_available_subactions, add_available_subaction, pick_subaction
        defined in fuzzer.utils to avoid dealing with templates here.
*/
static mlir::Value emitChosenActionDeclaration(
    mlir::rlc::ActionFunction action,
    mlir::Value actionEntity,
    mlir::rlc::ModuleBuilder &moduleBuilder,
    mlir::OpBuilder builder
) {
    auto ip = builder.saveInsertionPoint();

    auto initAvailableSubactions = findFunction(action->getParentOfType<mlir::ModuleOp>(), "fuzzer_init_available_subactions");
    auto addAvailableSubaction = findFunction(action->getParentOfType<mlir::ModuleOp>(), "fuzzer_add_available_subaction");
    auto pickSubaction = findFunction(action->getParentOfType<mlir::ModuleOp>(), "fuzzer_pick_subaction");

    // let availableSubactions = Vector<Int>
    auto intVectorType = mlir::rlc::EntityType::getIdentified(
        builder.getContext(),
        "Vector",
        {mlir::rlc::IntegerType::getInt64(builder.getContext())}
    );

    auto availableSubactions = builder.create<mlir::rlc::CallOp>(
        action->getLoc(),
        initAvailableSubactions,
        false,
        mlir::ValueRange({})
    )->getResult(0);

    // for each subaction,
    //      if(subAction.resumptionIndex == actionEntity.resumptionIndex)
    //          availableSubactions.push(subactionID)
    int64_t index = 0;
    for(auto subactionFunction : action.getActions()) {
        auto ifStatement = builder.create<mlir::rlc::IfStatement>(action->getLoc());
        builder.createBlock(&ifStatement.getCondition());
        auto storedResumptionPoint = builder.create<mlir::rlc::MemberAccess>(
            action->getLoc(),
            actionEntity,
            0
        );

        // the subactionFunction is available if the stored resumptionIndex matches any of its acitonStatements' resumptionIndex.
        auto actionStatements = moduleBuilder.actionFunctionValueToActionStatement(subactionFunction);
        mlir::Value lastOperand =
			builder.create<mlir::rlc::Constant>(action.getLoc(), false);
        for(auto *actionStatement : actionStatements) {
            auto cast = mlir::dyn_cast<mlir::rlc::ActionStatement>(actionStatement);
            auto subactionResumptionPoint = builder.create<mlir::rlc::Constant>(action.getLoc(), (int64_t) cast.getResumptionPoint());
            auto eq = builder.create<mlir::rlc::EqualOp>(action->getLoc(), storedResumptionPoint, subactionResumptionPoint);
            lastOperand = builder.create<mlir::rlc::OrOp>(action.getLoc(), lastOperand, eq.getResult());
        }
        
        builder.create<mlir::rlc::Yield>(ifStatement.getLoc(), lastOperand);

        builder.createBlock(&ifStatement.getTrueBranch());
        auto subactionIndex = builder.create<mlir::rlc::Constant>(action->getLoc(), index);
        builder.create<mlir::rlc::CallOp>(
            action->getLoc(),
            addAvailableSubaction,
            false,
            mlir::ValueRange{availableSubactions, subactionIndex.getResult()}
        );
        builder.create<mlir::rlc::Yield>(action->getLoc());

        // construct the false branch that does nothing
        auto *falseBranch = builder.createBlock(&ifStatement.getElseBranch());
        builder.create<mlir::rlc::Yield>(ifStatement.getLoc());
        
        builder.setInsertionPointAfter(ifStatement);
        index++;
    }

    // let chosenAction = pick_subaction(availableSubactions)
    auto chosenAction = builder.create<mlir::rlc::CallOp>(
        action->getLoc(),
        pickSubaction,
        false,
        mlir::ValueRange{availableSubactions}
    )->getResult(0);

    builder.restoreInsertionPoint(ip);
    return chosenAction;
}

/*
    let arg1 = pickArgument(arg_1_size)
    let arg2 = pickArgument(arg_2_size)
    ...

    where arg1,arg2,... are the arguments of the subaction.
*/
static llvm::SmallVector<mlir::Value, 2> emitSubactionArgumentDeclarations(
    mlir::Value subaction,
    mlir::Value actionEntity,
    mlir::Value pickArgument,
    mlir::Value print,
    mlir::Location loc,
    mlir::OpBuilder builder,
    mlir::rlc::ModuleBuilder &moduleBuilder
) {
    auto ip = builder.saveInsertionPoint();
    auto actionStatements = moduleBuilder.actionFunctionValueToActionStatement(subaction);

    llvm::SmallVector<mlir::Value, 2> arguments;
    
    // declare the arguments
    auto inputs = mlir::dyn_cast<mlir::FunctionType>(subaction.getType()).getInputs();
    // The first input is the actionEntity, which does not need to be declared here.
    auto inputsExcludingActionEntity = llvm::drop_begin(inputs);
    for(auto inputType : inputsExcludingActionEntity) {
        
        assert(inputType.isa<mlir::rlc::IntegerType>() && "Fuzzing can only handle integer arguments for now.");

        auto argDecl = builder.create<mlir::rlc::UninitializedConstruct>(
            loc,
            inputType
        );
        arguments.emplace_back(argDecl.getResult());
    }

    // for each action statement, if the resumeIndex matches that of the action statement, assign arguments respecting the action statement's constraints.
    auto storedResumptionPoint = builder.create<mlir::rlc::MemberAccess>(
            loc,
            actionEntity,
            0
        );
    for(auto *actionStatement : actionStatements) {
        auto cast = mlir::dyn_cast<mlir::rlc::ActionStatement>(*actionStatement);
        auto ifStatement = builder.create<mlir::rlc::IfStatement>(loc);
        builder.createBlock(&ifStatement.getCondition());
        auto subactionResumptionPoint = builder.create<mlir::rlc::Constant>(loc, (int64_t) cast.getResumptionPoint());
        auto eq = builder.create<mlir::rlc::EqualOp>(loc, storedResumptionPoint, subactionResumptionPoint);
        builder.create<mlir::rlc::Yield>(loc, eq.getResult());

        builder.createBlock(&ifStatement.getTrueBranch());
        mlir::rlc::ActionArgumentAnalysis analysis(cast);

        for(auto input : llvm::enumerate(cast.getPrecondition().getArguments())) {
            auto input_min = builder.create<mlir::rlc::Constant>(
                loc,
                analysis.getBoundsOf(input.value()).getMin()
            );
            auto input_max = builder.create<mlir::rlc::Constant>(
                loc,
                analysis.getBoundsOf(input.value()).getMax()
            );
            auto call = builder.create<mlir::rlc::CallOp>(
                loc,
                pickArgument,
                false,
                mlir::ValueRange({input_min.getResult(), input_max.getResult()})
            );
            // print the value picked for the argument for debugging purposes.
            builder.create<mlir::rlc::CallOp>(loc, print, false, call.getResult(0));
            builder.create<mlir::rlc::AssignOp>(loc, arguments[input.index()], call.getResult(0));
        }
        builder.create<mlir::rlc::Yield>(loc);

        builder.createBlock(&ifStatement.getElseBranch());
        builder.create<mlir::rlc::Yield>(loc);
        builder.setInsertionPointAfter(ifStatement);
    }
    builder.restoreInsertionPoint(ip);
    return arguments;
}

/*
    For each subaction, emits the block:
    {
        let arg1 = pickArgument(arg_1_size)
        let arg2 = pickArgument(arg_2_size)
        ...
        if( !can_subaction_function(arg1, arg2, ...))
            stop = true
        else 
            subaction_function(actionEntity, arg1, arg2, ...)
    }
*/
static llvm::SmallVector<mlir::Block*, 4> emitSubactionBlocks(
    mlir::rlc::ActionFunction action,
    mlir::Region *parentRegion,
    mlir::Value actionEntity,
    mlir::Value stopFlag,
    mlir::rlc::ModuleBuilder &moduleBuilder,
    mlir::OpBuilder builder
) {
    auto ip = builder.saveInsertionPoint();
    auto pickArgument = findFunction(action->getParentOfType<mlir::ModuleOp>(), "fuzzer_pick_argument");
    auto print = findFunction(action->getParentOfType<mlir::ModuleOp>(),"fuzzer_print");
    auto skipFuzzInput = findFunction(action->getParentOfType<mlir::ModuleOp>(),"fuzzer_skip_input");
    

    llvm::SmallVector<mlir::Block*, 4> result;
    for(auto subaction : action.getActions()) {
        auto *caseBlock = builder.createBlock(parentRegion);
        result.emplace_back(caseBlock);
        auto args = emitSubactionArgumentDeclarations(subaction, actionEntity, pickArgument, print, action->getLoc(), builder, moduleBuilder);
        args.insert(args.begin(), actionEntity); // the first argument should be the entity itself.

        auto ifStatement = builder.create<mlir::rlc::IfStatement>(action->getLoc());
        builder.createBlock(&ifStatement.getCondition());
        auto can = builder.create<mlir::rlc::CanOp>(action->getLoc(), subaction);
        auto can_call = builder.create<mlir::rlc::CallOp>(action->getLoc(), can.getResult(), false, args);
        auto neg = builder.create<mlir::rlc::NotOp>(action->getLoc(), can_call->getResult(0));
        builder.create<mlir::rlc::Yield>(action -> getLoc(), neg.getResult());

        builder.createBlock(&ifStatement.getTrueBranch());
        builder.create<mlir::rlc::CallOp>(action->getLoc(), skipFuzzInput, false, mlir::ValueRange({}));
        auto t = builder.create<mlir::rlc::Constant>(action.getLoc(), true);
        builder.create<mlir::rlc::AssignOp>(action.getLoc(), stopFlag, t.getResult());
        builder.create<mlir::rlc::Yield>(action->getLoc());

        auto *falseBranch = builder.createBlock(&ifStatement.getElseBranch());
        builder.create<mlir::rlc::CallOp>(
            action.getLoc(),
            subaction,
            false,
            args
        );
        builder.create<mlir::rlc::Yield>(action->getLoc());

        builder.setInsertionPointAfter(ifStatement);
        builder.create<mlir::rlc::Yield>(ifStatement.getLoc());
    }
    builder.restoreInsertionPoint(ip);
    return result;
}

/*
    fun fuzzer_fuzz_action_function():
        let actionEntity = play()
        let stop = false
        while not is_done_action(actionEntity) and not stop and isInputLongEnough():
            let availableSubactions = Vector<Int>
            if(subAction0.resumptionIndex == actionEntity.resumptionIndex)
                availableSubactions.push(0)
            if(subAction1.resumptionIndex == actionEntity.resumptionIndex)
                availableSubactions.push(1)
            ...

            let index = getInput(availableSubactions.size)
            let chosenAction = availableSubactions.get(index)
            switch chosenAction:
                case subaction1:
                    let arg1 = pickArgument(arg_1_size)
                    let arg2 = pickArgument(arg_2_size)
                    ...

                    subaction_function(actionEntity, arg1, arg2, ...)
                case subaction2:
                    ...
                ...
*/
static void emitFuzzActionFunction(mlir::rlc::ActionFunction action) {
    auto loc = action.getLoc();
    mlir::OpBuilder builder(action);
    mlir::rlc::ModuleBuilder moduleBuilder(action->getParentOfType<mlir::ModuleOp>());
    
    auto fuzzActionFunctionType = mlir::FunctionType::get(action.getContext(), {}, {});
    auto fuzzActionFunction = builder.create<mlir::rlc::FunctionOp>(
        loc,
        llvm::StringRef("fuzzer_fuzz_action_function"),
        fuzzActionFunctionType,
        builder.getStrArrayAttr({}),
        false
    );
    builder.createBlock(&fuzzActionFunction.getBody()); 

    auto entityDeclaration = emitActionEntityDeclaration(action, fuzzActionFunction, builder);

    auto stopFlag = builder.create<mlir::rlc::DeclarationStatement>(
        fuzzActionFunction->getLoc(),
        mlir::rlc::BoolType::get(builder.getContext()),
        llvm::StringRef("stop"));
    builder.createBlock(&stopFlag.getBody());
    auto f = builder.create<mlir::rlc::Constant>(fuzzActionFunction->getLoc(), false);
    builder.create<mlir::rlc::Yield>(fuzzActionFunction.getLoc(), f.getResult());
    builder.setInsertionPointAfter(stopFlag);

    auto whileStmt = builder.create<mlir::rlc::WhileStatement>(loc);
    emitLoopCondition(action, &whileStmt.getCondition(), entityDeclaration.getResult(), stopFlag.getResult(), builder);
    builder.createBlock(&whileStmt.getBody());
    auto chosenAction = emitChosenActionDeclaration(action, entityDeclaration.getResult(), moduleBuilder, builder);
    auto blocks = emitSubactionBlocks(action, &whileStmt.getBody(), entityDeclaration.getResult(), stopFlag.getResult(), moduleBuilder, builder);
    builder.create<mlir::rlc::SelectBranch>(loc, chosenAction, blocks);
   
    builder.setInsertionPointAfter(whileStmt);
    builder.create<mlir::rlc::Yield>(loc);
}

namespace mlir::rlc
{
#define GEN_PASS_DECL_EMITFUZZTARGETPASS
#define GEN_PASS_DEF_EMITFUZZTARGETPASS
#include "rlc/dialect/Passes.inc"
	struct EmitFuzzTargetPass: public impl::EmitFuzzTargetPassBase<EmitFuzzTargetPass>
	{
        using impl::EmitFuzzTargetPassBase<EmitFuzzTargetPass>::EmitFuzzTargetPassBase;
		void getDependentDialects(mlir::DialectRegistry& registry) const override
		{
			registry.insert<mlir::rlc::RLCDialect>();
		}

		void runOnOperation() override
		{   
			ModuleOp module = getOperation();
            mlir::IRRewriter rewriter(module->getContext());

            // invoke emitFuzzActionFunction on the ActionFunction with the correct unmangledName
            for(auto op :module.getOps<ActionFunction>()) {
                if(op.getUnmangledName().str() == actionToFuzz)
				    emitFuzzActionFunction(op);
			}
		}
	};

}