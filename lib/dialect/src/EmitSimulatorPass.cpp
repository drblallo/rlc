
#include <cassert>
#include <cstdint>
#include <string>
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
        mlir::rlc::FunctionOp simulatorFunction,
        mlir::OpBuilder builder
) {
    auto ip = builder.saveInsertionPoint();
    auto declaration = builder.create<mlir::rlc::DeclarationStatement>(
        simulatorFunction->getLoc(),
        action.getEntityType(),
        llvm::StringRef("actionEntity"));
    builder.createBlock(&declaration.getBody());

    auto call = builder.create<mlir::rlc::CallOp>(
        simulatorFunction->getLoc(),
        // the first result of the ActionFunction op is the function that initializes the entity.
        action->getResults().front(),
        mlir::ValueRange({}) // TODO Assuming the action has no args for now.
    );
    builder.create<mlir::rlc::Yield>(simulatorFunction->getLoc(), call.getResult(0));
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

    auto isInputLongEnough = findFunction(action->getParentOfType<mlir::ModuleOp>(), "RLC_Fuzzer_isInputLongEnough");    
    builder.createBlock(condition);
    auto actionIsDone = builder.create<mlir::rlc::CallOp>(
        action->getLoc(),
        action->getResult(1), // the second result of the actionFunction is the isDoneFunction.
        mlir::ValueRange({actionEntity})
    );
    auto longEnough = builder.create<mlir::rlc::CallOp>(
        action->getLoc(),
        isInputLongEnough,
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
static mlir::rlc::DeclarationStatement emitChosenActionDeclaration(
    mlir::rlc::ActionFunction action,
    mlir::Value actionEntity,
    mlir::OpBuilder builder
) {
    auto ip = builder.saveInsertionPoint();

    auto initAvailableSubactions = findFunction(action->getParentOfType<mlir::ModuleOp>(), "init_available_subactions");
    auto addAvailableSubaction = findFunction(action->getParentOfType<mlir::ModuleOp>(), "add_available_subaction");
    auto pickSubaction = findFunction(action->getParentOfType<mlir::ModuleOp>(), "pick_subaction");

    // let availableSubactions = Vector<Int>
    auto intVectorType = mlir::rlc::EntityType::getIdentified(
        builder.getContext(),
        "Vector",
        {mlir::rlc::IntegerType::getInt64(builder.getContext())}
    );
    auto availableSubactions = builder.create<mlir::rlc::DeclarationStatement>(
        action.getLoc(),
        intVectorType,
        llvm::StringRef("availableSubactions")
    );
    builder.createBlock(&availableSubactions.getBody());  
    auto initialized = builder.create<mlir::rlc::CallOp>(
        action->getLoc(),
        initAvailableSubactions,
        mlir::ValueRange({})
    );
    builder.create<mlir::rlc::Yield>(action->getLoc(), initialized.getResult(0));
    builder.setInsertionPointAfter(availableSubactions);

    // for each subaction,
    //      if(subAction.resumptionIndex == actionEntity.resumptionIndex)
    //          availableSubactions.push(subactionID)
    int64_t index = 0;
    action.getBody().walk([&](mlir::rlc::ActionStatement subaction) {
        auto ifStatement = builder.create<mlir::rlc::IfStatement>(action->getLoc());
        builder.createBlock(&ifStatement.getCondition());
        auto storedResumptionPoint = builder.create<mlir::rlc::MemberAccess>(
            action->getLoc(),
            actionEntity,
            0
        );
        auto subactionResumptionPoint = builder.create<mlir::rlc::Constant>(action.getLoc(), (int64_t) subaction.getResumptionPoint());
        auto eq = builder.create<mlir::rlc::EqualOp>(action->getLoc(), storedResumptionPoint, subactionResumptionPoint);
        builder.create<mlir::rlc::Yield>(ifStatement.getLoc(), eq.getResult());

        auto *trueBranch = builder.createBlock(&ifStatement.getTrueBranch());
        auto subactionIndex = builder.create<mlir::rlc::Constant>(action->getLoc(), index);
        builder.create<mlir::rlc::CallOp>(
            action->getLoc(),
            addAvailableSubaction,
            mlir::ValueRange{availableSubactions.getResult(), subactionIndex.getResult()}
        );
        builder.create<mlir::rlc::Yield>(action->getLoc());

        // construct the false branch that does nothing
        auto *falseBranch = builder.createBlock(&ifStatement.getElseBranch());
        builder.create<mlir::rlc::Yield>(ifStatement.getLoc());
        
        builder.setInsertionPointAfter(ifStatement);
        index++;
    });

    // let chosenAction = pick_subaction(availableSubactions)
    auto chosenActionDeclaration = builder.create<mlir::rlc::DeclarationStatement>(
        action->getLoc(),
        mlir::rlc::IntegerType::getInt64(builder.getContext()),
        llvm::StringRef("chosenAction")
    );
    builder.createBlock(&chosenActionDeclaration.getBody());
    auto call = builder.create<mlir::rlc::CallOp>(
        action->getLoc(),
        pickSubaction,
        mlir::ValueRange{availableSubactions.getResult()}
    );
    builder.create<mlir::rlc::Yield>(action->getLoc(), call.getResult(0));
    builder.restoreInsertionPoint(ip);
    return chosenActionDeclaration;
}

/*
    let arg1 = pickArgument(arg_1_size)
    let arg2 = pickArgument(arg_2_size)
    ...

    where arg1,arg2,... are the arguments of the subaction.
*/
static llvm::SmallVector<mlir::Value, 2> emitSubactionArgumentDeclarations(
    mlir::Value subaction,
    mlir::Value pickArgument,
    mlir::Value print,
    mlir::Location loc,
    mlir::OpBuilder builder,
    mlir::rlc::ModuleBuilder &moduleBuilder
) {
    auto ip = builder.saveInsertionPoint();
     // TODO Why/when does actionFunctionValueToActionStatement return multiple ActionStatements?
    auto actionStatement = mlir::dyn_cast<mlir::rlc::ActionStatement>(moduleBuilder.actionFunctionValueToActionStatement(subaction)[0]);
    mlir::rlc::ActionArgumentAnalysis analysis(actionStatement);
    llvm::SmallVector<mlir::Value, 2> arguments;
    int i = 0;
    for(auto input : actionStatement.getPrecondition().getArguments()) {

        assert(input.getType().isa<mlir::rlc::IntegerType>() && "Fuzzing can only handle integer arguments for now.");

        auto argDecl = builder.create<mlir::rlc::DeclarationStatement>(
            loc,
            input.getType(),
            llvm::StringRef("arg" + std::to_string(i++))
        );
        arguments.emplace_back(argDecl.getResult());

        builder.createBlock(&argDecl.getBody());
        auto input_min = builder.create<mlir::rlc::Constant>(
            loc,
            analysis.getBoundsOf(input).getMin()
        );
        auto input_max = builder.create<mlir::rlc::Constant>(
            loc,
            analysis.getBoundsOf(input).getMax()
        );
        auto call = builder.create<mlir::rlc::CallOp>(
            loc,
            pickArgument,
            mlir::ValueRange({input_min.getResult(), input_max.getResult()})
        );
        // print the value picked for the argument for debugging purposes.
        builder.create<mlir::rlc::CallOp>(loc, print, call.getResult(0));
        builder.create<mlir::rlc::Yield>(loc, call.getResult(0));
        builder.setInsertionPointAfter(argDecl);
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
    mlir::OpBuilder builder
) {
    auto ip = builder.saveInsertionPoint();
    auto pickArgument = findFunction(action->getParentOfType<mlir::ModuleOp>(), "RLC_Fuzzer_pickArgument");
    auto print = findFunction(action->getParentOfType<mlir::ModuleOp>(),"RLC_Fuzzer_print");
    auto skipFuzzInput = findFunction(action->getParentOfType<mlir::ModuleOp>(),"RLC_Fuzzer_skipInput");
    mlir::rlc::ModuleBuilder moduleBuilder(action->getParentOfType<mlir::ModuleOp>());

    llvm::SmallVector<mlir::Block*, 4> result;
    for(auto subaction : action.getActions()) {
        auto *caseBlock = builder.createBlock(parentRegion);
        result.emplace_back(caseBlock);
        auto args = emitSubactionArgumentDeclarations(subaction, pickArgument, print, action->getLoc(), builder, moduleBuilder);
        args.insert(args.begin(), actionEntity); // the first argument should be the entity itself.

        auto ifStatement = builder.create<mlir::rlc::IfStatement>(action->getLoc());
        builder.createBlock(&ifStatement.getCondition());
        auto can = builder.create<mlir::rlc::CanOp>(action->getLoc(), subaction);
        auto can_call = builder.create<mlir::rlc::CallOp>(action->getLoc(), can.getResult(), args);
        auto neg = builder.create<mlir::rlc::NotOp>(action->getLoc(), can_call->getResult(0));
        builder.create<mlir::rlc::Yield>(action -> getLoc(), neg.getResult());

        builder.createBlock(&ifStatement.getTrueBranch());
        builder.create<mlir::rlc::CallOp>(action->getLoc(), skipFuzzInput, mlir::ValueRange({}));
        auto t = builder.create<mlir::rlc::Constant>(action.getLoc(), true);
        builder.create<mlir::rlc::AssignOp>(action.getLoc(), stopFlag, t.getResult());
        builder.create<mlir::rlc::Yield>(action->getLoc());

        auto *falseBranch = builder.createBlock(&ifStatement.getElseBranch());
        builder.create<mlir::rlc::CallOp>(
            action.getLoc(),
            subaction,
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
    fun RLC_Fuzzer_simulate():
        let actionEntity = play()
        let stop = false
        while not is_done_action(actionEntity) and not stop:
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
static void emitSimulator(mlir::rlc::ActionFunction action) {
    auto loc = action.getLoc();

    mlir::OpBuilder builder(action);
    
    auto simulatorFunctionType = mlir::FunctionType::get(action.getContext(), {}, {});
    auto simulatorFunction = builder.create<mlir::rlc::FunctionOp>(
        loc,
        llvm::StringRef("RLC_Fuzzer_simulate"),
        simulatorFunctionType,
        builder.getStrArrayAttr({}));
    builder.createBlock(&simulatorFunction.getBody()); 

    auto entityDeclaration = emitActionEntityDeclaration(action, simulatorFunction, builder);

    auto stopFlag = builder.create<mlir::rlc::DeclarationStatement>(
        simulatorFunction->getLoc(),
        mlir::rlc::BoolType::get(builder.getContext()),
        llvm::StringRef("stop"));
    builder.createBlock(&stopFlag.getBody());
    auto f = builder.create<mlir::rlc::Constant>(simulatorFunction->getLoc(), false);
    builder.create<mlir::rlc::Yield>(simulatorFunction.getLoc(), f.getResult());
    builder.setInsertionPointAfter(stopFlag);

    auto whileStmt = builder.create<mlir::rlc::WhileStatement>(loc);
    emitLoopCondition(action, &whileStmt.getCondition(), entityDeclaration.getResult(), stopFlag.getResult(), builder);
    builder.createBlock(&whileStmt.getBody());
    auto chosenActionDeclaration = emitChosenActionDeclaration(action, entityDeclaration.getResult(), builder);
    auto blocks = emitSubactionBlocks(action, &whileStmt.getBody(), entityDeclaration.getResult(), stopFlag.getResult(), builder);
    builder.create<mlir::rlc::SelectBranch>(loc, chosenActionDeclaration.getResult(), blocks);
   
    builder.setInsertionPointAfter(whileStmt);
    builder.create<mlir::rlc::Yield>(loc);
}

namespace mlir::rlc
{
#define GEN_PASS_DECL_EMITSIMULATORPASS
#define GEN_PASS_DEF_EMITSIMULATORPASS
#include "rlc/dialect/Passes.inc"
	struct EmitSimulatorPass: public impl::EmitSimulatorPassBase<EmitSimulatorPass>
	{
        using impl::EmitSimulatorPassBase<EmitSimulatorPass>::EmitSimulatorPassBase;
		void getDependentDialects(mlir::DialectRegistry& registry) const override
		{
			registry.insert<mlir::rlc::RLCDialect>();
		}

		void runOnOperation() override
		{   
			ModuleOp module = getOperation();
            mlir::IRRewriter rewriter(module->getContext());

            // invoke emit_simulator on the ActionFunction with the correct unmangledName
            for(auto op :module.getOps<ActionFunction>()) {
                if(op.getUnmangledName().str() == actionToSimulate)
				    emitSimulator(op);
			}
		}
	};

}