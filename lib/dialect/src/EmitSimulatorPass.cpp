
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
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "rlc/dialect/Dialect.h"
#include "rlc/dialect/Operations.hpp"
#include "rlc/dialect/Types.hpp"

/*
    Declares the function RLC_Fuzzer_getInput(max: int64) -> int64
    The implementation of this function should return a (random | fuzz) number between 0 and max exclusive.
*/
static mlir::rlc::FunctionOp emitGetInputDeclaration(
    mlir::rlc::ActionFunction action
) {
    mlir::OpBuilder builder(action);

    auto functionType = mlir::FunctionType::get(
                action->getContext(),
                {mlir::rlc::IntegerType::getInt64(action->getContext())},
                {mlir::rlc::IntegerType::getInt64(action->getContext())}
                );

    auto getInput = builder.create<mlir::rlc::FunctionOp>(
			action->getLoc(),
            llvm::StringRef("RLC_Fuzzer_getInput"),
			functionType,
            builder.getStrArrayAttr({"max"})
	);
    return getInput;
}

/*
    Emits let actionEntity = action()
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
    Emits the condition of the main while loop.
        not is_done_action(actionEntity)
    TODO add a condition to check input length here.
*/
static void emitLoopCondition(
    mlir::rlc::ActionFunction action,
    mlir::Region *condition,
    mlir::Value actionEntity,
    mlir::OpBuilder builder
) {
    
    auto ip = builder.saveInsertionPoint(); 
    builder.createBlock(condition);
    auto actionIsDone = builder.create<mlir::rlc::CallOp>(
        action->getLoc(),
        action->getResult(1), // the second result of the actionFunction is the isDoneFunction.
        mlir::ValueRange({actionEntity})
    );
    auto neg = builder.create<mlir::rlc::NotOp>(action->getLoc(), actionIsDone.getResult(0));
    builder.create<mlir::rlc::Yield>(action->getLoc(), neg.getResult());
    builder.restoreInsertionPoint(ip);
}

/*
    Emits let chosenAction = getInput(number_of_subactions).
    TODO this should call getInput(number_of_available_subactions)
*/
static mlir::rlc::DeclarationStatement emitChosenActionDeclaration(
    mlir::rlc::ActionFunction action,
    mlir::rlc::FunctionOp getInput,
    mlir::OpBuilder builder
) {
    auto ip = builder.saveInsertionPoint();
    auto chosenActionDeclaration = builder.create<mlir::rlc::DeclarationStatement>(
        action->getLoc(),
        mlir::rlc::IntegerType::getInt64(builder.getContext()),
        llvm::StringRef("chosenAction")
    );

    builder.createBlock(&chosenActionDeclaration.getBody());
    auto numSubactions = builder.create<mlir::rlc::Constant>(
        action->getLoc(),
        (int64_t) action.getSubActionsSize());
    auto call = builder.create<mlir::rlc::CallOp>(
        action->getLoc(),
        getInput.getResult(),
        mlir::ValueRange({numSubactions.getResult()})
    );
    builder.create<mlir::rlc::Yield>(action->getLoc(), call.getResult(0));
    builder.restoreInsertionPoint(ip);
    return chosenActionDeclaration;
}

/*
    Declares the function RLC_Fuzzer_pickArgument(size:int64) -> int64
    The implementation of this function should return pick an input value for an integer argument with the given size.
    TODO pass other constraints to this function.
*/
static mlir::rlc::FunctionOp emitPickArgumentDeclaration(
    mlir::rlc::ActionFunction action
) {
    mlir::OpBuilder builder(action);

    auto functionType = mlir::FunctionType::get(
                action->getContext(),
                {mlir::rlc::IntegerType::getInt64(action->getContext())},
                {mlir::rlc::IntegerType::getInt64(action->getContext())}
                );

    auto pickArgument = builder.create<mlir::rlc::FunctionOp>(
			action->getLoc(),
            llvm::StringRef("RLC_Fuzzer_pickArgument"),
			functionType,
            builder.getStrArrayAttr({"size"})
	);
    return pickArgument;
}

/*
    Declares the function RLC_Fuzzer_print(message:int64) -> void
*/
static mlir::rlc::FunctionOp emitPrintDeclaration(
    mlir::rlc::ActionFunction action
) {
    mlir::OpBuilder builder(action);

    auto print = builder.create<mlir::rlc::FunctionOp>(
			action->getLoc(),
            llvm::StringRef("RLC_Fuzzer_print"),
			mlir::FunctionType::get(action->getContext(), {mlir::rlc::IntegerType::get(action->getContext(), 64)}, {}),
            builder.getStrArrayAttr({"message"})
	);
    return print;
}

/*
    Emits the declarations (and initializations) for each subaction argument.
*/
static llvm::SmallVector<mlir::Value, 2> emitSubactionArgumentDeclarations(
    mlir::FunctionType subactionFunctionType,
    mlir::rlc::FunctionOp pickArgument,
    mlir::rlc::FunctionOp print,
    mlir::Location loc,
    mlir::OpBuilder builder
) {
    auto ip = builder.saveInsertionPoint();
    llvm::SmallVector<mlir::Value, 2> arguments;
    int i = 0;
    for(auto inputType : subactionFunctionType.getInputs()) {
        if(i == 0) { i++; continue;} //TODO think of a more readable way to iterate over all arguments but the first.

        assert(inputType.isa<mlir::rlc::IntegerType>() && "Fuzzing can only handle integer arguments for now.");

        auto argDecl = builder.create<mlir::rlc::DeclarationStatement>(
            loc,
            inputType,
            llvm::StringRef("arg" + std::to_string(i))
        );
        arguments.emplace_back(argDecl.getResult());

        builder.createBlock(&argDecl.getBody());
        int64_t input_type_size = llvm::dyn_cast<mlir::rlc::IntegerType>(inputType).getSize();
        auto size = builder.create<mlir::rlc::Constant>(
            loc,
            (int64_t) round(pow(2,input_type_size)));
        auto call = builder.create<mlir::rlc::CallOp>(
            loc,
            pickArgument.getResult(),
            mlir::ValueRange({size.getResult()})
        );
        // print the value picked for the argument for debugging purposes.
        builder.create<mlir::rlc::CallOp>(loc, print.getResult(), call.getResult(0));
        builder.create<mlir::rlc::Yield>(loc, call.getResult(0));
        builder.setInsertionPointAfter(argDecl);
        i++;
    }
    builder.restoreInsertionPoint(ip);
    return arguments;
}


/*
    Emits the "case" block for each subaction. These blocks pick arguments for the subaction and invoke it.
*/
static llvm::SmallVector<mlir::Block*, 4> emitSubactionBlocks(
    mlir::rlc::ActionFunction action,
    mlir::Region *parentRegion,
    mlir::Value actionEntity,
    mlir::OpBuilder builder
) {
    auto ip = builder.saveInsertionPoint();
    auto pickArgument = emitPickArgumentDeclaration(action);
    auto print = emitPrintDeclaration(action);

    llvm::SmallVector<mlir::Block*, 4> result;
    for(auto subaction : action.getActions()) {
        auto functionType = llvm::dyn_cast<mlir::FunctionType>(subaction.getType());
        auto *caseBlock = builder.createBlock(parentRegion);
        result.emplace_back(caseBlock);
        auto args = emitSubactionArgumentDeclarations(functionType, pickArgument, print, action->getLoc(), builder);
        args.insert(args.begin(), actionEntity); // the first argument should be the entity itself.
        builder.create<mlir::rlc::CallOp>(
            action.getLoc(),
            subaction,
            args
        );
        builder.create<mlir::rlc::Yield>(action->getLoc());
    }
    builder.restoreInsertionPoint(ip);
    return result;
}

/*
    Emit a function that simulates the subaction by repeatedly invoking its subactions with
        random/fuzz inputs.
*/
static void emitSimulator(mlir::rlc::ActionFunction action) {
    auto getInput = emitGetInputDeclaration(action);
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
    
    auto whileStmt = builder.create<mlir::rlc::WhileStatement>(loc);
    emitLoopCondition(action, &whileStmt.getCondition(), entityDeclaration.getResult(), builder);
    builder.createBlock(&whileStmt.getBody());
    auto chosenActionDeclaration = emitChosenActionDeclaration(action, getInput, builder);
    auto blocks = emitSubactionBlocks(action, &whileStmt.getBody(), entityDeclaration.getResult(), builder);
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

            // invoke emit_simulator on the ActionFunction with the correct unmangledName
			llvm::SmallVector<mlir::rlc::ActionFunction, 4> actionFunctionOps;
			module.walk([&](mlir::rlc::ActionFunction op){
                if(op.getUnmangledName().str() == actionToSimulate)
				    emitSimulator(op);
			});
		}
	};

}