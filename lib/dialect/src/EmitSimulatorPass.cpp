
#include <cstdint>
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Pass/Pass.h"
#include "rlc/dialect/Dialect.h"
#include "rlc/dialect/Operations.hpp"
#include "rlc/dialect/Types.hpp"

static void emitSimulator(mlir::rlc::ActionFunction action, mlir::rlc::FunctionOp print) {
    mlir::OpBuilder builder(action);


    
    // at first, simply emit a function that calls a print(action_id) method implemented in C.
    
    // TODO this will ultimately take the fuzz input pointer and length as arguments.
    // declare the function type () -> void. 
    auto simulatorFunctionType = mlir::FunctionType::get(action.getContext(), {}, {});
    auto simulatorFunction = builder.create<mlir::rlc::FlatFunctionOp>(
        action->getLoc(),
        llvm::StringRef("RLC_simulate"),
        simulatorFunctionType,
        builder.getStrArrayAttr({})
        );
    simulatorFunction->dump();

    builder.createBlock(&simulatorFunction.getBody());

    auto seventeen = builder.create<mlir::rlc::Constant>(simulatorFunction->getLoc(), (std::int64_t) 17);
    builder.create<mlir::rlc::CallOp>(
        simulatorFunction->getLoc(),
        print.getResult(),
        mlir::ValueRange{seventeen}
    );
    builder.create<mlir::rlc::Yield>(simulatorFunction.getLoc());
}

static mlir::rlc::FunctionOp emitPrintDeclaration(mlir::ModuleOp module) {
    mlir::OpBuilder builder(module);
    builder.setInsertionPointToStart(module.getBody());

    auto print = builder.create<mlir::rlc::FunctionOp>(
			module->getLoc(),
            llvm::StringRef("print"),
			mlir::FunctionType::get(module->getContext(), {mlir::rlc::IntegerType::get(module->getContext(), 64)}, {}),
            builder.getStrArrayAttr({"message"})
	);
    return print;
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

            // TODO investigate how this needs to work with multiple modules.
            auto printFunction = emitPrintDeclaration(module);

            // invoke emit_simulator on the ActionFunction with the correct unmangledName
			llvm::SmallVector<mlir::rlc::ActionFunction, 4> actionFunctionOps;
			module.walk([&](mlir::rlc::ActionFunction op){
                if(op.getUnmangledName().str() == actionToSimulate)
				    emitSimulator(op, printFunction);
			});
		}
	};

}