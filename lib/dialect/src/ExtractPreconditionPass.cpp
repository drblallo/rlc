#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/PatternMatch.h"
#include "rlc/dialect/Operations.hpp"
#include "rlc/dialect/Passes.hpp"

static mlir::rlc::FunctionMetadataOp emitPreconditionFunction(mlir::rlc::FunctionOp fun)
{
	mlir::IRRewriter rewriter(fun.getContext());
	rewriter.setInsertionPoint(fun);
	if (fun.getPrecondition().empty())
		return nullptr;

	auto ftype = mlir::FunctionType::get(
			fun.getContext(),
			fun.getFunctionType().getInputs(),
			{ mlir::rlc::BoolType::get(fun.getContext()) });
	auto validityFunction = rewriter.create<mlir::rlc::FunctionOp>(
			fun.getLoc(),
			("can_" + fun.getUnmangledName()).str(),
			ftype,
			fun.getArgNamesAttr());
	rewriter.cloneRegionBefore(
			fun.getPrecondition(),
			validityFunction.getBody(),
			validityFunction.getBody().begin());

	auto& yieldedConditions = validityFunction.getBody().front().back();
	rewriter.setInsertionPoint(&yieldedConditions);

	mlir::Value lastOperand =
			rewriter.create<mlir::rlc::Constant>(fun.getLoc(), true);
	for (auto value : yieldedConditions.getOperands())
		lastOperand =
				rewriter.create<mlir::rlc::AndOp>(fun.getLoc(), lastOperand, value);
	rewriter.replaceOpWithNewOp<mlir::rlc::Yield>(
			&yieldedConditions, mlir::ValueRange({ lastOperand }));

	rewriter.setInsertionPoint(validityFunction);
	return rewriter.create<mlir::rlc::FunctionMetadataOp>(
			fun.getLoc(), fun.getResult(), validityFunction.getResult());
}

static void lowerCanOps(mlir::ModuleOp module, mlir::Value callee, mlir::Value precondition) {

	// filter the canOps using the source function of this metadata
	llvm::SmallVector<mlir::rlc::CanOp, 2> canOps;
	module->walk([&] (mlir::rlc::CanOp canOp) {
		if(canOp.getCallee() == callee)
			canOps.push_back(canOp);
	});

	// replace the canOps with calls to the precondition function.
	for (auto canOp : canOps)  {
		mlir::IRRewriter rewriter(canOp->getContext());
		rewriter.setInsertionPoint(canOp);
		if(precondition == nullptr) {
			// the function has no preconditions, can is trivially true 
			rewriter.replaceOpWithNewOp<mlir::rlc::Constant>(canOp, true);
		} else {
			auto call = rewriter.create<mlir::rlc::CallOp>(
				canOp->getLoc(),
				precondition,
				canOp.getArgs()
			);
			canOp.getResult().replaceAllUsesWith(call->getResult(0));
			rewriter.eraseOp(canOp);
		}
	}
}

namespace mlir::rlc
{
#define GEN_PASS_DEF_EXTRACTPRECONDITIONPASS
#include "rlc/dialect/Passes.inc"

	struct ExtractPreconditionPass
			: impl::ExtractPreconditionPassBase<ExtractPreconditionPass>
	{
		using impl::ExtractPreconditionPassBase<
				ExtractPreconditionPass>::ExtractPreconditionPassBase;

		void runOnOperation() override
		{
			auto range = getOperation().getOps<mlir::rlc::FunctionOp>();

			llvm::SmallVector<mlir::rlc::FunctionOp, 2> ops(
					range.begin(), range.end());

			for (auto function : ops) {
				auto metadata = emitPreconditionFunction(function);
				if(metadata)
					lowerCanOps(getOperation(), function, metadata.getPreconditionFunction());
				else
				 	lowerCanOps(getOperation(), function, nullptr);
			}
		}
	};
}	 // namespace mlir::rlc

namespace mlir::rlc
{
#define GEN_PASS_DEF_STRIPFUNCTIONMETADATAPASS
#include "rlc/dialect/Passes.inc"

	struct StripFunctionMetadataPass
			: impl::StripFunctionMetadataPassBase<StripFunctionMetadataPass>
	{
		using impl::StripFunctionMetadataPassBase<
				StripFunctionMetadataPass>::StripFunctionMetadataPassBase;

		void runOnOperation() override
		{
			auto range = getOperation().getOps<mlir::rlc::FunctionMetadataOp>();

			llvm::SmallVector<mlir::rlc::FunctionMetadataOp, 2> ops(
					range.begin(), range.end());

			for (auto metadata : ops)
				metadata.erase();
		}
	};
}	 // namespace mlir::rlc
