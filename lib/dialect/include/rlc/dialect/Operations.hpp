#pragma once

#include <variant>

#include "llvm/ADT/ArrayRef.h"
#include "mlir/IR/FunctionInterfaces.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeRange.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "rlc/dialect/Interfaces.hpp"
#include "rlc/dialect/Types.hpp"

namespace mlir::rlc::detail
{
	template<typename Op>
	mlir::LogicalResult handleBuiltin(
			Op op,
			mlir::IRRewriter& rewriter,
			mlir::Type expectedType,
			mlir::Type resType = nullptr)
	{
		mlir::SmallVector<mlir::Type, 4> operandTypes;
		mlir::SmallVector<mlir::Value, 4> operandValue;
		for (mlir::OpOperand& t : op.getOperation()->getOpOperands())
		{
			operandTypes.push_back(t.get().getType());
			operandValue.push_back(t.get());
		}

		if (llvm::all_of(operandTypes, [expectedType](mlir::Type in) {
					return in == expectedType;
				}))
		{
			rewriter.replaceOpWithNewOp<Op>(
					op, resType != nullptr ? resType : expectedType, operandValue);
			return mlir::success();
		}
		return mlir::failure();
	}

	template<typename Op>
	mlir::LogicalResult typeCheckInteralOp(
			Op op,
			mlir::IRRewriter& rewriter,
			mlir::rlc::SymbolTable<mlir::Value>& table,
			mlir::TypeConverter& conv,
			mlir::TypeRange accetableTypes,
			mlir::Type res = nullptr);

}	 // namespace mlir::rlc::detail

namespace mlir::rlc
{

	mlir::LogicalResult typeCheck(
			mlir::Operation& op,
			mlir::IRRewriter& rewriter,
			mlir::rlc::ValueTable& table,
			mlir::TypeConverter& typeConverter);
}

#define GET_OP_CLASSES
#include "rlc/dialect/Operations.inc"

namespace mlir::rlc::detail
{
	template<typename Op>
	mlir::LogicalResult typeCheckInteralOp(
			Op op,
			mlir::IRRewriter& rewriter,
			mlir::rlc::SymbolTable<mlir::Value>& table,
			mlir::TypeConverter& conv,
			mlir::TypeRange accetableTypes,
			mlir::Type resType)
	{
		std::string opName = ("_" + op.getOperationName().drop_front(4)).str();
		mlir::SmallPtrSet<mlir::Type, 4> set;
		mlir::SmallVector<mlir::Type, 4> operandTypes;
		mlir::SmallVector<mlir::Value, 4> operandValues;
		for (mlir::OpOperand& t : op.getOperation()->getOpOperands())
		{
			operandTypes.push_back(t.get().getType());
			set.insert(operandTypes.back());
			operandValues.push_back(t.get());
		}

		if (llvm::any_of(operandTypes, [&](mlir::Type t) {
					return t.isa<mlir::rlc::UnknownType>();
				}))
		{
			op.emitError("argument op operation had unknown type");
			return mlir::failure();
		}
		for (auto type : accetableTypes)
			if (handleBuiltin(op, rewriter, type, resType).succeeded())
				return mlir::success();

		// operands are all the same and are arrays
		if (set.size() == 1 and operandTypes.front().isa<mlir::rlc::ArrayType>())
		{
			rewriter.replaceOpWithNewOp<Op>(
					op,
					resType == nullptr ? operandTypes.front() : resType,
					operandValues);
			return mlir::success();
		}

		auto overloadSet = table.get(opName);
		for (auto entry : overloadSet)
		{
			auto candidate = entry.getType().dyn_cast<mlir::FunctionType>();
			if (not candidate)
				continue;

			if (candidate.getInputs() != operandTypes)
				continue;

			rewriter.replaceOpWithNewOp<mlir::rlc::CallOp>(op, entry, operandValues);
			return mlir::success();
		}

		op.emitError("no matching function " + opName);
		return mlir::failure();
	}

}	 // namespace mlir::rlc::detail

namespace rlc
{
	using StatementTypes = std::variant<
			mlir::rlc::StatementList,
			mlir::rlc::ExpressionStatement,
			mlir::rlc::DeclarationStatement,
			mlir::rlc::IfStatement,
			mlir::rlc::ReturnStatement,
			mlir::rlc::WhileStatement,
			mlir::rlc::ActionStatement>;
}