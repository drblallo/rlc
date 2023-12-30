
#include "rlc/dialect/Dialect.h"

#include "Dialect.inc"
#include "rlc/dialect/Operations.hpp"
#include "rlc/dialect/Types.hpp"

class TypeAliasASMInterface: public mlir::OpAsmDialectInterface
{
	mutable llvm::DenseMap<mlir::rlc::EntityType, int> counter;
	mutable int index = 0;

	public:
	using mlir::OpAsmDialectInterface::OpAsmDialectInterface;

	AliasResult getAlias(mlir::Type type, llvm::raw_ostream &OS) const final
	{
		if (auto casted = type.dyn_cast<mlir::rlc::EntityType>())
		{
			OS << casted.mangledName();
			if (counter.find(casted) == counter.end())
			{
				counter[casted] = index;
			}

			OS << "_" << counter[casted];
			return AliasResult::FinalAlias;
		}
		if (auto casted = type.dyn_cast<mlir::rlc::TraitMetaType>())
		{
			OS << "trait_" << casted.getName();
			return AliasResult::FinalAlias;
		}

		return AliasResult::NoAlias;
	}
};

void mlir::rlc::RLCDialect::initialize()
{
	registerTypes();
	registerOperations();
	addInterfaces<TypeAliasASMInterface>();
}

mlir::Operation *mlir::rlc::RLCDialect::materializeConstant(
		OpBuilder &builder, Attribute value, Type type, Location loc)
{
	if (auto boolAttr = value.dyn_cast<mlir::BoolAttr>())
	{
		if (type.isa<mlir::rlc::BoolType>())
		{
			return builder.create<mlir::rlc::Constant>(loc, boolAttr.getValue());
		}
	}

	return nullptr;
}

mlir::LogicalResult mlir::rlc::logRemark(mlir::Operation *op, llvm::Twine twine)
{
	op->getContext()->getDiagEngine().emit(
			op->getLoc(), mlir::DiagnosticSeverity::Remark)
			<< twine;
	return mlir::failure();
}

mlir::LogicalResult mlir::rlc::logError(mlir::Operation *op, llvm::Twine twine)
{
	op->getContext()->getDiagEngine().emit(
			op->getLoc(), mlir::DiagnosticSeverity::Error)
			<< twine;
	return mlir::failure();
}
