#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "rlc/dialect/Dialect.h"
#include "rlc/dialect/conversion/TypeConverter.h"

namespace mlir::rlc
{
#define GEN_PASS_DEF_EMITTYPETYPEACCESSORSPASS
#include "rlc/dialect/Passes.inc"
	struct EmitTypeTypeAccessors: public impl::EmitTypeTypeAccessorsPassBase<EmitTypeTypeAccessors>
	{
		void getDependentDialects(mlir::DialectRegistry& registry) const override
		{
			registry.insert<mlir::rlc::RLCDialect, mlir::cf::ControlFlowDialect>();
		}

		void runOnOperation() override
		{
			auto mangeledGetTypeNameName = mlir::rlc::mangledName(
					"getTypeName",
					mlir::FunctionType::get(
							&getContext(),
							mlir::TypeRange({  mlir::rlc::VoidType::get(&getContext())}),
							mlir::TypeRange(
									{ mlir::LLVM::LLVMPointerType::get(&getContext()) })));
			auto realGetTypeName =
					getOperation().lookupSymbol<mlir::LLVM::LLVMFuncOp>(mangeledGetTypeNameName);
			if (realGetTypeName == nullptr)
				return;

			mlir::OpBuilder builder(realGetTypeName);

			auto returnType = builder.getI32Type();
			auto op = builder.create<mlir::LLVM::LLVMFuncOp>(
					realGetTypeName.getLoc(),
					"getTypeName",
					mlir::LLVM::LLVMFunctionType::get(returnType, {}));

			auto* block = op.addEntryBlock();
			builder.setInsertionPoint(block, block->begin());

			auto call = builder.create<mlir::LLVM::CallOp>(
					realGetTypeName.getLoc(), realGetTypeName, mlir::ValueRange());

			auto res = *call.getResults().begin();
			auto trunchated = builder.create<mlir::LLVM::TruncOp>(
					realGetTypeName.getLoc(), returnType, res);
			builder.create<mlir::LLVM::ReturnOp>(
					realGetTypeName.getLoc(), mlir::ValueRange({ trunchated }));


			
			auto mangeledGetTypeFieldsCountName = mlir::rlc::mangledName(
					"getTypeFieldsCount",
					mlir::FunctionType::get(
							&getContext(),
							mlir::TypeRange({  mlir::rlc::VoidType::get(&getContext())}),
							mlir::TypeRange(
									{ mlir::rlc::IntegerType::getInt64(&getContext()) })));
			auto realGetTypeFieldsCount =
					getOperation().lookupSymbol<mlir::LLVM::LLVMFuncOp>(mangeledGetTypeFieldsCountName);
			if (realGetTypeFieldsCount == nullptr)
				return;

			mlir::OpBuilder builderTypeFieldsCount(realGetTypeFieldsCount);

			auto returnTypeTypeFieldsCount = builderTypeFieldsCount.getI32Type();
			auto opTypeFieldsCount = builderTypeFieldsCount.create<mlir::LLVM::LLVMFuncOp>(
					realGetTypeFieldsCount.getLoc(),
					"getTypeFieldsCount",
					mlir::LLVM::LLVMFunctionType::get(returnTypeTypeFieldsCount, {}));

			auto* blockTypeFieldsCount = opTypeFieldsCount.addEntryBlock();
			builderTypeFieldsCount.setInsertionPoint(blockTypeFieldsCount, blockTypeFieldsCount->begin());

			auto callTypeFieldsCount = builderTypeFieldsCount.create<mlir::LLVM::CallOp>(
					realGetTypeFieldsCount.getLoc(), realGetTypeFieldsCount, mlir::ValueRange());

			auto resTypeFieldsCount = *callTypeFieldsCount.getResults().begin();
			auto trunchatedTypeFieldsCount = builderTypeFieldsCount.create<mlir::LLVM::TruncOp>(
					realGetTypeFieldsCount.getLoc(), returnTypeTypeFieldsCount, resTypeFieldsCount);
			builderTypeFieldsCount.create<mlir::LLVM::ReturnOp>(
					realGetTypeFieldsCount.getLoc(), mlir::ValueRange({ trunchatedTypeFieldsCount }));
			


			auto mangeledGetTypeFieldNameName = mlir::rlc::mangledName(
					"getTypeFieldName",
					mlir::FunctionType::get(
							&getContext(),
							mlir::TypeRange( {  mlir::rlc::VoidType::get(&getContext()) ,mlir::rlc::IntegerType::getInt64(&getContext()) }),
							mlir::TypeRange(
									{ mlir::LLVM::LLVMPointerType::get(&getContext()) })));
			auto realGetTypeFieldName =
					getOperation().lookupSymbol<mlir::LLVM::LLVMFuncOp>(mangeledGetTypeFieldNameName);
			if (realGetTypeFieldName == nullptr)
				return;

			mlir::OpBuilder builderTypeFieldName(realGetTypeFieldName);

			auto returnTypeTypeFieldName = builderTypeFieldName.getI32Type();
			auto opTypeFieldName = builderTypeFieldName.create<mlir::LLVM::LLVMFuncOp>(
					realGetTypeFieldName.getLoc(),
					"getTypeFieldName",
					mlir::LLVM::LLVMFunctionType::get(returnTypeTypeFieldName, {}));

			auto* blockTypeFieldName = opTypeFieldName.addEntryBlock();
			builderTypeFieldName.setInsertionPoint(blockTypeFieldName, blockTypeFieldName->begin());

			auto callTypeFieldName = builderTypeFieldName.create<mlir::LLVM::CallOp>(
					realGetTypeFieldName.getLoc(), realGetTypeFieldName, mlir::ValueRange());

			auto resTypeFieldName = *callTypeFieldName.getResults().begin();
			auto trunchatedTypeFieldName = builderTypeFieldName.create<mlir::LLVM::TruncOp>(
					realGetTypeFieldName.getLoc(), returnTypeTypeFieldName, resTypeFieldName);
			builderTypeFieldName.create<mlir::LLVM::ReturnOp>(
					realGetTypeFieldName.getLoc(), mlir::ValueRange({ trunchatedTypeFieldName }));
		}

		private:
		mlir::TypeConverter converter;
	};

}	 // namespace mlir::rlc