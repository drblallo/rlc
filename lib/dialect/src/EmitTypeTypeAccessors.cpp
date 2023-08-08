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
			auto mangeledGetTypeNameName = "getTypeName";

			mlir::OpBuilder builder(getOperation());
			builder.setInsertionPoint(&getOperation().getBodyRegion().front(), getOperation().getBodyRegion().front().begin());

			auto structTest = mlir::rlc::getStructType(&getContext());
			
			auto op = builder.create<mlir::LLVM::LLVMFuncOp>(
					getOperation().getLoc(),
					"getTypeName",
					mlir::LLVM::LLVMFunctionType::get(structTest.getBody()[0], {mlir::LLVM::LLVMPointerType::get(structTest)}));
			
			auto* block = op.addEntryBlock();
			builder.setInsertionPoint(block, block->begin());

			
			
			//auto* newBlock = builder.createBlock(op->getParentRegion(), op->getParentRegion()->begin(), mlir::LLVM::LLVMPointerType::get(&getContext()), op->getLoc());
			//builder.setInsertionPoint(block, block->begin());
			//auto voidPointer = builder.create<mlir::LLVM::UndefOp>(getOperation()->getLoc(), mlir::LLVM::LLVMPointerType::get(&getContext()));
			

			//mlir::Value structValue = builder.create<mlir::LLVM::UndefOp>(op->getLoc(), structTest);
			mlir::Value cst0 = builder.create<mlir::LLVM::ConstantOp>(
			op->getLoc(), builder.getI64Type(), builder.getIndexAttr(0));
			auto pointer = builder.create<mlir::LLVM::GEPOp>(
				op->getLoc(),
				mlir::LLVM::LLVMPointerType::get(structTest.getBody()[0]),
				block->getArgument(0),
				mlir::ArrayRef<mlir::Value>({ cst0, cst0 }));

			auto loaded = builder.create<mlir::LLVM::LoadOp>(op->getLoc(), pointer);
			
			builder.create<mlir::LLVM::ReturnOp>(
					op->getLoc(), mlir::ValueRange({ loaded }));
			


			builder.setInsertionPoint(&getOperation().getBodyRegion().front(), getOperation().getBodyRegion().front().begin());
			auto mangeledGetTypeFieldsCountName = "getTypeFieldsCount";

			auto returnTypeTypeFieldsCount = builder.getI64Type();
			auto opTypeFieldsCount = builder.create<mlir::LLVM::LLVMFuncOp>(
					getOperation().getLoc(),
					"getTypeFieldsCount",
					mlir::LLVM::LLVMFunctionType::get(returnTypeTypeFieldsCount, {mlir::LLVM::LLVMPointerType::get(&getContext())}));
			/*
			auto* blockTypeFieldsCount = opTypeFieldsCount.addEntryBlock();
			builderTypeFieldsCount.setInsertionPoint(blockTypeFieldsCount, blockTypeFieldsCount->begin());

			auto callTypeFieldsCount = builderTypeFieldsCount.create<mlir::LLVM::CallOp>(
					realGetTypeFieldsCount.getLoc(), realGetTypeFieldsCount, mlir::ValueRange());

			auto resTypeFieldsCount = *callTypeFieldsCount.getResults().begin();
			auto trunchatedTypeFieldsCount = builderTypeFieldsCount.create<mlir::LLVM::TruncOp>(
					realGetTypeFieldsCount.getLoc(), returnTypeTypeFieldsCount, resTypeFieldsCount);
			builderTypeFieldsCount.create<mlir::LLVM::ReturnOp>(
					realGetTypeFieldsCount.getLoc(), mlir::ValueRange({ trunchatedTypeFieldsCount }));
			*/


			//builder.setInsertionPoint(&getOperation().getBodyRegion().front(), getOperation().getBodyRegion().front().begin());
			auto mangeledGetTypeFieldNameName = "getTypeFieldName";

			auto opTypeFieldName = builder.create<mlir::LLVM::LLVMFuncOp>(
					getOperation().getLoc(),
					"getTypeFieldName",
					mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMPointerType::get(&getContext()), {mlir::LLVM::LLVMPointerType::get(&getContext()), builder.getI64Type()}));
			/*
			auto* blockTypeFieldName = opTypeFieldName.addEntryBlock();
			builderTypeFieldName.setInsertionPoint(blockTypeFieldName, blockTypeFieldName->begin());

			auto callTypeFieldName = builderTypeFieldName.create<mlir::LLVM::CallOp>(
					realGetTypeFieldName.getLoc(), realGetTypeFieldName, mlir::ValueRange());

			auto resTypeFieldName = *callTypeFieldName.getResults().begin();
			auto trunchatedTypeFieldName = builderTypeFieldName.create<mlir::LLVM::TruncOp>(
					realGetTypeFieldName.getLoc(), returnTypeTypeFieldName, resTypeFieldName);
			builderTypeFieldName.create<mlir::LLVM::ReturnOp>(
					realGetTypeFieldName.getLoc(), mlir::ValueRange({ trunchatedTypeFieldName }));
			*/
		}

		private:
		mlir::TypeConverter converter;
	};

}	 // namespace mlir::rlc