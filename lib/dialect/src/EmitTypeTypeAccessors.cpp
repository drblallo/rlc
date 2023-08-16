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
					mlir::LLVM::LLVMFunctionType::get(returnTypeTypeFieldsCount, {mlir::LLVM::LLVMPointerType::get(structTest)}));
			
			auto* blockTypeFieldsCount = opTypeFieldsCount.addEntryBlock();
			builder.setInsertionPoint(blockTypeFieldsCount, blockTypeFieldsCount->begin());
			
			mlir::Value cst0ForCounter = builder.create<mlir::LLVM::ConstantOp>(
			opTypeFieldsCount->getLoc(), builder.getI64Type(), builder.getIndexAttr(0));
			mlir::Value cst1 = builder.create<mlir::LLVM::ConstantOp>(
			opTypeFieldsCount->getLoc(), builder.getI64Type(), builder.getIndexAttr(1));
			auto pointerToInteger = builder.create<mlir::LLVM::GEPOp>(
				opTypeFieldsCount->getLoc(),
				mlir::LLVM::LLVMPointerType::get(structTest.getBody()[1]),
				blockTypeFieldsCount->getArgument(0),
				mlir::ArrayRef<mlir::Value>({ cst0ForCounter, cst1 }));

			auto loadedInteger = builder.create<mlir::LLVM::LoadOp>(op->getLoc(), pointerToInteger);
			
			builder.create<mlir::LLVM::ReturnOp>(
					opTypeFieldsCount->getLoc(), mlir::ValueRange({ loadedInteger }));
			
			


			builder.setInsertionPoint(&getOperation().getBodyRegion().front(), getOperation().getBodyRegion().front().begin());
			auto mangeledGetTypeFieldNameName = "getTypeFieldName";

			auto opTypeFieldName = builder.create<mlir::LLVM::LLVMFuncOp>(
					getOperation().getLoc(),
					"getTypeFieldName",
					mlir::LLVM::LLVMFunctionType::get(mlir::LLVM::LLVMPointerType::get(builder.getI8Type()), {mlir::LLVM::LLVMPointerType::get(structTest), builder.getI64Type()}));
			
			auto* blockTypeFieldName = opTypeFieldName.addEntryBlock();
			builder.setInsertionPoint(blockTypeFieldName, blockTypeFieldName->begin());

			mlir::Value cst0ForName = builder.create<mlir::LLVM::ConstantOp>(
			opTypeFieldName->getLoc(), builder.getI64Type(), builder.getIndexAttr(0));
			mlir::Value cst2 = builder.create<mlir::LLVM::ConstantOp>(
			opTypeFieldName->getLoc(), builder.getI64Type(), builder.getIndexAttr(2));
			
			auto pointerToArray = builder.create<mlir::LLVM::GEPOp>(
				opTypeFieldName->getLoc(),
				mlir::LLVM::LLVMPointerType::get(structTest.getBody()[2]),
				blockTypeFieldName->getArgument(0),
				mlir::ArrayRef<mlir::Value>({ cst0ForName, cst2 }));
			

			auto loadedArray = builder.create<mlir::LLVM::LoadOp>(opTypeFieldName->getLoc(), pointerToArray);

			auto pointerToArrayField = builder.create<mlir::LLVM::GEPOp>(
				opTypeFieldName->getLoc(),
				mlir::LLVM::LLVMPointerType::get(structTest.getBody()[2].cast<mlir::LLVM::LLVMPointerType>().getElementType()),
				loadedArray,
				mlir::ArrayRef<mlir::Value>({ blockTypeFieldName->getArgument(1) }));

			auto loadedPointer = builder.create<mlir::LLVM::LoadOp>(opTypeFieldName->getLoc(), pointerToArrayField);
			
			builder.create<mlir::LLVM::ReturnOp>(
					opTypeFieldName->getLoc(), mlir::ValueRange({ loadedPointer }));
			
			
		}

		private:
		mlir::TypeConverter converter;
	};

}	 // namespace mlir::rlc