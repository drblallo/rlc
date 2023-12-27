#include "rlc/dialect/Operations.hpp"

#include "llvm/ADT/StringExtras.h"
#include "mlir/IR/BuiltinTypes.h"
#include "rlc/dialect/Dialect.h"
#include "rlc/utils/IRange.hpp"
#define GET_OP_CLASSES
#include "./Operations.inc"

void mlir::rlc::RLCDialect::registerOperations()
{
	addOperations<
#define GET_OP_LIST
#include "./Operations.inc"
			>();
}

static mlir::LogicalResult logNote(mlir::Operation *op, llvm::Twine twine)
{
	op->getContext()->getDiagEngine().emit(
			op->getLoc(), mlir::DiagnosticSeverity::Note)
			<< twine;
	return mlir::failure();
}

static mlir::LogicalResult logError(mlir::Operation *op, llvm::Twine twine)
{
	op->getContext()->getDiagEngine().emit(
			op->getLoc(), mlir::DiagnosticSeverity::Error)
			<< twine;
	return mlir::failure();
}

static llvm::SmallVector<mlir::Operation *, 4> ops(mlir::Region &region)
{
	llvm::SmallVector<mlir::Operation *, 4> toReturn;
	for (auto &op : region.getOps())
		toReturn.push_back(&op);

	return toReturn;
}

::mlir::LogicalResult mlir::rlc::CallOp::verifySymbolUses(
		::mlir::SymbolTableCollection &symbolTable)
{
	return LogicalResult::success();
}

::mlir::LogicalResult mlir::rlc::ArrayCallOp::verifySymbolUses(
		::mlir::SymbolTableCollection &symbolTable)
{
	return LogicalResult::success();
}

mlir::LogicalResult mlir::rlc::SubActionStatement::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	for (auto *child : ops(getBody()))
	{
		if (mlir::rlc::typeCheck(*child, builder).failed())
			return mlir::failure();
	}

	mlir::rlc::Yield yield =
			mlir::cast<mlir::rlc::Yield>(getBody().front().getTerminator());
	mlir::Value frameVar = yield.getArguments()[0];
	auto underlyingType = frameVar.getType().cast<mlir::rlc::EntityType>();
	auto underlying = builder.getActionOf(underlyingType)
												.getDefiningOp<mlir::rlc::ActionFunction>();

	llvm::SmallVector<mlir::Value, 2> actionValues = underlying.getActions();

	mlir::IRRewriter &rewiter = builder.getRewriter();
	rewiter.setInsertionPoint(*this);

	if (not getName().empty())
	{
		auto decl = rewiter.create<mlir::rlc::DeclarationStatement>(
				getLoc(), underlyingType, getName());
		decl.getBody().takeBody(getBody());
		builder.getSymbolTable().add(getName(), decl);
		frameVar = decl;
	}
	else
	{
		yield.erase();
		while (not getBody().front().empty())
			getBody().front().front().moveBefore(*this);
	}

	rewiter.setInsertionPoint(*this);

	if (not getRunOnce())
	{
		auto loop = rewiter.create<mlir::rlc::WhileStatement>(getLoc());
		rewiter.createBlock(&loop.getCondition());

		auto isDone =
				builder.emitCall(*this, "is_done", mlir::ValueRange({ frameVar }))
						->getResult(0);

		auto isNotDone = rewiter.create<mlir::rlc::NotOp>(getLoc(), isDone);

		rewiter.create<mlir::rlc::Yield>(getLoc(), mlir::ValueRange({ isNotDone }));

		rewiter.createBlock(&loop.getBody());

		auto finalYield = rewiter.create<mlir::rlc::Yield>(getLoc());
		rewiter.setInsertionPoint(finalYield);
	}

	auto actions = rewiter.create<mlir::rlc::ActionsStatement>(
			getLoc(), actionValues.size());

	for (size_t i = 0; i < actionValues.size(); i++)
	{
		auto *bb = rewiter.createBlock(
				&actions.getActions()[i], actions.getActions()[i].begin());
		rewiter.setInsertionPoint(bb, bb->begin());
		mlir::Value toCall = actionValues[i];
		mlir::rlc::ActionStatement referred =
				mlir::cast<mlir::rlc::ActionStatement>(
						*builder.actionFunctionValueToActionStatement(toCall)[0]);

		llvm::SmallVector<mlir::Type, 4> resultTypes;
		for (auto type :
				 llvm::drop_begin(referred.getResultTypes(), getForwardedArgs().size()))
			resultTypes.push_back(type);

		llvm::SmallVector<mlir::Location, 4> resultLoc;
		for (auto type :
				 llvm::drop_begin(referred.getResultTypes(), getForwardedArgs().size()))
			resultLoc.push_back(actions.getLoc());

		llvm::SmallVector<mlir::Attribute> nameAttrs;
		for (auto name : llvm::drop_begin(
						 referred.getDeclaredNames(), getForwardedArgs().size()))
			nameAttrs.push_back(name);

		auto fixed = rewiter.create<mlir::rlc::ActionStatement>(
				referred.getLoc(),
				resultTypes,
				referred.getName(),
				rewiter.getArrayAttr(nameAttrs),
				referred.getId(),
				referred.getResumptionPoint());

		auto *newBody = rewiter.createBlock(
				&fixed.getPrecondition(),
				fixed.getPrecondition().begin(),
				resultTypes,
				resultLoc);

		llvm::SmallVector<mlir::Value, 4> canArgs(
				getForwardedArgs().begin(), getForwardedArgs().end());
		for (auto arg : newBody->getArguments())
			canArgs.push_back(arg);
		canArgs.insert(canArgs.begin(), frameVar);

		auto casted = rewiter.create<mlir::rlc::CanOp>(actions.getLoc(), toCall);
		auto result =
				rewiter.create<mlir::rlc::CallOp>(actions.getLoc(), casted, canArgs);
		rewiter.create<mlir::rlc::Yield>(
				actions.getLoc(), mlir::ValueRange({ result.getResult(0) }));
		rewiter.setInsertionPointAfter(fixed);

		llvm::SmallVector<mlir::Value, 4> args(
				getForwardedArgs().begin(), getForwardedArgs().end());
		for (auto result : fixed.getResults())
			args.push_back(result);
		args.insert(args.begin(), frameVar);

		rewiter.create<mlir::rlc::CallOp>(actions.getLoc(), toCall, args);

		rewiter.create<mlir::rlc::Yield>(actions.getLoc());
	}
	rewiter.setInsertionPointAfter(*this);
	rewiter.eraseOp(*this);

	return mlir::success();
}

mlir::LogicalResult mlir::rlc::EnumUse::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::ExpressionStatement::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	for (auto *op : ops(getBody()))
		if (mlir::rlc::typeCheck(*op, builder).failed())
			return mlir::failure();

	return mlir::success();
}

mlir::LogicalResult mlir::rlc::DeclarationStatement::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	auto &rewriter = builder.getRewriter();
	for (auto *child : ops(getBody()))
	{
		if (mlir::rlc::typeCheck(*child, builder).failed())
			return mlir::failure();
	}

	rewriter.setInsertionPoint(getOperation());
	auto deducedType = getBody().front().getTerminator()->getOperand(0).getType();

	auto newOne = rewriter.create<mlir::rlc::DeclarationStatement>(
			getLoc(), deducedType, getName());
	newOne.getBody().takeBody(getBody());
	rewriter.replaceOp(*this, newOne);

	builder.getSymbolTable().add(newOne.getSymName(), newOne);
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::ArrayAccess::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	if (not getValue().getType().isa<mlir::rlc::ArrayType>() and
			not getValue().getType().isa<mlir::rlc::OwningPtrType>())
	{
		return logError(
				*this,
				"Type of argument of array access expression must be a array or a "
				"owning "
				"pointer");
	}
	builder.getRewriter().replaceOpWithNewOp<mlir::rlc::ArrayAccess>(
			*this, getValue(), getMemberIndex());
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::Yield::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::ReturnStatement::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	auto &rewriter = builder.getRewriter();
	for (auto *op : ops(getBody()))
	{
		if (mlir::rlc::typeCheck(*op, builder).failed())
			return mlir::failure();
	}
	rewriter.setInsertionPoint(*this);
	auto *yield = getBody().front().getTerminator();

	auto newOne = rewriter.create<mlir::rlc::ReturnStatement>(
			getLoc(),
			yield->getNumOperands() != 0 ? yield->getOpOperand(0).get().getType()
																	 : mlir::rlc::VoidType::get(getContext()));
	newOne.getBody().takeBody(getBody());
	rewriter.eraseOp(*this);

	return mlir::success();
}

mlir::LogicalResult mlir::rlc::UnresolvedReference::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	bool usedByCall = llvm::any_of(
			getResult().getUsers(), [&](const mlir::OpOperand &operand) -> bool {
				auto call = mlir::dyn_cast<mlir::rlc::CallOp>(*operand.getOwner());
				return call and call.getCallee() == getResult();
			});
	if (usedByCall)
	{
		assert(llvm::count_if(getResult().getUsers(), [](const auto &) {
						 return true;
					 }) == 1);

		return mlir::success();
	}
	auto candidates = builder.getSymbolTable().get(getName());

	if (candidates.empty())
		return logError(*this, "No known value " + getName());

	replaceAllUsesWith(candidates.front());
	builder.getRewriter().eraseOp(*this);
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::Constant::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::ValueUpcastOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::UncheckedIsOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	auto &rewriter = builder.getRewriter();
	auto deducedType = builder.getConverter().convertType(getTypeOrTrait());
	if (deducedType == nullptr)
	{
		emitRemark("In Is expression");
		return mlir::failure();
	}

	rewriter.setInsertionPoint(getOperation());
	auto replaced = rewriter.replaceOpWithNewOp<mlir::rlc::IsOp>(
			*this, getExpression(), deducedType);

	return mlir::success();
}

mlir::LogicalResult mlir::rlc::UsingTypeOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	{
		auto _ = builder.addSymbolTable();

		for (auto *op : ops(getBody()))
			if (mlir::rlc::typeCheck(*op, builder).failed())
				return mlir::failure();
	}

	builder.getConverter().registerType(
			getName(),
			mlir::dyn_cast<mlir::rlc::Yield>(getBody().back().getTerminator())
					.getArguments()[0]
					.getType());

	builder.getRewriter().eraseOp(*this);
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::StatementList::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	auto _ = builder.addSymbolTable();

	for (auto *op : ops(getBody()))
		if (mlir::rlc::typeCheck(*op, builder).failed())
			return mlir::failure();

	return mlir::success();
}

mlir::LogicalResult mlir::rlc::UncheckedTraitDefinition::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	auto &rewriter = builder.getRewriter();
	auto _ = builder.addSymbolTable();
	builder.getConverter().registerType(
			getTemplateParameter(), getTemplateParameterType());

	llvm::SmallVector<mlir::rlc::FunctionOp, 4> Ops(
			getBody().getOps<mlir::rlc::FunctionOp>());
	for (auto op : Ops)
		if (op.typeCheckFunctionDeclaration(rewriter, builder.getConverter())
						.failed())
			return mlir::failure();

	llvm::SmallVector<mlir::StringAttr> names;
	llvm::SmallVector<mlir::FunctionType> types;

	for (auto op : getBody().getOps<mlir::rlc::FunctionOp>())
	{
		names.push_back(op.getUnmangledNameAttr());
		types.push_back(op.getFunctionType());
	}

	auto type = mlir::rlc::TraitMetaType::get(
			getContext(), getName(), getTemplateParameter(), types, names);
	rewriter.setInsertionPointAfter(*this);
	rewriter.create<mlir::rlc::TraitDefinition>(getLoc(), type);
	rewriter.eraseOp(*this);
	return mlir::success();
}

static void promoteArgumentOfIsOp(
		mlir::rlc::ModuleBuilder &builder, mlir::rlc::IfStatement op)
{
	auto &rewriter = builder.getRewriter();
	auto &table = builder.getSymbolTable();
	auto isOp = op.getCondition()
									.front()
									.getTerminator()
									->getOperand(0)
									.getDefiningOp<mlir::rlc::IsOp>();
	if (not isOp)
		return;

	rewriter.setInsertionPointToStart(&op.getTrueBranch().front());
	if (auto trait = isOp.getTypeOrTrait().dyn_cast<mlir::rlc::TraitMetaType>())
	{
		for (size_t index : ::rlc::irange(trait.getRequestedFunctionTypes().size()))
		{
			auto methodType =
					trait.getRequestedFunctionTypes()[index].cast<mlir::FunctionType>();
			auto instantiated = replaceTemplateParameter(
					methodType,
					trait.getTemplateParameterType(),
					isOp.getExpression().getType());
			auto upcastedValue = rewriter.create<mlir::rlc::TemplateInstantiationOp>(
					isOp.getLoc(),
					instantiated,
					builder.getTraitDefinition(trait).getResults()[index]);

			llvm::StringRef methodName = trait.getRequestedFunctionNames()[index];
			table.add(methodName, upcastedValue);
		}
		return;
	}

	auto name = table.lookUpValue(isOp.getExpression());
	if (name.empty())
		return;

	auto upcastedValue = rewriter.create<mlir::rlc::ValueUpcastOp>(
			isOp.getLoc(), isOp.getTypeOrTrait(), isOp.getExpression());
	table.add(name, upcastedValue);
}

mlir::LogicalResult mlir::rlc::TemplateInstantiationOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::IfStatement::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	auto &rewriter = builder.getRewriter();
	for (auto *op : ops(getCondition()))
		if (mlir::rlc::typeCheck(*op, builder).failed())
			return mlir::failure();

	if (not getCondition()
							.front()
							.getTerminator()
							->getOperand(0)
							.getType()
							.isa<mlir::rlc::BoolType>())
	{
		return logError(*this, "While loop condition type must be Bool");
		return mlir::failure();
	}

	{
		auto _ = builder.addSymbolTable();
		promoteArgumentOfIsOp(builder, *this);
		for (auto *op : ops(getTrueBranch()))
			if (mlir::rlc::typeCheck(*op, builder).failed())
				return mlir::failure();
	}

	{
		auto _ = builder.addSymbolTable();
		for (auto *op : ops(getElseBranch()))
			if (mlir::rlc::typeCheck(*op, builder).failed())
				return mlir::failure();
	}

	return mlir::success();
}

mlir::LogicalResult mlir::rlc::ArrayCallOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	assert(
			(getNumResults() == 0 or
			 not getResult(0).getType().isa<mlir::rlc::UnknownType>()) and
			"unuspported, array calls should be only emitted with a correct type "
			"already");
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::CallOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	auto &rewriter = builder.getRewriter();

	mlir::Operation *newCall = nullptr;
	auto unresolvedCallee =
			getCallee().getDefiningOp<mlir::rlc::UnresolvedReference>();
	if (unresolvedCallee)
	{
		rewriter.setInsertionPoint(getOperation());
		newCall = builder.emitCall(*this, unresolvedCallee.getName(), getArgs());
		if (newCall == nullptr)
			return mlir::failure();
	}
	else
	{
		if (not getCallee().getType().isa<mlir::FunctionType>())
		{
			return logError(*this, "Cannot call non function type");
		}

		newCall =
				rewriter.create<mlir::rlc::CallOp>(getLoc(), getCallee(), getArgs());
	}

	if (newCall->getNumResults() != 0)
	{
		rewriter.replaceOp(*this, newCall->getResults());
		if (unresolvedCallee)
			rewriter.eraseOp(unresolvedCallee);
		return mlir::success();
	}

	rewriter.eraseOp(*this);
	if (unresolvedCallee)
		rewriter.eraseOp(unresolvedCallee);
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::CanOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	auto &rewriter = builder.getRewriter();
	rewriter.replaceOpWithNewOp<CanOp>(*this, getCallee());
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::ForFieldStatement::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	auto &rewriter = builder.getRewriter();
	for (auto *op : ops(getCondition()))
		if (mlir::rlc::typeCheck(*op, builder).failed())
			return mlir::failure();

	auto yield = mlir::cast<mlir::rlc::Yield>(
			getCondition().getBlocks().back().getTerminator());
	if (yield.getArguments().size() != getNames().size())
	{
		return logError(
				*this,
				"Missmatched count between for induction variables and for arguments");
	}

	for (auto argument : yield.getArguments())
	{
		if (argument.getType() != yield.getArguments()[0].getType())
		{
			auto _ = logError(
					*this,
					"for field statement does not support expressions with "
					"different types");
			return mlir::failure();
		}
	}

	auto _ = builder.addSymbolTable();
	for (auto [argument, name] : llvm::zip(getBody().getArguments(), getNames()))
		builder.getSymbolTable().add(name.cast<mlir::StringAttr>(), argument);

	for (auto *op : ops(getBody()))
		if (mlir::rlc::typeCheck(*op, builder).failed())
			return mlir::failure();

	return mlir::success();
}

mlir::LogicalResult mlir::rlc::ActionsStatement::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	auto &rewriter = builder.getRewriter();

	for (auto &region : getActions())
	{
		auto _ = builder.addSymbolTable();
		for (auto *op : ops(region))
			if (mlir::rlc::typeCheck(*op, builder).failed())
				return mlir::failure();
	}

	return mlir::success();
}

mlir::LogicalResult mlir::rlc::WhileStatement::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	auto &rewriter = builder.getRewriter();
	for (auto *op : ops(getCondition()))
		if (mlir::rlc::typeCheck(*op, builder).failed())
			return mlir::failure();

	if (not getCondition()
							.front()
							.getTerminator()
							->getOperand(0)
							.getType()
							.isa<mlir::rlc::BoolType>())
	{
		return logError(*this, "While loop condition is not boolean");
	}

	auto _ = builder.addSymbolTable();
	for (auto *op : ops(getBody()))
		if (mlir::rlc::typeCheck(*op, builder).failed())
			return mlir::failure();

	return mlir::success();
}

mlir::LogicalResult mlir::rlc::ConstructOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	auto deducedType = builder.getConverter().convertType(getType());
	auto &rewriter = builder.getRewriter();
	if (deducedType == nullptr)
	{
		emitRemark("in construction expression");
		return mlir::failure();
	}

	rewriter.replaceOpWithNewOp<mlir::rlc::ConstructOp>(*this, deducedType);

	return mlir::success();
}

mlir::LogicalResult mlir::rlc::CastOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::ActionStatement::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	auto &rewriter = builder.getRewriter();
	llvm::SmallVector<mlir::Type> deducedTypes;

	for (auto &operand : getPrecondition().front().getArguments())
	{
		auto converted = builder.getConverter().convertType(operand.getType());
		if (not converted)
		{
			getOperation()->emitRemark("in of argument of action statement");
			return mlir::failure();
		}
		operand.setType(converted);
	}

	for (auto result : getResultTypes())
	{
		auto deduced = builder.getConverter().convertType(result);
		if (deduced == nullptr)
			return mlir::failure();
		deducedTypes.push_back(deduced);
	}

	llvm::SmallVector<mlir::Operation *, 4> ToCheck;
	for (auto &ops : getPrecondition().getOps())
	{
		ToCheck.push_back(&ops);
	}

	{
		auto _ = builder.addSymbolTable();
		for (auto [name, res] :
				 llvm::zip(getDeclaredNames(), getPrecondition().getArguments()))
			builder.getSymbolTable().add(name.cast<mlir::StringAttr>(), res);

		for (auto *op : ToCheck)
		{
			if (mlir::rlc::typeCheck(*op, builder).failed())
				return mlir::failure();
		}
	}

	rewriter.setInsertionPoint(*this);
	auto newDecl = rewriter.create<mlir::rlc::ActionStatement>(
			getLoc(),
			deducedTypes,
			getName(),
			getDeclaredNames(),
			getId(),
			getResumptionPoint());
	newDecl.getPrecondition().takeBody(getPrecondition());
	rewriter.replaceOp(getOperation(), newDecl.getResults());

	for (auto [name, res] :
			 llvm::zip(newDecl.getDeclaredNames(), newDecl.getResults()))
		builder.getSymbolTable().add(name.cast<mlir::StringAttr>(), res);
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::AsByteArrayOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	int64_t size = 0;
	if (getLhs().getType().isa<mlir::rlc::BoolType>())
	{
		size = 1;
	}
	else if (getLhs().getType().isa<mlir::rlc::FloatType>())
	{
		size = 8;
	}
	else if (auto casted = getLhs().getType().dyn_cast<mlir::rlc::IntegerType>())
	{
		size = casted.getSize() / 8;
	}
	else
	{
		return logError(*this, "Input of to_byte_array must be a primitive type");
	}

	builder.getRewriter().replaceOpWithNewOp<mlir::rlc::AsByteArrayOp>(
			*this,
			mlir::rlc::ArrayType::get(
					getContext(), mlir::rlc::IntegerType::getInt8(getContext()), size),
			getLhs());
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::FromByteArrayOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	if (not getLhs().getType().isa<mlir::rlc::ArrayType>())
	{
		return logError(
				*this, "Builtin from byte_array_argument must be a byte array");
		return mlir::failure();
	}
	auto castedInput = getLhs().getType().cast<mlir::rlc::ArrayType>();
	if (castedInput.getUnderlying() !=
			mlir::rlc::IntegerType::getInt8(getContext()))
	{
		return logError(
				*this, "builtin from_byte_array argument must be a byte array");
	}

	auto converted = builder.getConverter().convertType(getResult().getType());

	if (converted.isa<mlir::rlc::FloatType>())
	{
		if (castedInput.getArraySize() != 8)
		{
			return logError(
					*this,
					"Builtin from_byte_array to float must have a array of 8 bytes "
					"as input");
		}
		builder.getRewriter().replaceOpWithNewOp<mlir::rlc::FromByteArrayOp>(
				*this, converted, getLhs());
		return mlir::success();
	}

	if (converted.isa<mlir::rlc::BoolType>())
	{
		if (castedInput.getArraySize() != 1)
		{
			return logError(
					*this,
					"Builtin from_byte_array to bool must have a array of 1 bytes "
					"as input");
		}
		builder.getRewriter().replaceOpWithNewOp<mlir::rlc::FromByteArrayOp>(
				*this, converted, getLhs());
		return mlir::success();
	}

	if (auto casted = converted.dyn_cast<mlir::rlc::IntegerType>())
	{
		if (castedInput.getArraySize() != casted.getSize() / 8)
		{
			return logError(
					*this,
					std::string(
							"Builtin from_byte_array to integer must have a array of ") +
							llvm::Twine((casted.getSize() / 8)) +
							std::string(" bytes as input"));
		}
		builder.getRewriter().replaceOpWithNewOp<mlir::rlc::FromByteArrayOp>(
				*this, casted, getLhs());
		return mlir::success();
	}
	return logError(
			*this,
			"Cannot convert byte array to desiderated output, only primitive types "
			"are supported");
}

mlir::LogicalResult mlir::rlc::InitOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	llvm::SmallVector<mlir::Type> acceptable;
	acceptable.push_back(mlir::rlc::IntegerType::getInt64(this->getContext()));
	acceptable.push_back(mlir::rlc::IntegerType::getInt8(this->getContext()));
	acceptable.push_back(mlir::rlc::BoolType::get(this->getContext()));
	acceptable.push_back(mlir::rlc::FloatType::get(this->getContext()));
	return mlir::rlc::detail::typeCheckInteralOp(
			*this, builder, acceptable, mlir::rlc::VoidType::get(this->getContext()));
}

static bool initializerListTypeIsValid(mlir::Type t)
{
	if (t.isa<mlir::rlc::IntegerType>())
	{
		return true;
	}
	if (t.isa<mlir::rlc::BoolType>())
	{
		return true;
	}
	if (t.isa<mlir::rlc::FloatType>())
	{
		return true;
	}

	if (auto type = mlir::dyn_cast<mlir::rlc::ArrayType>(t))
	{
		return initializerListTypeIsValid(type.getUnderlying());
	}
	return false;
}

mlir::LogicalResult mlir::rlc::InitializerListOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	for (auto element : getArgs())
	{
		if (element.getType() != getArgs()[0].getType())
		{
			emitOpError("initializer list has arguments of different type");
			element.getDefiningOp()->emitRemark("missmatched argument here");
			return mlir::failure();
		}
	}

	auto type = mlir::rlc::ArrayType::get(
			getContext(), getArgs()[0].getType(), getArgs().size());

	// the reason we only accept this types is because they are trivially
	// copiable and the back. if the backend to mlir invoked copy assigment
	// operators correctly there would not be a need for this.
	if (not initializerListTypeIsValid(type))
	{
		emitOpError("only acceptable types in initializer list are primitive types "
								"or arrays of primitive types");
		return mlir::failure();
	}

	builder.getRewriter().replaceOpWithNewOp<mlir::rlc::InitializerListOp>(
			*this, type, getArgs());

	return mlir::success();
}

mlir::LogicalResult mlir::rlc::ImplicitAssignOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	builder.getRewriter().replaceOpWithNewOp<mlir::rlc::ImplicitAssignOp>(
			*this, getLhs(), getRhs());
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::MallocOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	auto converted = builder.getConverter().convertType(getResult().getType());
	if (not converted)
	{
		return mlir::failure();
	}

	builder.getRewriter().replaceOpWithNewOp<mlir::rlc::MallocOp>(
			*this, converted, this->getSize());

	return mlir::success();
}

mlir::LogicalResult mlir::rlc::DestroyOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::FreeOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::BuiltinAssignOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	builder.getRewriter().replaceOpWithNewOp<mlir::rlc::BuiltinAssignOp>(
			*this, getLhs(), getRhs());
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::AssignOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	builder.getRewriter().replaceOpWithNewOp<mlir::rlc::AssignOp>(
			*this, getLhs(), getRhs());
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::MemberAccess::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	builder.getRewriter().replaceOpWithNewOp<mlir::rlc::MemberAccess>(
			*this, getValue(), getMemberIndex());
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::IntegerLiteralUse::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	if (not getInputType()
							.cast<mlir::rlc::TemplateParameterType>()
							.getIsIntLiteral())
	{
		return logError(
				*this,
				"Input type of int literal must be a template parameter literal");
	}
	return mlir::success();
}

mlir::LogicalResult mlir::rlc::UnresolvedMemberAccess::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	auto structType = getValue().getType().dyn_cast<mlir::rlc::EntityType>();
	if (structType == nullptr)
	{
		return logError(*this, "Members of non-entity types cannot be accessed");
	}

	for (const auto &index : llvm::enumerate(structType.getFieldNames()))
	{
		if (index.value() != getMemberName())
			continue;

		builder.getRewriter().replaceOpWithNewOp<mlir::rlc::MemberAccess>(
				*this, getValue(), index.index());
		return mlir::success();
	}

	return logError(
			*this,
			"no known member " + getMemberName() + " in struct " +
					structType.getName());
}

mlir::LogicalResult mlir::rlc::IsAlternativeTypeOp::typeCheck(
		mlir::rlc::ModuleBuilder &builder)
{
	return mlir::LogicalResult::success();
}

mlir::LogicalResult mlir::rlc::typeCheck(
		mlir::Operation &op, mlir::rlc::ModuleBuilder &builder)
{
	if (not op.hasTrait<mlir::rlc::TypeCheckable::Trait>())
	{
		op.emitOpError("does not implement type check");
		return mlir::failure();
	}

	builder.getRewriter().setInsertionPoint(&op);
	if (mlir::cast<mlir::rlc::TypeCheckable>(op).typeCheck(builder).failed())
		return mlir::failure();

	return mlir::LogicalResult::success();
}

mlir::LogicalResult mlir::rlc::FunctionOp::typeCheckFunctionDeclaration(
		mlir::IRRewriter &rewriter, mlir::rlc::RLCTypeConverter &converter)
{
	rewriter.setInsertionPoint(*this);
	auto scopedConverter = mlir::rlc::RLCTypeConverter(&converter);
	llvm::SmallVector<mlir::Type, 2> checkedTemplateParameters;
	for (auto parameter : getTemplateParameters())
	{
		auto unchecked = parameter.cast<mlir::TypeAttr>()
												 .getValue()
												 .cast<mlir::rlc::UncheckedTemplateParameterType>();

		auto checkedParameterType = converter.convertType(unchecked);
		if (not checkedParameterType)
		{
			emitRemark("in function definition");
			return mlir::failure();
		}
		checkedTemplateParameters.push_back(checkedParameterType);
		auto actualType =
				checkedParameterType.cast<mlir::rlc::TemplateParameterType>();
		scopedConverter.registerType(actualType.getName(), actualType);
	}

	auto deducedType = scopedConverter.convertType(getFunctionType());
	if (deducedType == nullptr)
	{
		emitRemark("in function declaration " + getUnmangledName());
		return mlir::failure();
	}
	assert(deducedType.isa<mlir::FunctionType>());

	auto newF = rewriter.create<mlir::rlc::FunctionOp>(
			getLoc(),
			getUnmangledName(),
			deducedType.cast<mlir::FunctionType>(),
			getArgNames(),
			checkedTemplateParameters);
	newF.getBody().takeBody(getBody());
	newF.getPrecondition().takeBody(getPrecondition());
	rewriter.eraseOp(*this);
	return mlir::success();
}

void mlir::rlc::ActionsStatement::getRegionInvocationBounds(
		llvm::ArrayRef<mlir::Attribute> operands,
		llvm::SmallVectorImpl<mlir::InvocationBounds> &invocationBounds)
{
	// every region can be executed at most one.
	invocationBounds.append(getRegions().size(), mlir::InvocationBounds(0, 1));
}

mlir::MutableOperandRange mlir::rlc::Yield::getMutableSuccessorOperands(
		mlir::RegionBranchPoint point)
{
	return getArgumentsMutable().slice(0, 0);
}

void mlir::rlc::ActionsStatement::getSuccessorRegions(
		mlir::RegionBranchPoint succ,
		llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions)
{
	// from the op you can jump to any region
	if (succ.isParent())
		for (auto *region : getRegions())
			regions.push_back(
					mlir::RegionSuccessor(region, region->front().getArguments()));

	// from any region you jump out
	if (not succ.isParent())
		regions.push_back(mlir::RegionSuccessor(getOperation()->getResults()));
}

void mlir::rlc::ActionStatement::getRegionInvocationBounds(
		llvm::ArrayRef<mlir::Attribute> operands,
		llvm::SmallVectorImpl<mlir::InvocationBounds> &invocationBounds)
{
	if (not getPrecondition().empty())
		invocationBounds.push_back(mlir::InvocationBounds(1, 1));
}

void mlir::rlc::ActionStatement::getSuccessorRegions(
		mlir::RegionBranchPoint succ,
		llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions)
{
	if (succ.isParent())
	{
		if (not getPrecondition().empty())
			regions.push_back(mlir::RegionSuccessor(
					&getPrecondition(),
					getPrecondition().front().getArguments().slice(0, 0)));
		else
			regions.push_back(mlir::RegionSuccessor(getResults().slice(0, 0)));

		return;
	}

	regions.push_back(mlir::RegionSuccessor(getResults().slice(0, 0)));
}

void mlir::rlc::DeclarationStatement::getRegionInvocationBounds(
		llvm::ArrayRef<mlir::Attribute> operands,
		llvm::SmallVectorImpl<mlir::InvocationBounds> &invocationBounds)
{
	invocationBounds.push_back(mlir::InvocationBounds(1, 1));
}

void mlir::rlc::DeclarationStatement::getSuccessorRegions(
		mlir::RegionBranchPoint succ,
		llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions)
{
	// When you hit a declaration you jump into the single body block
	if (succ.isParent())
		regions.push_back(
				mlir::RegionSuccessor(&getBody(), getBody().front().getArguments()));

	// when you are done with the region, you get out back to the statement
	if (not succ.isParent())
		regions.push_back(mlir::RegionSuccessor({}));
}

void mlir::rlc::StatementList::getRegionInvocationBounds(
		llvm::ArrayRef<mlir::Attribute> operands,
		llvm::SmallVectorImpl<mlir::InvocationBounds> &invocationBounds)
{
	invocationBounds.push_back(mlir::InvocationBounds(1, 1));
}

void mlir::rlc::StatementList::getSuccessorRegions(
		mlir::RegionBranchPoint succ,
		llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions)
{
	// When you hit a statement you jump into the single body block
	if (succ.isParent())
		regions.push_back(
				mlir::RegionSuccessor(&getBody(), getBody().front().getArguments()));

	// when you are done with the region, you get out back to the statement
	if (not succ.isParent())
		regions.push_back(mlir::RegionSuccessor({}));
}

void mlir::rlc::Yield::getRegionInvocationBounds(
		llvm::ArrayRef<mlir::Attribute> operands,
		llvm::SmallVectorImpl<mlir::InvocationBounds> &invocationBounds)
{
	if (not getOnEnd().empty())
		invocationBounds.push_back(mlir::InvocationBounds(1, 1));
}

void mlir::rlc::Yield::getSuccessorRegions(
		mlir::RegionBranchPoint succ,
		llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions)
{
	// When you hit a yield you jump into the single body block
	if (succ.isParent())
	{
		if (not getOnEnd().empty())
		{
			regions.push_back(mlir::RegionSuccessor(
					&getOnEnd(), getOnEnd().front().getArguments()));
		}
		else
		{
			regions.push_back(mlir::RegionSuccessor());
		}
	}

	// when you are done with the region, you get out back to yield
	if (not succ.isParent())
		regions.push_back(mlir::RegionSuccessor());
}

void mlir::rlc::Yield::getSuccessorRegions(
		llvm::ArrayRef<::mlir::Attribute> operands,
		llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions)
{
	llvm::cast<::mlir::RegionBranchOpInterface>((*this)->getParentOp())
			.getSuccessorRegions((*this)->getParentRegion(), regions);
}

void mlir::rlc::ReturnStatement::getRegionInvocationBounds(
		llvm::ArrayRef<mlir::Attribute> operands,
		llvm::SmallVectorImpl<mlir::InvocationBounds> &invocationBounds)
{
	invocationBounds.push_back(mlir::InvocationBounds(1, 1));
}

void mlir::rlc::ReturnStatement::getSuccessorRegions(
		mlir::RegionBranchPoint succ,
		llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions)
{
	// When you hit a return statement you jump into the single body block
	if (succ.isParent())
		regions.push_back(
				mlir::RegionSuccessor(&getBody(), getBody().front().getArguments()));

	// when you are done with the region, you don't go anywhere!
}

void mlir::rlc::ExpressionStatement::getRegionInvocationBounds(
		llvm::ArrayRef<mlir::Attribute> operands,
		llvm::SmallVectorImpl<mlir::InvocationBounds> &invocationBounds)
{
	invocationBounds.push_back(mlir::InvocationBounds(1, 1));
}

void mlir::rlc::ExpressionStatement::getSuccessorRegions(
		mlir::RegionBranchPoint succ,
		llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions)
{
	// When you hit a expression statement you jump into the single body block
	if (succ.isParent())
		regions.push_back(
				mlir::RegionSuccessor(&getBody(), getBody().front().getArguments()));

	// when you are done with the region, you get out back to ExpressionStatement
	if (not succ.isParent())
		regions.push_back(mlir::RegionSuccessor());
}

void mlir::rlc::IfStatement::getRegionInvocationBounds(
		llvm::ArrayRef<mlir::Attribute> operands,
		llvm::SmallVectorImpl<mlir::InvocationBounds> &invocationBounds)
{
	// executes the precondition once
	invocationBounds.push_back(mlir::InvocationBounds(1, 1));

	// executes the then else blocks zero times or one time
	invocationBounds.push_back(mlir::InvocationBounds(0, 1));
	invocationBounds.push_back(mlir::InvocationBounds(0, 1));
}

void mlir::rlc::IfStatement::getSuccessorRegions(
		mlir::RegionBranchPoint succ,
		llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions)
{
	// When you hit a if statement you jump into the precondition
	if (succ.isParent())
		regions.push_back(mlir::RegionSuccessor(
				&getCondition(), getCondition().front().getArguments()));

	// from the condition, you can jump to the then or else branch
	if (succ.getRegionOrNull() == &getCondition())
	{
		regions.push_back(mlir::RegionSuccessor(
				&getTrueBranch(), getTrueBranch().front().getArguments()));
		regions.push_back(mlir::RegionSuccessor(
				&getElseBranch(), getElseBranch().front().getArguments()));
	}

	// from the then and else branch you go back out
	if (succ.getRegionOrNull() == &getTrueBranch() or
			succ.getRegionOrNull() == &getElseBranch())
	{
		regions.push_back(mlir::RegionSuccessor({}));
	}
}

void mlir::rlc::SubActionStatement::getRegionInvocationBounds(
		llvm::ArrayRef<mlir::Attribute> operands,
		llvm::SmallVectorImpl<mlir::InvocationBounds> &invocationBounds)
{
	if (getRunOnce())
		invocationBounds.push_back(mlir::InvocationBounds(1, 1));
	else
		invocationBounds.push_back(mlir::InvocationBounds(0, std::nullopt));
}

void mlir::rlc::SubActionStatement::getSuccessorRegions(
		mlir::RegionBranchPoint succ,
		llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions)
{
	if (succ.isParent())
		regions.push_back(
				mlir::RegionSuccessor(&getBody(), getBody().front().getArguments()));

	if (not succ.isParent())
	{
		regions.push_back(mlir::RegionSuccessor({}));
	}
}

void mlir::rlc::WhileStatement::getRegionInvocationBounds(
		llvm::ArrayRef<mlir::Attribute> operands,
		llvm::SmallVectorImpl<mlir::InvocationBounds> &invocationBounds)
{
	// executes the precondition once or more
	invocationBounds.push_back(mlir::InvocationBounds(1, std::nullopt));

	// executes the body any amount of time
	invocationBounds.push_back(mlir::InvocationBounds(0, std::nullopt));
}

void mlir::rlc::WhileStatement::getSuccessorRegions(
		mlir::RegionBranchPoint succ,
		llvm::SmallVectorImpl<::mlir::RegionSuccessor> &regions)
{
	// When you hit a for statement you jump into the precondition
	if (succ.isParent())
		regions.push_back(mlir::RegionSuccessor(
				&getCondition(), getCondition().front().getArguments()));

	// from the condition, you can jump out or to the body
	if (succ.getRegionOrNull() == &getCondition())
	{
		regions.push_back(
				mlir::RegionSuccessor(&getBody(), getBody().front().getArguments()));
		regions.push_back(mlir::RegionSuccessor({}));
	}

	// from the body you jump to the condition
	if (succ.getRegionOrNull() == &getBody())
	{
		regions.push_back(mlir::RegionSuccessor(
				&getCondition(), getCondition().front().getArguments()));
	}
}

mlir::SuccessorOperands mlir::rlc::Branch::getSuccessorOperands(unsigned index)
{
	return mlir::SuccessorOperands(
			mlir::MutableOperandRange(getOperation(), 0, 0));
}

mlir::SuccessorOperands mlir::rlc::CondBranch::getSuccessorOperands(
		unsigned index)
{
	return mlir::SuccessorOperands(
			mlir::MutableOperandRange(getOperation(), 0, 0));
}

mlir::SuccessorOperands mlir::rlc::FlatActionStatement::getSuccessorOperands(
		unsigned index)
{
	return mlir::SuccessorOperands(
			mlir::MutableOperandRange(getOperation(), 0, 0));
}

mlir::SuccessorOperands mlir::rlc::SelectBranch::getSuccessorOperands(
		unsigned index)
{
	return mlir::SuccessorOperands(
			mlir::MutableOperandRange(getOperation(), 0, 0));
}

mlir::rlc::TemplateParameterType
mlir::rlc::UncheckedTraitDefinition::getTemplateParameterType()
{
	return mlir::rlc::TemplateParameterType::get(
			getContext(), getTemplateParameter(), nullptr, false);
}
