
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/Pass/Pass.h"
#include "rlc/dialect/Operations.hpp"
#include "rlc/dialect/conversion/TypeConverter.h"

static mlir::LogicalResult findEntityDecls(
		mlir::ModuleOp op, llvm::StringMap<mlir::rlc::EntityDeclaration>& out)
{
	for (auto entityDecl : op.getOps<mlir::rlc::EntityDeclaration>())
	{
		if (auto prevDef = out.find(entityDecl.getName()); prevDef != out.end())
		{
			auto _ = mlir::rlc::logError(
					entityDecl, "Multiple definition of entity " + entityDecl.getName());

			return mlir::rlc::logRemark(prevDef->second, "Previous definition");
		}

		out[entityDecl.getName()] = entityDecl;
	}

	return mlir::success();
}

static void collectEntityUsedTyepNames(
		mlir::Type type, llvm::SmallVector<llvm::StringRef, 2>& out)
{
	if (auto casted = type.dyn_cast<mlir::rlc::ScalarUseType>())
	{
		for (auto templ : casted.getExplicitTemplateParameters())
			collectEntityUsedTyepNames(templ, out);

		if (casted.getUnderlying() != nullptr)
		{
			collectEntityUsedTyepNames(casted.getUnderlying(), out);
			return;
		}
		out.push_back(casted.getReadType());
		return;
	}
	if (auto casted = type.dyn_cast<mlir::rlc::FunctionUseType>())
	{
		for (auto arg : casted.getSubTypes())
			collectEntityUsedTyepNames(arg, out);
		return;
	}
	if (auto casted = type.dyn_cast<mlir::rlc::OwningPtrType>())
	{
		collectEntityUsedTyepNames(casted.getUnderlying(), out);
		return;
	}
	if (auto casted = type.dyn_cast<mlir::rlc::AlternativeType>())
	{
		for (auto subType : casted.getUnderlying())
			collectEntityUsedTyepNames(subType, out);
		return;
	}
	llvm_unreachable("unrechable");
}

static mlir::LogicalResult getEntityDeclarationSortedByDependencies(
		mlir::ModuleOp op, llvm::SmallVector<mlir::rlc::EntityDeclaration, 2>& out)
{
	llvm::StringMap<mlir::rlc::EntityDeclaration> nameToEntityDeclaration;
	if (findEntityDecls(op, nameToEntityDeclaration).failed())
		return mlir::failure();

	std::map<
			mlir::rlc::EntityDeclaration,
			llvm::SmallVector<mlir::rlc::EntityDeclaration, 2>>
			EntityToUsedEntities;

	for (auto entityDecl : op.getOps<mlir::rlc::EntityDeclaration>())
	{
		llvm::SmallVector<mlir::StringRef, 2> names;
		for (auto subtypes : entityDecl.getMemberTypes())
			collectEntityUsedTyepNames(
					subtypes.cast<mlir::TypeAttr>().getValue(), names);

		EntityToUsedEntities[entityDecl];
		for (auto name : names)
		{
			if (auto iter = nameToEntityDeclaration.find(name);
					iter != nameToEntityDeclaration.end())
			{
				EntityToUsedEntities[entityDecl].push_back(iter->second);
			}
		}
	}

	while (not EntityToUsedEntities.empty())
	{
		llvm::SmallVector<mlir::rlc::EntityDeclaration> justEmitted;

		for (auto& entry : EntityToUsedEntities)
		{
			llvm::erase_if(entry.second, [&](mlir::rlc::EntityDeclaration decl) {
				return not EntityToUsedEntities.contains(decl);
			});
			if (entry.second.empty())
			{
				justEmitted.push_back(entry.first);
				auto e = entry.first;
				out.push_back(entry.first);
			}
		}

		for (auto& entry : justEmitted)
			EntityToUsedEntities.erase(entry);

		if (justEmitted.empty() and not EntityToUsedEntities.empty())
		{
			return mlir::rlc::logError(
					*EntityToUsedEntities.begin()->second.begin(),
					"Forbidden mutual dependency in entities");
		}
	}

	return mlir::success();
}

static mlir::LogicalResult declareEntities(mlir::ModuleOp op)
{
	mlir::IRRewriter rewriter(op.getContext());
	llvm::SmallVector<mlir::rlc::EntityDeclaration, 2> decls;
	if (getEntityDeclarationSortedByDependencies(op, decls).failed())
		return mlir::failure();

	for (auto& decl : decls)
	{
		llvm::SmallVector<mlir::Type, 2> templates;
		for (auto type :
				 decl.getTemplateParameters().getAsValueRange<mlir::TypeAttr>())
			templates.push_back(type);
		rewriter.setInsertionPoint(decl);
		auto newDecl = rewriter.create<mlir::rlc::EntityDeclaration>(
				decl.getLoc(),
				mlir::rlc::EntityType::getIdentified(
						decl.getContext(), decl.getName(), templates),
				decl.getName(),
				decl.getMemberTypes(),
				decl.getMemberNames(),
				decl.getTemplateParameters());
		rewriter.eraseOp(decl);
	}
	return mlir::success();
}

static mlir::LogicalResult declareActionEntities(mlir::ModuleOp op)
{
	mlir::IRRewriter rewriter(op.getContext());
	llvm::StringMap<mlir::rlc::EntityDeclaration> decls;
	if (findEntityDecls(op, decls).failed())
		return mlir::failure();

	rewriter.setInsertionPointToEnd(op.getBody());
	for (auto action : op.getOps<mlir::rlc::ActionFunction>())
	{
		auto type = action.getEntityType();

		auto entity = rewriter.create<mlir::rlc::EntityDeclaration>(
				action.getLoc(),
				type,
				rewriter.getStringAttr(type.getName()),
				rewriter.getTypeArrayAttr({}),
				rewriter.getStrArrayAttr({}),
				rewriter.getArrayAttr({}));

		if (decls.count(type.getName()) != 0)
		{
			auto _ = mlir::rlc::logError(
					decls[type.getName()],
					"User defined type clashes with implicit action type" +
							type.getName());
			return mlir::rlc::logRemark(action, "Action defined here");
		}

		decls.try_emplace(type.getName(), entity);
	}

	return mlir::success();
}

static mlir::LogicalResult deduceEntitiesBodies(mlir::ModuleOp op)
{
	llvm::SmallVector<mlir::rlc::EntityDeclaration, 2> decls;
	if (getEntityDeclarationSortedByDependencies(op, decls).failed())
		return mlir::failure();

	for (auto& decl : decls)
	{
		mlir::rlc::ModuleBuilder builder(op);
		mlir::IRRewriter& rewriter = builder.getRewriter();
		rewriter.setInsertionPoint(decl);

		// entities of actions are discovered later, because they require to
		// typecheck the body of the action itself.
		if (builder.isEntityOfAction(decl.getType()))
			continue;

		llvm::SmallVector<std::string> names;
		llvm::SmallVector<mlir::Type, 2> types;

		llvm::SmallVector<mlir::Type, 2> checkedTemplateParameters;

		auto scopedConverter = mlir::rlc::RLCTypeConverter(&builder.getConverter());
		for (auto parameter : decl.getTemplateParameters())
		{
			auto unchecked = parameter.cast<mlir::TypeAttr>()
													 .getValue()
													 .cast<mlir::rlc::UncheckedTemplateParameterType>();

			auto checkedParameterType = scopedConverter.convertType(unchecked);
			if (not checkedParameterType)
			{
				auto _ = mlir::rlc::logError(
						decl, "No known type named " + mlir::rlc::prettyType(unchecked));
				return mlir::rlc::logRemark(decl, "In entity declaration");
			}
			checkedTemplateParameters.push_back(checkedParameterType);
			auto actualType =
					checkedParameterType.cast<mlir::rlc::TemplateParameterType>();
			scopedConverter.registerType(actualType.getName(), actualType);
		}

		for (const auto& [field, name] :
				 llvm::zip(decl.getMemberTypes(), decl.getMemberNames()))
		{
			auto fieldType = field.cast<mlir::TypeAttr>().getValue();
			auto converted = scopedConverter.convertType(fieldType);
			if (!converted)
			{
				return mlir::rlc::logError(
						decl,
						"No known type named " + mlir::rlc::prettyType(fieldType) +
								" used in field " + name.cast<mlir::StringAttr>().strref() +
								" in entity declaration.");
			}

			types.push_back(converted);
			names.push_back(name.cast<mlir::StringAttr>().str());
		}
		auto finalType = mlir::rlc::EntityType::getIdentified(
				decl.getContext(), decl.getName(), checkedTemplateParameters);

		if (finalType.setBody(types, names).failed())
		{
			assert(false && "unrechable");
			return mlir::failure();
		}

		decl = rewriter.replaceOpWithNewOp<mlir::rlc::EntityDeclaration>(
				decl,
				finalType,
				decl.getName(),
				rewriter.getTypeArrayAttr(types),
				decl.getMemberNames(),
				rewriter.getTypeArrayAttr(checkedTemplateParameters));
	}

	return mlir::success();
}

static mlir::LogicalResult deduceFunctionTypes(mlir::ModuleOp op)
{
	mlir::rlc::RLCTypeConverter converter(op);
	mlir::IRRewriter rewriter(op.getContext());

	llvm::SmallVector<mlir::rlc::FunctionOp, 4> funs;
	for (auto fun : op.getOps<mlir::rlc::FunctionOp>())
		funs.push_back(fun);

	for (auto fun : funs)
	{
		if (fun.typeCheckFunctionDeclaration(rewriter, converter).failed())
			return mlir::failure();
	}

	return mlir::success();
}

static mlir::LogicalResult deduceOperationTypes(mlir::ModuleOp op)
{
	mlir::rlc::ModuleBuilder builder(op);

	mlir::IRRewriter rewriter(op.getContext());

	llvm::SmallVector<mlir::rlc::FunctionOp, 4> funs(
			op.getOps<mlir::rlc::FunctionOp>());
	for (auto fun : funs)
	{
		auto _ = builder.addSymbolTable();
		for (auto templateParameter : fun.getTemplateParameters())
		{
			auto casted = templateParameter.cast<mlir::TypeAttr>()
												.getValue()
												.cast<mlir::rlc::TemplateParameterType>();
			builder.getConverter().registerType(casted.getName(), casted);
			if (casted.getIsIntLiteral() == true)
			{
				builder.getRewriter().setInsertionPoint(&fun.getBody().front().front());
				auto integerLiteral =
						builder.getRewriter().create<mlir::rlc::IntegerLiteralUse>(
								op.getLoc(),
								mlir::rlc::IntegerType::getInt64(op.getContext()),
								casted);
				builder.getSymbolTable().add(casted.getName(), integerLiteral);
			}
			else
			{
				if (auto trait = casted.getTrait())
					builder.addTraitToAviableOverloads(trait);
			}
		}

		if (mlir::rlc::typeCheck(*fun.getOperation(), builder).failed())
			return mlir::failure();
	}
	return mlir::success();
}

static mlir::LogicalResult deduceTraitTypes(mlir::ModuleOp op)
{
	mlir::rlc::ModuleBuilder builder(op);
	mlir::IRRewriter rewriter(op.getContext());

	llvm::SmallVector<mlir::rlc::UncheckedTraitDefinition, 4> funs(
			op.getOps<mlir::rlc::UncheckedTraitDefinition>());
	for (auto fun : funs)
	{
		if (mlir::rlc::typeCheck(*fun.getOperation(), builder).failed())
			return mlir::failure();
	}
	return mlir::success();
}

static mlir::LogicalResult deduceActionsMainFunctionType(mlir::ModuleOp op)
{
	mlir::IRRewriter rewriter(op.getContext());

	mlir::rlc::ModuleBuilder builder(op);
	llvm::SmallVector<mlir::rlc::ActionFunction, 4> funs(
			op.getOps<mlir::rlc::ActionFunction>());
	for (auto fun : funs)
	{
		auto funType = builder.typeOfAction(fun);

		mlir::Type actionType =
				builder.getConverter().convertType(mlir::FunctionType::get(
						op.getContext(), fun.getFunctionType().getInputs(), { funType }));

		rewriter.setInsertionPoint(fun);
		auto newAction = rewriter.create<mlir::rlc::ActionFunction>(
				fun.getLoc(),
				actionType,
				mlir::FunctionType::get(
						rewriter.getContext(),
						mlir::TypeRange({ funType }),
						mlir::TypeRange(
								{ mlir::rlc::BoolType::get(rewriter.getContext()) })),
				mlir::TypeRange(),
				fun.getUnmangledName(),
				fun.getArgNames());
		newAction.getBody().takeBody(fun.getBody());
		newAction.getPrecondition().takeBody(fun.getPrecondition());
		rewriter.eraseOp(fun);
	}
	return mlir::success();
}

static mlir::LogicalResult typeCheckActions(mlir::ModuleOp op)
{
	mlir::IRRewriter rewriter(op.getContext());

	bool foundOne = true;
	// the actions are allowed to type check each other, but that means that they
	// will change in the process, so if one typechecks and change, we have to
	// start again typechecking them all. if one typechecks and does not change,
	// it can't possible have changed any other either.
	while (foundOne)
	{
		foundOne = false;
		llvm::SmallVector<mlir::rlc::ActionFunction, 4> funs(
				op.getOps<mlir::rlc::ActionFunction>());
		for (auto fun : funs)
		{
			auto typeChecked = mlir::rlc::detail::typeCheckAction(fun);
			if (typeChecked == nullptr)
				return mlir::failure();
			if (typeChecked != fun)
			{
				foundOne = true;
				break;
			}
		}
	}
	return mlir::success();
}

namespace mlir::rlc
{
#define GEN_PASS_DEF_TYPECHECKENTITIESPASS
#include "rlc/dialect/Passes.inc"

	struct TypeCheckEntitiesPass
			: impl::TypeCheckEntitiesPassBase<TypeCheckEntitiesPass>
	{
		using impl::TypeCheckEntitiesPassBase<
				TypeCheckEntitiesPass>::TypeCheckEntitiesPassBase;
		void runOnOperation() override
		{
			if (declareEntities(getOperation()).failed())
			{
				signalPassFailure();
				return;
			}

			if (declareActionEntities(getOperation()).failed())
			{
				signalPassFailure();
				return;
			}

			if (deduceEntitiesBodies(getOperation()).failed())
			{
				signalPassFailure();
				return;
			}

			if (deduceTraitTypes(getOperation()).failed())
			{
				signalPassFailure();
				return;
			}
		}
	};

#define GEN_PASS_DEF_TYPECHECKPASS
#include "rlc/dialect/Passes.inc"

	struct TypeCheckPass: impl::TypeCheckPassBase<TypeCheckPass>
	{
		using impl::TypeCheckPassBase<TypeCheckPass>::TypeCheckPassBase;
		void runOnOperation() override
		{
			if (deduceFunctionTypes(getOperation()).failed())
			{
				signalPassFailure();
				return;
			}

			if (deduceActionsMainFunctionType(getOperation()).failed())
			{
				signalPassFailure();
				return;
			}

			if (typeCheckActions(getOperation()).failed())
			{
				signalPassFailure();
				return;
			}

			if (deduceOperationTypes(getOperation()).failed())
			{
				signalPassFailure();
				return;
			}
		}
	};
}	 // namespace mlir::rlc
