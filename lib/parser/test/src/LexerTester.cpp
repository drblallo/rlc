#include "gtest/gtest.h"

#include "rlc/parser/Lexer.hpp"
#include "rlc/utils/ScopeGuard.hpp"

using namespace rlc;

TEST(LexerTest, lexerTestIf)
{
	Lexer lexer("if");
	EXPECT_EQ(lexer.next(), Token::KeywordIf);
	EXPECT_EQ(lexer.next(), Token::End);
}

TEST(LexerTest, lexerLongTest)
{
	Lexer lexer("if 34 in ()");
	EXPECT_EQ(lexer.next(), Token::KeywordIf);
	EXPECT_EQ(lexer.next(), Token::Int64);
	EXPECT_EQ(lexer.lastInt64(), 34);
	EXPECT_EQ(lexer.next(), Token::KeywordIn);
	EXPECT_EQ(lexer.next(), Token::LPar);
	EXPECT_EQ(lexer.next(), Token::RPar);
	EXPECT_EQ(lexer.next(), Token::End);
}

TEST(LexerTest, lexerDoubleTest)
{
	Lexer lexer("if 34.0 in ()");
	EXPECT_EQ(lexer.next(), Token::KeywordIf);
	EXPECT_EQ(lexer.next(), Token::Double);
	EXPECT_EQ(lexer.lastDouble(), 34.0);
	EXPECT_EQ(lexer.next(), Token::KeywordIn);
	EXPECT_EQ(lexer.next(), Token::LPar);
	EXPECT_EQ(lexer.next(), Token::RPar);
	EXPECT_EQ(lexer.next(), Token::End);
}

TEST(LexerTest, lexerTestIfElse)
{
	Lexer lexer("if else");
	EXPECT_EQ(lexer.next(), Token::KeywordIf);
	EXPECT_EQ(lexer.next(), Token::KeywordElse);
	EXPECT_EQ(lexer.next(), Token::End);
}

TEST(LexerTest, operators)
{
	Lexer lexer("if + else/3");
	EXPECT_EQ(lexer.next(), Token::KeywordIf);
	EXPECT_EQ(lexer.next(), Token::Plus);
	EXPECT_EQ(lexer.next(), Token::KeywordElse);
	EXPECT_EQ(lexer.next(), Token::Divide);
	EXPECT_EQ(lexer.next(), Token::Int64);
	EXPECT_EQ(lexer.next(), Token::End);
}

TEST(LexerTest, lexerIndentTest)
{
	Lexer lexer("ent \n\t\telse\n\t\tin\t  \nfor");
	EXPECT_EQ(lexer.next(), Token::KeywordEntity);
	EXPECT_EQ(lexer.next(), Token::Newline);
	EXPECT_EQ(lexer.next(), Token::Indent);
	EXPECT_EQ(lexer.next(), Token::KeywordElse);
	EXPECT_EQ(lexer.next(), Token::Newline);
	EXPECT_EQ(lexer.next(), Token::KeywordIn);
	EXPECT_EQ(lexer.next(), Token::Newline);
	EXPECT_EQ(lexer.next(), Token::Deindent);
	EXPECT_EQ(lexer.next(), Token::KeywordFor);
	EXPECT_EQ(lexer.next(), Token::End);
}

TEST(LexerTest, nestedIndentTest)
{
	Lexer lexer("ent \n\t\telse\n\t\t\tin\t  \n\t\tfor\n[");
	EXPECT_EQ(lexer.next(), Token::KeywordEntity);
	EXPECT_EQ(lexer.next(), Token::Newline);
	EXPECT_EQ(lexer.next(), Token::Indent);
	EXPECT_EQ(lexer.next(), Token::KeywordElse);
	EXPECT_EQ(lexer.next(), Token::Newline);
	EXPECT_EQ(lexer.next(), Token::Indent);
	EXPECT_EQ(lexer.next(), Token::KeywordIn);
	EXPECT_EQ(lexer.next(), Token::Newline);
	EXPECT_EQ(lexer.next(), Token::Deindent);
	EXPECT_EQ(lexer.next(), Token::KeywordFor);
	EXPECT_EQ(lexer.next(), Token::Newline);
	EXPECT_EQ(lexer.next(), Token::Deindent);
	EXPECT_EQ(lexer.next(), Token::LSquare);
	EXPECT_EQ(lexer.next(), Token::End);
}

TEST(LexerTest, twoCharSymbolsTest)
{
	Lexer lexer("==,!=,<=,>=,=,<");
	EXPECT_EQ(lexer.next(), Token::EqualEqual);
	EXPECT_EQ(lexer.next(), Token::Comma);
	EXPECT_EQ(lexer.next(), Token::NEqual);
	EXPECT_EQ(lexer.next(), Token::Comma);
	EXPECT_EQ(lexer.next(), Token::LEqual);
	EXPECT_EQ(lexer.next(), Token::Comma);
	EXPECT_EQ(lexer.next(), Token::GEqual);
	EXPECT_EQ(lexer.next(), Token::Comma);
	EXPECT_EQ(lexer.next(), Token::Equal);
	EXPECT_EQ(lexer.next(), Token::Comma);
	EXPECT_EQ(lexer.next(), Token::LAng);
}
