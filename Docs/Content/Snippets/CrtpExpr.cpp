template<typename E>
struct Expr
{
	using ExprT =  E;

	constexpr ExprT expression_;

	Expr(constexpr ExptT& expression) constexpr
		: expression_(expression)
	{}

	auto constexpr eval() -> decltype(expression_.eval())
	{
		 return expression_.eval();
	}
};

template<typename T>
struct TerminalExpr : Expr<TerminalExpr<T>>
{
	using ReturnT = T;

	constexpr T& terminal_;

	constexpr TerminalExpr(constexpr T& terminal)
		: terminal_(terminal)
	{}

	auto constexpr eval() -> T
	{
		return terminal_;
	}
};

template<typename L, typename R>
struct MultiplyExpr : Expr<MultiplyExpr<L, R>>
{
	constexpr L& l_;
	constexpr R& r_;

	constexpr MultiplyExpr(constexpr L& l, constexpr R& r) 
		: l_(l), r_(r)
	{}

	auto constexpr eval()  -> decltype(l_.eval() * r_.eval())
	{
		return l_.eval() * r_.eval();
	}
};
