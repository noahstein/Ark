template<typename T>
concept Evaluateable = requires(T a)
{
	typename T::ReturnT;
	{ a.eval() };
};

template<Evaluateable L, Evaluateable R>
struct MultiplyExpr
{
	constexpr L& l_;
	constexpr R& r_;

	constexpr MultiplyExpr(constexpr L& l, constexpr R& r)
		: l_(l), r_(r)
	{}

	using ReturnT = decltype(l_ * r_);

	constexpr auto eval()
	{
		return l_ * r_;
	}
};
