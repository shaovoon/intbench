// BenchmarkIntConv.cpp : Defines the entry point for the console application.
//

// BenchmarkFloatConv.cpp : Defines the entry point for the console application.
//

#include <vector>
#include <iostream>
#include <iomanip>
#include <string>
#include <cstring>
#include <cmath>
#include <cassert>
#include <sstream>
#include <cstdlib>
#include <chrono>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <boost/spirit/include/qi.hpp>
#include <boost/lexical_cast.hpp>
#include <cstdint>
#include <charconv>

typedef std::pair<const std::string, const std::int64_t> pair_type;
typedef std::vector< pair_type > vector_type;

#ifdef WIN32

#pragma optimize("", off)
template <class T>
void do_not_optimize_away(T&& datum) {
	datum = datum;
}
#pragma optimize("", on)

#else
static void do_not_optimize_away(void* p) { 
	asm volatile("" : : "g"(p) : "memory");
}
#endif

void init(vector_type& vec);


#define MYASSERT(value, expected) 

//#define MYASSERT(value, expected) if(value != expected) { std::cerr << value << " and expected:" << expected << " are different" << std::endl; }

class timer
{
public:
	timer() = default;
	void start(const std::string& text_)
	{
		text = text_;
		begin = std::chrono::high_resolution_clock::now();
	}
	void stop()
	{
		auto end = std::chrono::high_resolution_clock::now();
		auto dur = end - begin;
		auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();
		std::cout << std::setw(19) << text << ":" << std::setw(5) << ms << "ms" << std::endl;
	}

private:
	std::string text;
	std::chrono::high_resolution_clock::time_point begin;
};

// bit lookup table of valid ascii code for decimal string conversion, white space, sign, numeric digits
static  char BtMLValDecInt[32] = { 0x0, 0x3e, 0x0, 0x0, 0x1, 0x28, 0xff, 0x03,
0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };

// bit lookup table, white space only
static char BtMLws[32] = { 0x0, 0x3e, 0x0, 0x0, 0x1, 0x0, 0x0, 0x0,
0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0,
0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
// list of white space for sttni use
static  char listws[16] =
{ 0x20, 0x9, 0xa, 0xb, 0xc, 0xd, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
// list of numeric digits for sttni use
static  char rangenumint[16] =
{ 0x30, 0x39, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
static  char rangenumintzr[16] =
{ 0x30, 0x30, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0, 0x0 };
// we use pmaddwd to merge two adjacent short integer pair, this is the second step of merging each pair of 2 - digit integers
static short MulplyPairBaseP2[8] =
{ 100, 1, 100, 1, 100, 1, 100, 1 };
// Multiplier-pair for two adjacent short integer pair, this is the third step of merging each pair of 4-digit integers
static short MulplyPairBaseP4[8] =
{ 10000, 1, 10000, 1, 10000, 1, 10000, 1 };
// multiplier for pmulld for normalization of > 16 digits
static int MulplyByBaseExpN[8] =
{ 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000 };
static short MulplyQuadBaseExp3To0[8] =
{ 1000, 100, 10, 1,
1000, 100, 10, 1 };

__m128i __m128i_shift_right(__m128i value, int offset);
__m128i __m128i_strloadu_page_boundary(const  char *s);
__m128i ShfLAlnLSByte(__m128i value, int offset);

std::int64_t sse4i_atol(const  char* s1);

#define RINT64VALNEG -1
#define RINT64VALPOS 1

std::int64_t simple_atol(const char* start)
{
	std::int64_t ret = 0;
	size_t my_size = strlen(start);

	bool neg = (*start == '-');
	if (neg) {
		start++;
		my_size--;
	}

	while (*start >= '0' && *start <= '9' && my_size) {
		ret = ret * 10 + *start - '0';
		start++;
		my_size--;
	}
	return (!neg) ? ret : -ret;
}

int main(int argc, char *argv [])
{
	const size_t MAX_LOOP = (argc == 2) ? atoi(argv[1]) : 1000000;

	vector_type vec;
	init(vec);
	timer stopwatch;
	std::int64_t n = 0;

	stopwatch.start("atol");
	for (size_t k = 0; k<MAX_LOOP; ++k)
	{
		for(size_t i=0; i<vec.size(); ++i)
		{
			pair_type& pr = vec[i];
			n = std::atol(pr.first.c_str());
			do_not_optimize_away(&n);
			MYASSERT(n, pr.second);
		}
	}
	stopwatch.stop();

	stopwatch.start("lexical_cast");
	for (size_t k = 0; k < MAX_LOOP; ++k)
	{
		for(size_t i=0; i<vec.size(); ++i)
		{
			pair_type& pr = vec[i];
			n = boost::lexical_cast<std::int64_t>(pr.first.c_str());
			do_not_optimize_away(&n);
			MYASSERT(n, pr.second);
		}
	}
	stopwatch.stop();

	stopwatch.start("std::istringstream");
	for (size_t k = 0; k < MAX_LOOP; ++k)
	{
		for (size_t i = 0; i<vec.size(); ++i)
		{
			pair_type& pr = vec[i];
			std::istringstream oss(pr.first);
			oss >> n;
			do_not_optimize_away(&n);
			MYASSERT(n, pr.second);
		}
	}
	stopwatch.stop();

	stopwatch.start("std::stoll");
	for (size_t k = 0; k < MAX_LOOP; ++k)
	{
		for (size_t i = 0; i<vec.size(); ++i)
		{
			pair_type& pr = vec[i];
			n = std::stoll(pr.first, nullptr);
			do_not_optimize_away(&n);
			MYASSERT(n, pr.second);
		}
	}
	stopwatch.stop();

	stopwatch.start("simple_atol");
	for (size_t k = 0; k < MAX_LOOP; ++k)
	{
		for(size_t i=0; i<vec.size(); ++i)
		{
			pair_type& pr = vec[i];
			n = simple_atol(pr.first.c_str());
			do_not_optimize_away(&n);
			MYASSERT(n, pr.second);
		}
	}
	stopwatch.stop();

	stopwatch.start("sse4i_atol");
	for (size_t k = 0; k < MAX_LOOP; ++k)
	{
		for (size_t i = 0; i<vec.size(); ++i)
		{
			pair_type& pr = vec[i];
			n = sse4i_atol(pr.first.c_str());
			do_not_optimize_away(&n);
			MYASSERT(n, pr.second);
		}
	}
	stopwatch.stop();

	stopwatch.start("boost_spirit");
	namespace qi = boost::spirit::qi;
	for (size_t k = 0; k < MAX_LOOP; ++k)
	{
		for (size_t i = 0; i<vec.size(); ++i)
		{
			pair_type& pr = vec[i];
			bool success = qi::parse(pr.first.cbegin(), pr.first.cend(), qi::long_long, n);
			do_not_optimize_away(&n);
			MYASSERT(n, pr.second);
		}
	}
	stopwatch.stop();

	stopwatch.start("std::from_chars");
	for (size_t k = 0; k < MAX_LOOP; ++k)
	{
		for (size_t i = 0; i < vec.size(); ++i)
		{
			pair_type& pr = vec[i];
			std::from_chars(pr.first.data(), pr.first.data()+pr.first.size(), n);
			do_not_optimize_away(&n);
			MYASSERT(n, pr.second);
		}
	}
	stopwatch.stop();

	std::cout << "\nLast int value: " << n << " <-- ignore this value" << std::endl;
	return 0;
}

void init(vector_type& vec)
{
	std::string int_str = "12369";
	vec.push_back(std::make_pair(int_str, atol(int_str.c_str())));
	int_str = "25934";
	vec.push_back(std::make_pair(int_str, atol(int_str.c_str())));
	int_str = "4789636";
	vec.push_back(std::make_pair(int_str, atol(int_str.c_str())));
	int_str = "532102";
	vec.push_back(std::make_pair(int_str, atol(int_str.c_str())));
	int_str = "45655";
	vec.push_back(std::make_pair(int_str, atol(int_str.c_str())));
	int_str = "83658";
	vec.push_back(std::make_pair(int_str, atol(int_str.c_str())));
	int_str = "1256900";
	vec.push_back(std::make_pair(int_str, atol(int_str.c_str())));
	int_str = "12362311";
	vec.push_back(std::make_pair(int_str, atol(int_str.c_str())));
	int_str = "55222389";
	vec.push_back(std::make_pair(int_str, atol(int_str.c_str())));
	int_str = "1423";
	vec.push_back(std::make_pair(int_str, atol(int_str.c_str())));

}

std::int64_t sse4i_atol(const  char* s1)
{
	char  *p = (char *) s1;
	int NegSgn = 0;
	__m128i mask0;
	__m128i  value0, value1;
	__m128i  w1, w1_l8, w1_u8, w2, w3 = _mm_setzero_si128();
	std::int64_t xxi;
	int index, cflag, sflag, zflag, oob = 0;
	// check the first character is valid via lookup
	if ((BtMLValDecInt[*p >> 3] & (1
		<< ((*p) & 7))) == 0) return 0;
	// if the first character is white space, skip remaining white spaces
	if (BtMLws[*p >> 3] & (1 << ((*p) & 7)))
	{
		p++;
		value0 = _mm_loadu_si128((__m128i *) listws);
	skip_more_ws:
		mask0 = __m128i_strloadu_page_boundary(p);
		/* look for the 1s
		t non-white space character */
		index = _mm_cmpistri(value0, mask0, 0x10);
		cflag = _mm_cmpistrc
			(value0, mask0, 0x10);
		sflag = _mm_cmpistrs
			(value0, mask0, 0x10);
		if (!sflag && !cflag)
		{
			p = (char *) ((size_t) p + 16);
			goto skip_more_ws;
		}
		else         p = (char *) ((size_t) p + index);
	}
	if (*p == '-')
	{
		p++;
		NegSgn = 1;
	}
	else if (*p == '+') p++;
	/* load up to 16 byte safely and check how
	many valid numeric digits we can do SIMD */
	value0 = _mm_loadu_si128((__m128i *) rangenumint);
	mask0 = __m128i_strloadu_page_boundary(p);
	index = _mm_cmpistri(value0, mask0, 0x14);
	zflag = _mm_cmpistrz(value0, mask0, 0x14);
	/* index points to the first digit that is not a valid numeric digit */
	if (!index) return 0;
	else if (index == 16)
	{
		if (*p == '0') /* if all
					   16 bytes are numeric digits */
		{  /* skip leading zero */
			value1 = _mm_loadu_si128((__m128i *) rangenumintzr);
			index = _mm_cmpistri(value1, mask0, 0x14);
			zflag = _mm_cmpistrz(value1, mask0, 0x14);
			while (index == 16 && !zflag)
			{
				p = (char *) ((size_t) p + 16);
				mask0 = __m128i_strloadu_page_boundary(p);
				index = _mm_cmpistri(value1, mask0, 0x14);
				zflag = _mm_cmpistrz(value1, mask0, 0x14);
			}
			/* now the 1st digit is non-zero, load up to 16 bytes and update index  */
			if (index < 16)
				p = (char *) ((size_t) p + index);
			/* load up to 16 bytes of non-zero leading numeric digits */
			mask0 = __m128i_strloadu_page_boundary(p);
			/* update index to point to non-numeric character or indicate we may have more than 16 bytes */
			index = _mm_cmpistri(value0, mask0, 0x14);
		}
	}
	if (index == 0)  return 0;
	else if (index == 1)   return (NegSgn ? (long long) -(p[0] - 48) : (long long) (p[0] - 48));
	// Input digits in xmm are ordered in reverse order.the LS digit of output is next to eos
	// least sig numeric digit aligned to byte 15 , and subtract 0x30 from each ascii code 
	mask0 = ShfLAlnLSByte(mask0, 16 - index);
	w1_u8 = _mm_slli_si128(mask0, 1);
	w1 = _mm_add_epi8(mask0, _mm_slli_epi16(w1_u8, 3)); /* mul by 8  and add */
	w1 = _mm_add_epi8(w1, _mm_slli_epi16(w1_u8, 1)); /* 7 LS bits per byte, in bytes 0, 2, 4, 6, 8, 10, 12, 14*/
	w1 = _mm_srli_epi16(w1, 8);  /* clear out upper bits of each wd*/
	w2 = _mm_madd_epi16(w1, _mm_loadu_si128((__m128i *) &MulplyPairBaseP2[0])); /* multiply base^2, add adjacent word,*/
	w1_u8 = _mm_packus_epi32(w2, w2);  /* pack 4 low word of each dword into 63:0 */
	w1 = _mm_madd_epi16(w1_u8, _mm_loadu_si128((__m128i*) &MulplyPairBaseP4[0])); /* multiply base^4, add  adjacent word,*/
	w1 = _mm_cvtepu32_epi64(w1);  /*   converted dw was in 63:0, expand to qw */
	w1_l8 = _mm_mul_epu32(w1, _mm_setr_epi32(100000000, 0, 0, 0));
	w2 = _mm_add_epi64(w1_l8, _mm_srli_si128(w1, 8));
	if (index < 16)
	{
		xxi = _mm_extract_epi64(w2, 0);
		return (NegSgn ? (long
			long) -xxi : (long long) xxi);
	}
	/* 64-bit integer allow up to 20 non-zero-leading digits. */
	/* accumulate each 16-
	digit fragment*/
	w3 = _mm_add_epi64(w3, w2);
	/* handle next batch of up to 16 digits,
	64-bit integer only allow 4 more digits */
	p = (char *) ((size_t) p + 16);
	if (*p == 0)
	{
		xxi = _mm_extract_epi64(w2, 0);
		return (NegSgn ? (long
			long) -xxi : (long long) xxi);
	}
	mask0 = __m128i_strloadu_page_boundary(p);
	/* index points to first non-numeric digit */
	index = _mm_cmpistri(value0, mask0, 0x14);
	zflag = _mm_cmpistrz(value0, mask0, 0x14);
	if (index == 0) /* the first char is not valid numeric digit */
	{
		xxi = _mm_extract_epi64(w2, 0);
		return (NegSgn ? (long
			long) -xxi : (long long) xxi);
	}
	if (index > 3)  return (NegSgn ? (long long) RINT64VALNEG : (long long) RINT64VALPOS);
	/* multiply low qword by base^index */
	w1 = _mm_mul_epu32(_mm_shuffle_epi32(w2, 0x50), _mm_setr_epi32(MulplyByBaseExpN[index - 1], 0, MulplyByBaseExpN[index - 1], 0));
	w3 = _mm_add_epi64(w1, _mm_slli_epi64(_mm_srli_si128(w1, 8), 32));
	mask0 = ShfLAlnLSByte(mask0, 16 - index);
	// convert upper 8 bytes of xmm: only least sig. 4 digits of output will be added to prev 16 digits
	w1_u8 = _mm_cvtepi8_epi16(_mm_srli_si128(mask0, 8));
	/* merge 2 digit  at
	a time with multiplier
	into each dword*/
	w1_u8 = _mm_madd_epi16(w1_u8, _mm_loadu_si128((__m128i *) &MulplyQuadBaseExp3To0[0]));
	/* bits 63:0 has two dword integer, bits 63:32 is the LS dword of output; bits 127:64 is not needed*/
	w1_u8 = _mm_cvtepu32_epi64(_mm_hadd_epi32(w1_u8, w1_u8));
	w3 = _mm_add_epi64(w3, _mm_srli_si128(w1_u8, 8));
	xxi = _mm_extract_epi64(w3, 0);
	if (xxi >> 63)
		return (NegSgn ? (long long) RINT64VALNEG : (long long) RINT64VALPOS);
	else   return (NegSgn ? (long long) -xxi : (long long) xxi);
}

__m128i __m128i_shift_right(__m128i value, int offset)
{
	switch (offset)
	{
	case 1: value = _mm_srli_si128(value, 1); break;
	case 2: value = _mm_srli_si128(value, 2); break;
	case 3: value = _mm_srli_si128(value, 3); break;
	case 4: value = _mm_srli_si128(value, 4); break;
	case 5: value = _mm_srli_si128(value, 5); break;
	case 6: value = _mm_srli_si128(value, 6); break;
	case 7: value = _mm_srli_si128(value, 7); break;
	case 8: value = _mm_srli_si128(value, 8); break;
	case 9: value = _mm_srli_si128(value, 9); break;
	case 10: value = _mm_srli_si128(value, 10); break;
	case 11: value = _mm_srli_si128(value, 11); break;
	case 12: value = _mm_srli_si128(value, 12); break;
	case 13: value = _mm_srli_si128(value, 13); break;
	case 14: value = _mm_srli_si128(value, 14); break;
	case 15: value = _mm_srli_si128(value, 15); break;
	}
	return value;
}

/* Load string at S near page boundary safely.  */
__m128i __m128i_strloadu_page_boundary(const  char *s)
{
	int offset = ((size_t) s & (16 - 1));
	if (offset)
	{
		__m128i v = _mm_load_si128((__m128i *) (s - offset));
		__m128i zero = _mm_setzero_si128();
		int bmsk = _mm_movemask_epi8(_mm_cmpeq_epi8(v, zero));
		if ((bmsk >> offset) != 0
			)  return __m128i_shift_right(v, offset);
	}
	return _mm_loadu_si128((__m128i *) s);
}
__m128i ShfLAlnLSByte(__m128i value, int offset)
{
	/*now remove constant bias, so each byte element are unsigned byte int */
	value = _mm_sub_epi8(value, _mm_setr_epi32(0x30303030, 0x30303030, 0x30303030, 0x30303030));
	switch (offset)
	{
	case 1:
		value = _mm_slli_si128(value, 1);   break;
	case 2:
		value = _mm_slli_si128(value, 2);   break;
	case 3:
		value = _mm_slli_si128(value, 3);   break;
	case 4:
		value = _mm_slli_si128(value, 4);   break;
	case 5:
		value = _mm_slli_si128(value, 5);   break;
	case 6:
		value = _mm_slli_si128(value, 6);   break;
	case 7:
		value = _mm_slli_si128(value, 7);   break;
	case 8:
		value = _mm_slli_si128(value, 8);   break;
	case 9:
		value = _mm_slli_si128(value, 9);   break;
	case 10:
		value = _mm_slli_si128(value, 10);   break;
	case 11:
		value = _mm_slli_si128(value, 11);   break;
	case 12:
		value = _mm_slli_si128(value, 12);   break;
	case 13:
		value = _mm_slli_si128(value, 13);   break;
	case 14:
		value = _mm_slli_si128(value, 14);   break;
	case 15:
		value = _mm_slli_si128(value, 15);   break;
	}
	return value;
}


