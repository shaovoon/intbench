# C++ String to Integer Benchmark

**Note:** The SSE4.2 code needs to be compile in x64 mode.

```
               atol:  329ms
       lexical_cast:  792ms
 std::istringstream: 6626ms
        std::stoull:  730ms
        my_atol_neg:   80ms
         sse4i_atol:  103ms
       boost_spirit:   93ms
   boost_spirit_chr:   96ms
```
