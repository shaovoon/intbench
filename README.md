# C++ String to Integer Benchmark

**Note:** The SSE4.2 code needs to be compile in x64 mode.

Looping 10 million times

```
	           atol:  186ms
       lexical_cast: 1096ms
 std::istringstream: 7054ms <== Probably unfair comparison since istringstream instaniate a string
         std::stoll:  715ms
        simple_atol:   97ms
         sse4i_atol:  101ms
       boost_spirit:   92ms
```
