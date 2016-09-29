# C++ String to Integer Benchmark

**Note:** The SSE4.2 code needs to be compile in x64 mode.

Looping 10 million times

```
               atol:  329ms
       lexical_cast:  792ms
 std::istringstream: 6626ms <== Probably unfair comparison since istringstream instaniate a string
        std::stoull:  730ms
        my_atol_neg:   80ms
         sse4i_atol:  103ms
       boost_spirit:   93ms
```
