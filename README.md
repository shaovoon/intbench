# C++ String to Integer Benchmark

**Note:** The SSE4.2 code needs to be compile in x64 mode.

Looping 10 million times

```
               atol:  213ms
       lexical_cast: 1267ms
 std::istringstream: 7269ms <== Probably unfair comparison since istringstream instaniate a string
         std::stoll:  466ms
        simple_atol:   94ms
         sse4i_atol:   92ms
       boost_spirit:   99ms
    std::from_chars:   64ms

```
