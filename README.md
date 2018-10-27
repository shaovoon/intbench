# C++ String to Integer Benchmark

**Note:** The SSE4.2 code needs to be compile in x64 mode.

Looping 10 million times

```
               atol:  243ms
       lexical_cast:  952ms
 std::istringstream: 5338ms
         std::stoll:  383ms
        simple_atol:   74ms
         sse4i_atol:   72ms
       boost_spirit:   78ms
    std::from_chars:   59ms
```
