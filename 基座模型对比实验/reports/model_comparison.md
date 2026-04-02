# 基座模型评估对比

| Model                        |   Executable Rate |   Field Validity Rate |   Fallback Rate |   Avg Modified Fields | Cost Proxy   |
|:-----------------------------|------------------:|----------------------:|----------------:|----------------------:|:-------------|
| claude-3-5-sonnet            |                 1 |                  0.6  |               0 |                     5 | 1961 chars   |
| deepseek-r1-70b              |                 1 |                  1    |               0 |                    12 | 3923 chars   |
| gemma3-27b                   |                 1 |                  0.86 |               0 |                    14 | 5969 chars   |
| gpt-oss-20b                  |                 1 |                  1    |               0 |                     6 | 2092 chars   |
| llama3_1-70b-instruct-q4_K_M |                 1 |                  0.87 |               0 |                    15 | 5365 chars   |
| mistral.mistral-large-2402   |                 1 |                  1    |               0 |                     3 | 1110 chars   |
| qwen3-32b                    |                 1 |                  0.64 |               0 |                    11 | 3559 chars   |
