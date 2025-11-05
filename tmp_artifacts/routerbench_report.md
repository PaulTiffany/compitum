# Compitum RouterBench Evaluation Summary

This report compares Compitum against baseline routers on a bounded evaluation set.
Higher oracle_match indicates lower regret relative to the oracle assignment.

## Metrics
- compitum
  - accuracy_mean: 0.5022
  - cost_mean: 1.0159
- WizardLM/WizardLM-13B-V1.2
  - accuracy_mean: 0.4423
  - cost_mean: 0.0902
- claude-instant-v1
  - accuracy_mean: 0.3923
  - cost_mean: 0.2571
- claude-v1
  - accuracy_mean: 0.4612
  - cost_mean: 2.4918
- claude-v2
  - accuracy_mean: 0.5111
  - cost_mean: 2.6118
- gpt-3.5-turbo-1106
  - accuracy_mean: 0.5842
  - cost_mean: 0.3007
- gpt-4-1106-preview
  - accuracy_mean: 0.7091
  - cost_mean: 3.3715
- meta/code-llama-instruct-34b-chat
  - accuracy_mean: 0.4275
  - cost_mean: 0.2264
- meta/llama-2-70b-chat
  - accuracy_mean: 0.4853
  - cost_mean: 0.2640
- mistralai/mistral-7b-chat
  - accuracy_mean: 0.4401
  - cost_mean: 0.0589
- mistralai/mixtral-8x7b-chat
  - accuracy_mean: 0.5768
  - cost_mean: 0.1757
- oracle
  - accuracy_mean: 0.8704
  - cost_mean: 0.2064
- zero-one-ai/Yi-34B-Chat
  - accuracy_mean: 0.5883
  - cost_mean: 0.2368

## Where Compitum Wins
- Cost mean vs WizardLM/WizardLM-13B-V1.2: +0.9257
- Accuracy mean vs WizardLM/WizardLM-13B-V1.2: +0.0599
- Cost mean vs claude-instant-v1: +0.7588
- Accuracy mean vs claude-instant-v1: +0.1100
- Cost mean vs claude-v1: -1.4759
- Accuracy mean vs claude-v1: +0.0410
- Cost mean vs claude-v2: -1.5958
- Accuracy mean vs claude-v2: -0.0089
- Cost mean vs gpt-3.5-turbo-1106: +0.7152
- Accuracy mean vs gpt-3.5-turbo-1106: -0.0820
- Cost mean vs gpt-4-1106-preview: -2.3555
- Accuracy mean vs gpt-4-1106-preview: -0.2069
- Cost mean vs meta/code-llama-instruct-34b-chat: +0.7895
- Accuracy mean vs meta/code-llama-instruct-34b-chat: +0.0747
- Cost mean vs meta/llama-2-70b-chat: +0.7519
- Accuracy mean vs meta/llama-2-70b-chat: +0.0169
- Cost mean vs mistralai/mistral-7b-chat: +0.9570
- Accuracy mean vs mistralai/mistral-7b-chat: +0.0622
- Cost mean vs mistralai/mixtral-8x7b-chat: +0.8402
- Accuracy mean vs mistralai/mixtral-8x7b-chat: -0.0745
- Cost mean vs oracle: +0.8095
- Accuracy mean vs oracle: -0.3682
- Cost mean vs zero-one-ai/Yi-34B-Chat: +0.7791
- Accuracy mean vs zero-one-ai/Yi-34B-Chat: -0.0860

### Regret (accuracy gap to oracle)
- Compitum: +0.3682
- WizardLM/WizardLM-13B-V1.2: +0.4281
- claude-instant-v1: +0.4782
- claude-v1: +0.4092
- claude-v2: +0.3593
- gpt-3.5-turbo-1106: +0.2863
- gpt-4-1106-preview: +0.1613
- meta/code-llama-instruct-34b-chat: +0.4429
- meta/llama-2-70b-chat: +0.3851
- mistralai/mistral-7b-chat: +0.4304
- mistralai/mixtral-8x7b-chat: +0.2937
- zero-one-ai/Yi-34B-Chat: +0.2822

## Determinism
Compitum routing is deterministic given fixed models and parameters, reducing variance and
improving reproducibility compared to routers relying on stochastic LLM calls for decisions.
