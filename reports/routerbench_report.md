# Compitum RouterBench Evaluation Summary

This report compares Compitum against baseline routers on a bounded evaluation set.
Higher oracle_match indicates lower regret relative to the oracle assignment.

## Metrics
- compitum
  - accuracy_mean: 0.7091
  - cost_mean: 3.3715
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
- Cost mean vs WizardLM/WizardLM-13B-V1.2: +3.2813
- Accuracy mean vs WizardLM/WizardLM-13B-V1.2: +0.2668
- Cost mean vs claude-instant-v1: +3.1144
- Accuracy mean vs claude-instant-v1: +0.3168
- Cost mean vs claude-v1: +0.8796
- Accuracy mean vs claude-v1: +0.2479
- Cost mean vs claude-v2: +0.7597
- Accuracy mean vs claude-v2: +0.1980
- Cost mean vs gpt-3.5-turbo-1106: +3.0708
- Accuracy mean vs gpt-3.5-turbo-1106: +0.1249
- Cost mean vs gpt-4-1106-preview: +0.0000
- Accuracy mean vs gpt-4-1106-preview: +0.0000
- Cost mean vs meta/code-llama-instruct-34b-chat: +3.1450
- Accuracy mean vs meta/code-llama-instruct-34b-chat: +0.2816
- Cost mean vs meta/llama-2-70b-chat: +3.1075
- Accuracy mean vs meta/llama-2-70b-chat: +0.2238
- Cost mean vs mistralai/mistral-7b-chat: +3.3125
- Accuracy mean vs mistralai/mistral-7b-chat: +0.2691
- Cost mean vs mistralai/mixtral-8x7b-chat: +3.1957
- Accuracy mean vs mistralai/mixtral-8x7b-chat: +0.1324
- Cost mean vs oracle: +3.1651
- Accuracy mean vs oracle: -0.1613
- Cost mean vs zero-one-ai/Yi-34B-Chat: +3.1346
- Accuracy mean vs zero-one-ai/Yi-34B-Chat: +0.1209

### Regret (accuracy gap to oracle)
- Compitum: +0.1613
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
