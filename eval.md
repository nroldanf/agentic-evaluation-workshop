Interesting things to review later:
- Defeating nondeterminism in LLm inference (thinking machines)
- eval says 92%..
    - best of k tries
    - got them right 92%
    - you got lucky?
Solution:
- Give an uncertainty
- how to catch drift?

most commong metric: pass@5 (succeed a threshold in 5 tries, but pass all k should be reported)
Wilson score/distribution
    - the sample size determines the true pass rate
    - different distributions

The scores evolve with time, so to detect drift we can apply filters: CUSUM, EWMA, Shewhart
The types of drift we can detect are:
- glitch
- cliff

BH (Benjamin-Hochberg) correction for multiple hypothesis testing
BY (Benjamini-Yekutieli) correction for multiple hypothesis testing
e-BH (empirical BH) correction for multiple hypothesis testing

AUTONOMOUS EVALUATION
- Define what good means
- stop thinking what's correct -> HITL
    - every prompt change kicked off a whole new round of human review
    - turnaround: days to weeks
- Clone the expert: capture the human expert knowledge and reasoning process, and then use it to evaluate the model's output
- check what experts actually check and define the dimensions of evaluation: factual content, task match, clarity
- An LLM judge
    - reads like an auditor
    - verification mission
    - composite score - weighted sum of the dimensions
- DSPY
    - stop writing prompts, declare intent
    - task -> prompts + examples -> tune by the optimizer
    - compile a prompt
        candidate prompt, run on examples, judge panel scores it, program mutates propmts or examples, repeat until convergence
    - two optimizers: MIPROv2, GEPA