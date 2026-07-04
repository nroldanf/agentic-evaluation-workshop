Let's design and implement an skill for writing langgraph agents grounded in the official documentation available online. The skill should use as references. Use as a reference the existing skill loka-strands-agentcore-docs


TODO:
- Check why is using experity and not pycon project in langfuse
- Extend the models (with other fields)
- Expand the prompt for diagnosis and assessment (it could be different, one that uses the icd10 tool and another that doesn't so we can compare the performance of both)
- Create the tool for reading the diagnosis from the parquet file `Diagnosis.parquet`