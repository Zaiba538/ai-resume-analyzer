"""LLM agents: resume, interview, skill-gap and answer-evaluator.

Each agent is a thin, typed function over the M3 foundation: it sanitises its
untrusted inputs, builds role-separated messages from a versioned prompt, runs
``StructuredLLM.generate`` (strict schema, one repair, output validation,
typed errors) and returns an ``AgentResult``. Agents hold no state, call no
tools and have no dependency on the Critic (M6).
"""
