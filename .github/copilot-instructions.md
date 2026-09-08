# oron-tts

Read [AGENTS.md](../AGENTS.md). It is the canonical brief for this repository.

It covers the failure modes that do not raise: EMA weights that synthesise
fluent non-words, out-of-vocabulary characters that silently become spaces, a
position-addressed vocabulary, a CER metric with a 0.123 floor, a
`bandwidth_hz` column that was censored before filter policy v4, and a publish
call that reports success when nothing shipped.
