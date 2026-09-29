# Sky language spec

This folder is the source of truth for Sky's grammar and typing rules. The v0.1 scope is a checker plus an interpreter: units, symbolic tensor shapes, and matrix-property tags, checked together before anything runs. There is no codegen in v0.1.

Security requirements for v0.1:

1. `sky check` never runs user code.
2. Imports resolve only from the project root and the standard library.
3. Saved REPL sessions and caches use a data-only format.
4. The lexer rejects Unicode bidi control characters and flags confusable identifiers.
