# Sky

Sky is a scientific programming language that checks physical units, tensor shapes, and matrix properties together at compile time, in syntax that reads like Python.

This repo is at the very start. Right now only the workspace layout and the lexer's safety check exist. See `spec/` for the v0.1 scope.

```
crates/
  sky-syntax/   lexer and parser
  sky-ast/      typed AST with source spans
  sky-units/    dimensions and unit tables
  sky-types/    type checker (units, shapes, matrix properties)
  sky-interp/   tree-walking interpreter
  sky-cli/      the `sky` command: check, run, repl
spec/           grammar and typing rules
tests/ui/       golden diagnostic tests
```

Licensed under Apache-2.0.
