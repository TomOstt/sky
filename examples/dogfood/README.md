# Dogfooding, round 1

Five people's worth of real work written in Sky v0.1, each by an agent playing
a scientist: a first-time user coming from Python, 1-D heat diffusion,
nonlinear model fitting with uncertainties, finite-element structures, and
mechanics/ODEs. Each folder holds the programs as written (workarounds
included), their `.out` files at the time, and a `FRICTION.md` log of every
place the language got in the way, graded blocker / major / minor.

These programs are a record, not tests: `examples/*.sky` are checked by
`crates/sky-cli/tests/examples.rs`, folders below it are not. Some outputs show
bugs that have since been fixed.

| Folder | Domain |
|---|---|
| `firstuser/` | A Python user's first hour: everyday calculations, ten deliberate mistakes, the REPL |
| `heat/` | FTCS and implicit heat equation with Dirichlet boundaries |
| `fitting/` | Gauss-Newton / Levenberg-Marquardt fits, `±` propagation, lab units |
| `structures/` | FE bar and truss assembly, modal analysis, cost contracts |
| `mechanics/` | Projectile with drag, double pendulum, figure-eight three-body, Robertson kinetics |
