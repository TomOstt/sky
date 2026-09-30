//! Lowering of arrays: literals and their types, indexing and slicing,
//! the builtins, and the Cholesky solve. What the operations compute is
//! tested by the differential tests, against `sky run`.

mod common;

use common::*;
use sky_ir::{OpKind, SliceAxis, Type};
use sky_types::{ShapeDim, Tag};

/// The number type of value `v`.
fn number(f: &sky_ir::Function, v: sky_ir::ValueId) -> &sky_ir::NumberType {
    f.ty(v).as_number().expect("a number")
}

fn dims(n: &sky_ir::NumberType) -> Vec<u64> {
    n.shape
        .dims()
        .iter()
        .map(|d| match d {
            ShapeDim::Known(k) => *k,
            ShapeDim::Var(_) => u64::MAX,
        })
        .collect()
}

#[test]
fn a_vector_literal_is_one_array_literal_of_its_entries() {
    let m = lower_ok("v = [1 m, 2 m, 3 m]\n");
    let f = main_fn(&m);
    assert_eq!(top_kinds(f)[..3], ["const.float"; 3]);
    let v = named(f, "v");
    let op = f.op(def_op(f, v).expect("op"));
    assert_eq!(op.kind, OpKind::ArrayLiteral { cols: None });
    assert_eq!(op.operands.len(), 3);
    let n = number(f, v);
    assert_eq!(dims(n), [3]);
    assert_eq!(n.dim, sky_units::Dimension::LENGTH);
    assert!(n.tags.is_empty());
}

#[test]
fn a_matrix_literal_is_row_major_with_its_row_length() {
    let m = lower_ok("A = [[1, 2, 3], [4, 5, 6]]\nR = [[1, 2, 3]]\n");
    let f = main_fn(&m);
    let a = f.op(def_op(f, named(f, "A")).expect("op"));
    assert_eq!(a.kind, OpKind::ArrayLiteral { cols: Some(3) });
    assert_eq!(a.operands.len(), 6);
    assert_eq!(dims(number(f, named(f, "A"))), [2, 3]);
    // One row is a matrix, not a vector.
    let r = f.op(def_op(f, named(f, "R")).expect("op"));
    assert_eq!(r.kind, OpKind::ArrayLiteral { cols: Some(3) });
    assert_eq!(dims(number(f, named(f, "R"))), [1, 3]);
    // Entries are lowered left to right, row by row.
    let consts: Vec<f64> = a
        .operands
        .iter()
        .map(|&v| {
            let OpKind::ConstFloat { bits } = f.op(def_op(f, v).expect("op")).kind else {
                panic!("not a constant");
            };
            f64::from_bits(bits)
        })
        .collect();
    assert_eq!(consts, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn entries_can_be_any_scalar_expression() {
    let m = lower_ok("x = 2 m\nv = [x, x * 2, x + 1 m]\n");
    let f = main_fn(&m);
    assert_eq!(dims(number(f, named(f, "v"))), [3]);
    assert_eq!(count(f, "array.literal"), 1);
}

#[test]
fn a_literal_with_an_entry_of_unknown_type_is_unknown_but_keeps_its_arrangement() {
    let m = lower_ok("def pair(x, y):\n    return [[x, y]]\n");
    let f = func(&m, "pair");
    let lit = f
        .ops
        .iter()
        .find(|o| matches!(o.kind, OpKind::ArrayLiteral { .. }))
        .expect("a literal");
    assert_eq!(lit.kind, OpKind::ArrayLiteral { cols: Some(2) });
    assert_eq!(f.ty(lit.results[0]), &Type::Unknown);
}

#[test]
fn a_declared_tag_on_a_literal_is_on_the_value_bound_to_the_name() {
    let m = lower_ok(
        "P: PositiveDefinite Matrix[2, 2] = [[4, 1], [1, 3]]\n\
         Q = [[4, 1], [1, 3]]\n\
         D: Diagonal Matrix[2, 2] = [[1, 0], [0, 2]]\n",
    );
    let f = main_fn(&m);
    let p = number(f, named(f, "P"));
    assert!(p.tags.contains(Tag::PositiveDefinite) && p.tags.contains(Tag::Symmetric));
    // The same literal, undeclared, has no tags of its own.
    assert!(number(f, named(f, "Q")).tags.is_empty());
    let d = number(f, named(f, "D"));
    assert!(d.tags.contains(Tag::Diagonal) && !d.tags.contains(Tag::PositiveDefinite));
}

#[test]
fn a_position_is_index_and_a_range_is_slice() {
    let m = lower_ok(
        "A = [[1, 2, 3], [4, 5, 6]]\ni = 1\n\
         e = A[i, 2]\nrow = A[0]\ncol = A[:, 1]\nblock = A[0:1, 1:3]\ntail = A[1, 1:]\nhead = A[0, :2]\n",
    );
    let f = main_fn(&m);
    let kind = |name: &str| f.op(def_op(f, named(f, name)).expect("op")).kind.clone();
    assert_eq!(kind("e"), OpKind::Index);
    assert_eq!(kind("row"), OpKind::Index);
    let range = |start, stop| SliceAxis::Range { start, stop };
    assert_eq!(
        kind("col"),
        OpKind::Slice {
            axes: vec![range(false, false), SliceAxis::At]
        }
    );
    assert_eq!(
        kind("block"),
        OpKind::Slice {
            axes: vec![range(true, true), range(true, true)]
        }
    );
    assert_eq!(
        kind("tail"),
        OpKind::Slice {
            axes: vec![SliceAxis::At, range(true, false)]
        }
    );
    assert_eq!(
        kind("head"),
        OpKind::Slice {
            axes: vec![SliceAxis::At, range(false, true)]
        }
    );
    // Operands: the base, then each item's expressions in order.
    let block = f.op(def_op(f, named(f, "block")).expect("op"));
    assert_eq!(block.operands.len(), 5);
    assert_eq!(dims(number(f, named(f, "e"))), Vec::<u64>::new());
    assert_eq!(dims(number(f, named(f, "row"))), [3]);
    assert_eq!(dims(number(f, named(f, "col"))), [2]);
    assert_eq!(dims(number(f, named(f, "block"))), [1, 2]);
}

#[test]
fn an_index_keeps_the_dimension_and_takes_a_computed_position() {
    let m = lower_ok("v = [1 m, 2 m]\ni = 1\nx = v[i + 0]\n");
    let f = main_fn(&m);
    let x = named(f, "x");
    assert_eq!(number(f, x).dim, sky_units::Dimension::LENGTH);
    let op = f.op(def_op(f, x).expect("op"));
    assert_eq!(op.kind, OpKind::Index);
    // The position is the sum, not a constant.
    assert_eq!(
        f.op(def_op(f, op.operands[1]).expect("op")).kind,
        OpKind::Add
    );
}

#[test]
fn len_is_a_constant_where_the_shape_is_known_and_an_operation_where_it_is_not() {
    let m = lower_ok(
        "v = [1, 2, 3]\nA = [[1, 2], [3, 4], [5, 6]]\nn = len(v)\nk = len(A)\n\
         def count(u: Vector[N]):\n    return len(u)\n\
         def anything(a):\n    return len(a)\n",
    );
    let f = main_fn(&m);
    for (name, want) in [("n", 3.0), ("k", 3.0)] {
        let op = f.op(def_op(f, named(f, name)).expect("op"));
        let OpKind::ConstFloat { bits } = op.kind else {
            panic!("len is not folded\n{}", dump(f))
        };
        assert_eq!(f64::from_bits(bits), want);
        assert!(number(f, named(f, name)).dim.is_dimensionless());
    }
    assert_eq!(count(f, "len"), 0);
    assert_eq!(count(func(&m, "count"), "len"), 1);
    assert_eq!(count(func(&m, "anything"), "len"), 1);
}

#[test]
fn a_folded_len_still_evaluates_its_operand() {
    // The operand's failures and effects stay: the `inverse` is computed.
    let m = lower_ok("A = [[1, 2], [3, 4]]\nn = len(A⁻¹)\n");
    let f = main_fn(&m);
    assert_eq!(count(f, "inverse"), 1);
    assert_eq!(count(f, "len"), 0);
}

#[test]
fn zeros_and_identity_take_their_sizes_as_operands_and_are_typed_by_the_checker() {
    let m = lower_ok(
        "z = zeros(3)\nZ = zeros(2, 3)\nI = identity(3)\nS = zeros(2, 2)\n\
         def sized(n):\n    return zeros(n)\n",
    );
    let f = main_fn(&m);
    let op = |name: &str| f.op(def_op(f, named(f, name)).expect("op"));
    assert_eq!(op("z").kind, OpKind::Zeros);
    assert_eq!(op("z").operands.len(), 1);
    assert_eq!(dims(number(f, named(f, "z"))), [3]);
    assert_eq!(op("Z").operands.len(), 2);
    assert_eq!(dims(number(f, named(f, "Z"))), [2, 3]);
    assert_eq!(op("I").kind, OpKind::Identity);
    let eye = number(f, named(f, "I"));
    assert_eq!(dims(eye), [3, 3]);
    assert!(eye.tags.contains(Tag::Orthogonal) && eye.tags.contains(Tag::Diagonal));
    // An all-zero square matrix is Diagonal, as the checker says.
    assert!(number(f, named(f, "S")).tags.contains(Tag::Diagonal));
    // `zeros(n)` for a parameter `n` of unknown type: the checker cannot
    // type the result, so it is Unknown and the size is the operand.
    let g = func(&m, "sized");
    let z = g.op(top_op(g, "zeros"));
    assert_eq!(g.ty(z.results[0]), &Type::Unknown);
    assert_eq!(z.operands, g.param_values());
}

#[test]
fn a_recorded_cholesky_site_is_a_solve_and_an_untagged_one_an_lu_solve() {
    let m = lower_ok(
        "K: PositiveDefinite Matrix[2, 2] = [[4, 1], [1, 3]]\nf = [1, 2]\n\
         a = K⁻¹ @ f\nb = inv(K) @ f\nc = (K⁻¹) @ f\nd = (K + K)⁻¹ @ f\ne = K @ f\n",
    );
    let f = main_fn(&m);
    // `d` is a solve too, an LU one: `(K + K)` carries no tags.
    assert_eq!(count(f, "solve"), 4);
    assert_eq!(count(f, "inverse"), 0);
    assert_eq!(count(f, "matmul"), 1);
    for name in ["a", "b", "c"] {
        let op = f.op(def_op(f, named(f, name)).expect("op"));
        assert_eq!(
            op.kind,
            OpKind::Solve {
                method: sky_ir::SolveMethod::Cholesky
            },
            "{name}"
        );
        assert_eq!(
            f.op(def_op(f, op.operands[0]).expect("op")).results[0],
            named(f, "K")
        );
        assert_eq!(op.operands[1], named(f, "f"));
        assert_eq!(dims(number(f, named(f, name))), [2]);
    }
    // `(K + K)⁻¹ @ f` solves against the sum, by LU.
    let d = f.op(def_op(f, named(f, "d")).expect("op"));
    assert_eq!(
        d.kind,
        OpKind::Solve {
            method: sky_ir::SolveMethod::Lu
        }
    );
    let operand = f.op(def_op(f, d.operands[0]).expect("op"));
    assert_eq!(operand.kind, OpKind::Add);
}

#[test]
fn an_inverse_of_an_untagged_matrix_is_an_lu_solve_and_a_named_inverse_stays() {
    let m = lower_ok("K = [[4, 1], [1, 3]]\nf = [1, 2]\nu = K⁻¹ @ f\n");
    let f = main_fn(&m);
    assert_eq!(count(f, "solve"), 1);
    let u = f.op(def_op(f, named(f, "u")).expect("op"));
    assert_eq!(
        u.kind,
        OpKind::Solve {
            method: sky_ir::SolveMethod::Lu
        }
    );
    assert_eq!(top_kinds(f).iter().filter(|k| **k == "inverse").count(), 0);
    // An inverse the program names is formed: `M @ f` is a product.
    let m = lower_ok("K = [[4, 1], [1, 3]]\nf = [1, 2]\nM = K⁻¹\nu = M @ f\n");
    let f = main_fn(&m);
    assert_eq!(count(f, "solve"), 0);
    assert_eq!(top_kinds(f).iter().filter(|k| **k == "inverse").count(), 1);
    assert_eq!(top_kinds(f).iter().filter(|k| **k == "matmul").count(), 1);
}

#[test]
fn a_function_reading_a_positive_definite_module_name_solves() {
    let m = lower_ok(
        "K: PositiveDefinite Matrix[2, 2] = [[4, 1], [1, 3]]\n\
         def displace(f: Vector[2]):\n    return K⁻¹ @ f\n",
    );
    let f = func(&m, "displace");
    assert_eq!(count(f, "solve"), 1, "{}", dump(f));
    // The module name arrives as a parameter typed with the tags.
    let k = number(f, *f.param_values().last().expect("K"));
    assert!(k.tags.contains(Tag::PositiveDefinite));
}

#[test]
fn a_declared_matrix_input_is_an_input_operation_in_main() {
    let m = lower_ok("A: Matrix[3, 3]\nb: Vector[3]\nx = A @ b\ny = A[0, 1]\nn = len(b)\n");
    let f = main_fn(&m);
    assert_eq!(f.param_values().len(), 0);
    assert_eq!(count(f, "input"), 2);
    assert_eq!(dims(number(f, named(f, "A"))), [3, 3]);
    assert_eq!(dims(number(f, named(f, "x"))), [3]);
}

#[test]
fn the_array_operations_are_verified_and_print_and_parse_back() {
    let src = "\
v = [1 m, 2 m, 3 m]\nA = [[1, 2], [3, 4]]\nK: PositiveDefinite Matrix[2, 2] = [[4, 1], [1, 3]]\n\
i = 1\nprint(v[i], A[:, 1], A[0:1, 1:2], zeros(2, 2), identity(2), len(v), K⁻¹ @ [1, 2])\n";
    let m = lower_ok(src);
    let text = sky_ir::print(&m);
    let back = sky_ir::parse(&text).expect("parses");
    assert_eq!(sky_ir::print(&back), text);
    assert!(sky_ir::verify(&back).is_ok());
}
