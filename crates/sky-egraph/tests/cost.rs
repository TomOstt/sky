//! The extraction cost is the static cost model of `sky_ir::cost`: prices,
//! weights, and the `EG-COST-001` explanation that says why a form was chosen.

mod common;

use common::*;
use sky_egraph::{
    matmul_cost, op_cost, rewrite_function, show_poly, RewriteConfig, ALLOC_FLOPS, DEFAULT_SIZE,
};
use sky_ir::{NumberType, OpKind, Repr, Type};
use sky_types::{Shape, ShapeDim, ShapeVar, TagSet};
use sky_units::Dimension;

fn num_ty(shape: Shape) -> NumberType {
    NumberType {
        dim: Dimension::DIMENSIONLESS,
        shape,
        tags: TagSet::EMPTY,
        repr: Repr::Unchosen,
    }
}

fn size(name: &str) -> ShapeDim {
    ShapeDim::Var(ShapeVar::from_parts(1, name))
}

#[test]
fn prices_come_from_the_static_cost_model() {
    let n = size("N");
    let m = num_ty(Shape::matrix(n.clone(), n.clone()));
    let v = num_ty(Shape::vector(n));
    // matmul: 2·n·k·m.
    let mm = matmul_cost(&m, &m, &m).expect("priced");
    assert_eq!(show_poly(&mm.flops), "2·N³");
    let mv = op_cost(&OpKind::MatMul, &[m.clone(), v.clone()], &v).expect("priced");
    assert_eq!(show_poly(&mv.flops), "2·N²");
    // Elementwise: one flop per entry, three arrays of traffic, one
    // allocation.
    let add = op_cost(&OpKind::Add, &[v.clone(), v.clone()], &v).expect("priced");
    assert_eq!(show_poly(&add.flops), "N");
    assert_eq!(show_poly(&add.bytes), "24·N");
    assert_eq!(show_poly(&add.allocs), "1");
    // A transpose is a copy.
    let tr = op_cost(&OpKind::Transpose, &[m.clone()], &m).expect("priced");
    assert_eq!(show_poly(&tr.flops), "0");
    assert_eq!(show_poly(&tr.bytes), "16·N²");
    // Scalars: a flop each, transcendentals 20, no traffic.
    let s = num_ty(Shape::scalar());
    let exp = op_cost(&OpKind::Math(sky_ir::MathFn::Exp), &[s.clone()], &s).expect("priced");
    assert_eq!(show_poly(&exp.flops), "20");
    assert_eq!(show_poly(&exp.bytes), "0");
    let sqrt = op_cost(&OpKind::Math(sky_ir::MathFn::Sqrt), &[s.clone()], &s).expect("priced");
    assert_eq!(show_poly(&sqrt.flops), "1");
    let div = op_cost(&OpKind::Div, &[s.clone(), s.clone()], &s).expect("priced");
    assert_eq!(show_poly(&div.flops), "1");
}

#[test]
fn big_array_operations_dominate_small_scalar_ones() {
    let n = size("N");
    let m = num_ty(Shape::matrix(n.clone(), n.clone()));
    let v = num_ty(Shape::vector(n));
    let s = num_ty(Shape::scalar());
    let exp = op_cost(&OpKind::Math(sky_ir::MathFn::Exp), &[s.clone()], &s).expect("priced");
    let div = op_cost(&OpKind::Div, &[s.clone(), s], &num_ty(Shape::scalar())).expect("priced");
    let add = op_cost(&OpKind::Add, &[v.clone(), v.clone()], &v).expect("priced");
    let mv = op_cost(&OpKind::MatMul, &[m.clone(), v.clone()], &v).expect("priced");
    let mm = matmul_cost(&m, &m, &m).expect("priced");
    // A symbolic size is 64.
    assert_eq!(DEFAULT_SIZE, 64.0);
    assert!(add.weighted() > exp.weighted());
    assert!(mv.weighted() > 100 * div.weighted());
    assert!(mm.weighted() > 10 * mv.weighted());
    // The weight is 8·flops + bytes + 8·ALLOC_FLOPS·allocations (eighths of
    // a flop) at the default size.
    let (flops, bytes, allocs) = (64u64, 24 * 64, 1);
    assert_eq!(add.weighted(), 8 * flops + bytes + 8 * ALLOC_FLOPS * allocs);
}

/// Where the model orders two forms for every size, the explanation says so,
/// with the polynomials, for an array-valued result.
#[test]
fn an_extracted_array_form_is_explained_with_its_costs() {
    let m = |r: u64, c: u64| Type::Number(num_ty(Shape::matrix(r, c)));
    let mut t = T::new(vec![m(2, 3)], vec![m(2, 3)]);
    let a = t.params()[0];
    t.b.set_name(a, "A");
    let x = t.op(OpKind::Transpose, &[a], m(3, 2));
    let y = t.op(OpKind::Transpose, &[x], m(2, 3));
    let mut f = t.ret(&[y]);
    let rw = rewrite_function(&mut f, &RewriteConfig::default());
    let rules: Vec<_> = rw.iter().map(|r| r.rule).collect();
    assert_eq!(rules, ["LA-002", "EG-COST-001"]);
    assert_eq!(rw[1].before, "(Aᵀ)ᵀ");
    assert_eq!(rw[1].after, "A");
    assert_eq!(
        rw[1].because,
        "chose `A` over `(Aᵀ)ᵀ`: 0 vs 0 flops, 0 vs 192 bytes, 0 vs 2 allocations \
         (weighted cost 0 vs 152)"
    );
}

/// A scalar result gets no cost explanation: its own rule says it all.
#[test]
fn scalar_results_are_not_given_a_cost_explanation() {
    let mut t = T::new(vec![num()], vec![num()]);
    let x = t.params()[0];
    let one = t.rat(1, 1);
    let y = t.bin(OpKind::Mul, x, one);
    let mut f = t.ret(&[y]);
    let rw = rewrite_function(&mut f, &RewriteConfig::default());
    assert_eq!(rw.len(), 1);
    assert_eq!(rw[0].rule, "ALG-001");
}
