#![allow(unused)]
use crate::{pedersen, transcript::Transcript};
use anyhow::Result;
use p1::Zero;
use p1::poly::Multilinear;
use p2::ec::{EllipticCurve, ScalarOf};
use serde::{Deserialize, Serialize};

pub type PublicParams<E> = pedersen::dot_product::PublicParams<E>;

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(deserialize = ""))]
pub struct Statement<E: EllipticCurve> {
    /// Commitments to the rows of the evaluation matrix (sqrt(N) elements).
    pub comm_rows: Vec<E>,
    /// The public evaluation point `r = (r_1, ..., r_l)`.
    pub point: Vec<E::Scalar>,
    /// Commitment to the evaluation `f(r)`.
    pub comm_eval: E,
}

impl<E: EllipticCurve> Statement<E> {
    /// Creates a new Quokka opening statement with validation.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `point` has odd length
    /// - `comm_rows.len()` does not equal `2^(point.len()/2)`
    pub fn new(comm_rows: Vec<E>, point: Vec<E::Scalar>, comm_eval: E) -> Self {
        assert!(point.len().is_multiple_of(2), "point must have even length");
        let num_rows = 1usize << (point.len() / 2);
        assert_eq!(comm_rows.len(), num_rows, "comm_rows.len() != sqrt(N)");
        Self {
            comm_rows,
            point,
            comm_eval,
        }
    }
}

#[derive(Clone, Debug)]
pub struct Witness<E: EllipticCurve> {
    /// The multilinear polynomial in evaluation form.
    pub poly: Multilinear<E::Scalar>,
    /// Blinding factors for each row commitment (sqrt(N) elements).
    pub openings: Vec<E::Scalar>,
    /// The evaluation `f(r)`.
    pub eval: E::Scalar,
    /// Blinding factor for `comm_eval`.
    pub r_eval: E::Scalar,
}

impl<E: EllipticCurve> Witness<E> {
    /// Creates a new Quokka witness with validation.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The polynomial has an odd number of variables
    /// - `openings.len()` does not equal `2^(num_vars/2)`
    pub fn new(
        poly: Multilinear<E::Scalar>,
        openings: Vec<E::Scalar>,
        eval: E::Scalar,
        r_eval: E::Scalar,
    ) -> Self {
        assert!(
            poly.num_vars().is_multiple_of(2),
            "poly must have even number of variables"
        );
        let num_rows = 1usize << (poly.num_vars() / 2);
        assert_eq!(openings.len(), num_rows, "openings.len() != sqrt(N)");
        Self {
            poly,
            openings,
            eval,
            r_eval,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(deserialize = ""))]
pub struct Proof<E: EllipticCurve> {
    pub dp_proof: pedersen::dot_product::Proof<E>,
}

pub type Commitment<E> = Vec<E>;

/// Commit to a multilinear polynomial by arranging its evaluations into a √N × √N
/// matrix and vector-committing each row.
///
/// Returns √N row commitments. Requires an even number of variables.
/// `randomness` provides one blinding factor per row (the "openings").
pub fn commit<E: EllipticCurve>(
    poly: &Multilinear<E::Scalar>,
    randomness: &[E::Scalar],
    params: &PublicParams<E>,
) -> Commitment<E> {
    // test
    assert!(poly.num_vars().is_multiple_of(2));
    let num_rows = 1 << (poly.num_vars() / 2);
    assert_eq!(params.vec_gens.len(), num_rows);
    assert_eq!(randomness.len(), num_rows);

    poly.evals
        .chunks(num_rows)
        .zip(randomness)
        .map(|(chunk, &r)| {
            pedersen::vector_commit(chunk, r, &params.vec_gens, params.scalar_gens[1])
        })
        .collect()
}

/// Prove that the committed polynomial evaluates to the committed value at the
/// given public point.
///
/// Recall from project 2 that the evaluation `f(r)` can be decomposed as
/// `f(r) = <a, b^T * M>`, where `a = eq_tilde(r_bot)`, `b = eq_tilde(r_top)`,
/// and `M` is the evaluation matrix. The non-ZK prover sent `b^T * M` in the
/// clear; here, we must keep it hidden.
///
/// The key observation is that the row commitments are vector Pedersen
/// commitments, so `MSM(b, comm_rows)` is *already* a vector commitment to
/// `b^T * M` (think about why -- this follows from the homomorphic property).
/// You need to figure out what the corresponding randomness is.
///
/// Once you have a vector commitment to `b^T * M` and a scalar commitment to
/// the evaluation (given in the statement), the remaining task is to prove
/// a dot-product relation using the sigma protocol from `pedersen::dot_product`.
///
/// Both prover and verifier must append the same messages to the transcript
/// before invoking the dot product sub-protocol.
#[must_use]
pub fn prove<E: EllipticCurve>(
    params: &PublicParams<E>,
    statement: &Statement<E>,
    witness: &Witness<E>,
    trans: &mut Transcript,
    mut rng: impl rand::Rng,
) -> Proof<E> {
    // Cross-consistency between statement and witness.
    assert_eq!(statement.point.len(), witness.poly.num_vars(), "point.len() != num_vars");

    let l = statement.point.len();
    assert!(l.is_multiple_of(2), "point must have even length");

    let num_rows = 1usize << (l / 2);
    assert_eq!(statement.comm_rows.len(), num_rows, "comm_rows.len() != sqrt(N)");
    assert_eq!(params.vec_gens.len(), num_rows, "vec_gens.len() != sqrt(N)");
    assert_eq!(witness.openings.len(), num_rows, "openings.len() != sqrt(N)");
    assert_eq!(witness.poly.evals.len(), num_rows * num_rows, "poly.evals.len() != N");

    let r_bot = &statement.point[..(l / 2)];
    let r_top = &statement.point[(l / 2)..];

    let a = Multilinear::<E::Scalar>::eq_tilde(r_bot).evals;
    let b = Multilinear::<E::Scalar>::eq_tilde(r_top).evals;
    assert_eq!(a.len(), num_rows, "eq_tilde(r_bot).len() != sqrt(N)");
    assert_eq!(b.len(), num_rows, "eq_tilde(r_top).len() != sqrt(N)");

    // Compute c = b^T M, the hidden row-combination vector.
    let mut c = vec![E::Scalar::zero(); num_rows];
    for (row, &b_i) in witness.poly.evals.chunks(num_rows).zip(&b) {
        for (c_j, &m_ij) in c.iter_mut().zip(row) {
            *c_j += b_i * m_ij;
        }
    }

    let derived_eval = pedersen::inner_product(&a, &c);
    assert_eq!(witness.eval, derived_eval, "witness.eval != f(point)");

    // Homomorphically derive the commitment to c = b^T M and its randomness.
    let comm_bm = E::msm(&b, &statement.comm_rows);
    let r_bm = pedersen::inner_product(&b, &witness.openings);

    trans.append_message("quokka_open_protocol", ());
    trans.append_message("quokka_open_params", params);
    trans.append_message("quokka_open_statement", statement);
    trans.append_message("quokka_open_derived_comm", &comm_bm);

    let dp_statement = pedersen::dot_product::Statement {
        a,
        comm_x: comm_bm,
        comm_result: statement.comm_eval,
    };
    let dp_witness = pedersen::dot_product::Witness {
        x: c,
        r_x: r_bm,
        result: witness.eval,
        r_result: witness.r_eval,
    };

    let dp_proof = pedersen::dot_product::prove(params, &dp_statement, &dp_witness, trans, &mut rng);
    Proof { dp_proof }
}

/// Verify an opening proof.
///
/// The verifier must reconstruct the same dot-product statement as the prover: compute `a` and `b`
/// from the evaluation point, derive the vector commitment to `b^T * M`  and then verify the
/// dot-product proof.
pub fn verify<E: EllipticCurve>(
    params: &PublicParams<E>,
    statement: &Statement<E>,
    proof: &Proof<E>,
    trans: &mut Transcript,
) -> Result<()> {
    let l = statement.point.len();
    anyhow::ensure!(l.is_multiple_of(2), "point must have even length");

    let num_rows = 1usize << (l / 2);
    anyhow::ensure!(statement.comm_rows.len() == num_rows, "comm_rows.len() != sqrt(N)");
    anyhow::ensure!(params.vec_gens.len() == num_rows, "vec_gens.len() != sqrt(N)");

    let r_bot = &statement.point[..(l / 2)];
    let r_top = &statement.point[(l / 2)..];

    let a = Multilinear::<E::Scalar>::eq_tilde(r_bot).evals;
    let b = Multilinear::<E::Scalar>::eq_tilde(r_top).evals;
    anyhow::ensure!(a.len() == num_rows, "eq_tilde(r_bot).len() != sqrt(N)");
    anyhow::ensure!(b.len() == num_rows, "eq_tilde(r_top).len() != sqrt(N)");

    let comm_bm = E::msm(&b, &statement.comm_rows);

    trans.append_message("quokka_open_protocol", ());
    trans.append_message("quokka_open_params", params);
    trans.append_message("quokka_open_statement", statement);
    trans.append_message("quokka_open_derived_comm", &comm_bm);

    let dp_statement = pedersen::dot_product::Statement {
        a,
        comm_x: comm_bm,
        comm_result: statement.comm_eval,
    };

    pedersen::dot_product::verify(params, &dp_statement, &proof.dp_proof, trans)
}
