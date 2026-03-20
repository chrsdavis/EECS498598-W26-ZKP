use crate::{
    pedersen::{self, CommittedValue},
    transcript::Transcript,
};
use anyhow::Result;
use itertools::Itertools;
use p1::{Random, Zero};
use p2::{combined::CombinedMLE, ec::EllipticCurve};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PublicParams<E> {
    /// Vector generators for committing round polynomial coefficients (`num_vars * (max_degree + 1)` elements).
    /// Each round uses a distinct chunk of `max_degree + 1` generators.
    pub vec_gens: Vec<E>,
    /// Scalar generators `[g, h]` for Pedersen commitments.
    pub scalar_gens: [E; 2],
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Statement<E> {
    /// Commitment to the claimed sum.
    pub comm_sum: E,
    /// Number of variables in the polynomial.
    pub num_vars: usize,
    /// Maximum degree of the polynomial in any single variable.
    pub max_degree: usize,
}

#[derive(Clone, Debug)]
pub struct Witness<E: EllipticCurve> {
    /// The polynomial to sum over the hypercube.
    pub polynomial: CombinedMLE<E::Scalar>,
    /// The claimed sum value.
    pub sum: E::Scalar,
    /// Blinding factor for `comm_sum`.
    pub r_sum: E::Scalar,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(deserialize = ""))]
pub struct Proof<E: EllipticCurve> {
    /// One vector commitment per round (to the round polynomial coefficients).
    pub round_commitments: Vec<E>,
    /// Commitment to the final evaluation `g(r_0, ..., r_{n-1})`.
    pub comm_final: E,
    /// Dot product proof for the batched check.
    pub dp_proof: pedersen::dot_product::Proof<E>,
}

fn pad_coeffs<F>(coeffs: &[F], len: usize) -> Vec<F>
where
    F: Copy + Zero,
{
    let mut out = coeffs.to_vec();
    out.resize(len, F::zero());
    out
}

fn batch_coeffs<F>(challenges: &[F], rhos: &[F], max_degree: usize) -> Vec<F>
where
    F: p1::Field,
{
    let num_vars = challenges.len();
    let block = max_degree + 1;
    assert_eq!(rhos.len(), num_vars + 1, "rhos.len() must equal num_vars + 1");

    let mut y = vec![F::zero(); num_vars * block];
    if num_vars == 0 {
        return y;
    }

    // check g_1(0) + g_1(1) == claimed sum
    y[0] += rhos[0] * F::from(2u64);
    for deg in 1..block {
        y[deg] += rhos[0];
    }

    // check -g_j(r_j) + g_{j+1}(0) + g_{j+1}(1) == 0
    for j in 0..(num_vars - 1) {
        let rho = rhos[j + 1];
        let prev_off = j * block;
        let next_off = (j + 1) * block;

        let mut pow = F::from(1u64);
        for deg in 0..block {
            y[prev_off + deg] -= rho * pow;
            pow *= challenges[j];
        }

        y[next_off] += rho * F::from(2u64);
        for deg in 1..block {
            y[next_off + deg] += rho;
        }
    }

    // check g_v(r_v) == g(r_1, ..., r_v)
    let rho_last = rhos[num_vars];
    let last_off = (num_vars - 1) * block;
    let mut pow = F::from(1u64);
    for deg in 0..block {
        y[last_off + deg] += rho_last * pow;
        pow *= challenges[num_vars - 1];
    }

    y
}

/// Produce a non-interactive ZK sumcheck proof.
///
/// This version leans on `CombinedMLE` directly:
/// - `to_univariate(0)` computes each round polynomial,
/// - `partial_eval(&[r_j])` advances to the next round,
/// - `evaluate(&[])` obtains the final fully-restricted value.
#[must_use]
pub fn prove<E: EllipticCurve>(
    params: &PublicParams<E>,
    statement: &Statement<E>,
    witness: &Witness<E>,
    trans: &mut Transcript,
    mut rng: impl rand::Rng,
) -> (Proof<E>, Vec<E::Scalar>, E::Scalar) {
    let n = statement.num_vars;
    let d = statement.max_degree;
    let block = d + 1;

    assert_eq!(params.vec_gens.len(), n * block, "bad vec_gens length");
    assert_eq!(
        witness.polynomial.num_vars(),
        n,
        "polynomial.num_vars != statement.num_vars"
    );

    trans.append_message("zk_sumcheck_protocol", ());
    trans.append_message("zk_sumcheck_params", params);
    trans.append_message("zk_sumcheck_statement", statement);

    let mut current_poly = witness.polynomial.clone();
    let mut round_commitments = Vec::with_capacity(n);
    let mut flattened_coeffs = Vec::with_capacity(n * block);
    let mut challenges = Vec::with_capacity(n);
    let mut r_pi = E::Scalar::zero();

    for round in 0..n {
        let gj = current_poly.to_univariate(0);
        let coeffs = pad_coeffs(gj.coeffs(), block);
        let r_gj = E::Scalar::random(&mut rng);
        let gens = &params.vec_gens[(round * block)..((round + 1) * block)];
        let comm_gj = pedersen::vector_commit(&coeffs, r_gj, gens, params.scalar_gens[1]);

        round_commitments.push(comm_gj);
        flattened_coeffs.extend_from_slice(&coeffs);
        r_pi += r_gj;

        trans.append_message("zk_sumcheck_round_commitment", &comm_gj);
        let r_j: E::Scalar = trans.get_challenge("zk_sumcheck_round_challenge");
        challenges.push(r_j);
        current_poly = current_poly.partial_eval(&[r_j]);
    }

    let final_eval = current_poly.evaluate(&[]);
    let r_final = E::Scalar::random(&mut rng);
    let comm_final = pedersen::commit(final_eval, r_final, &params.scalar_gens);
    trans.append_message("zk_sumcheck_final_commitment", &comm_final);

    let rhos = (0..=n)
        .map(|_| trans.get_challenge("zk_sumcheck_batch_challenge"))
        .collect_vec();
    let y = batch_coeffs(&challenges, &rhos, d);

    let comm_pi: E = round_commitments.iter().copied().sum();
    let comm_sum = CommittedValue::from_parts(witness.sum, witness.r_sum, statement.comm_sum);
    let final_cv = CommittedValue::from_parts(final_eval, r_final, comm_final);
    let cres = comm_sum * rhos[0] + final_cv * rhos[n];

    let dp_params = pedersen::dot_product::PublicParams {
        vec_gens: params.vec_gens.clone(),
        scalar_gens: params.scalar_gens,
    };
    let dp_stmt = pedersen::dot_product::Statement {
        a: y,
        comm_x: comm_pi,
        comm_result: cres.comm,
    };
    let dp_wit = pedersen::dot_product::Witness {
        x: flattened_coeffs,
        r_x: r_pi,
        result: cres.val,
        r_result: cres.r,
    };
    let dp_proof = pedersen::dot_product::prove(&dp_params, &dp_stmt, &dp_wit, trans, &mut rng);

    (
        Proof {
            round_commitments,
            comm_final,
            dp_proof,
        },
        challenges,
        r_final,
    )
}

/// Verify a ZK sumcheck proof.
///
/// The verifier re-derives all Fiat–Shamir challenges from the transcript,
/// reconstructs the batched linear check, and verifies the single dot-product
/// proof that compresses the `v + 1` ordinary sumcheck checks.
pub fn verify<E: EllipticCurve>(
    params: &PublicParams<E>,
    statement: &Statement<E>,
    proof: &Proof<E>,
    trans: &mut Transcript,
) -> Result<(E, Vec<E::Scalar>)> {
    let n = statement.num_vars;
    let d = statement.max_degree;
    let block = d + 1;

    anyhow::ensure!(params.vec_gens.len() == n * block, "bad vec_gens length");
    anyhow::ensure!(
        proof.round_commitments.len() == n,
        "bad number of round commitments"
    );

    trans.append_message("zk_sumcheck_protocol", ());
    trans.append_message("zk_sumcheck_params", params);
    trans.append_message("zk_sumcheck_statement", statement);

    let mut challenges = Vec::with_capacity(n);
    for comm_gj in &proof.round_commitments {
        trans.append_message("zk_sumcheck_round_commitment", comm_gj);
        let r_j: E::Scalar = trans.get_challenge("zk_sumcheck_round_challenge");
        challenges.push(r_j);
    }

    trans.append_message("zk_sumcheck_final_commitment", &proof.comm_final);
    let rhos = (0..=n)
        .map(|_| trans.get_challenge("zk_sumcheck_batch_challenge"))
        .collect_vec();
    let y = batch_coeffs(&challenges, &rhos, d);

    let comm_pi: E = proof.round_commitments.iter().copied().sum();
    let cres = statement.comm_sum * rhos[0] + proof.comm_final * rhos[n];

    let dp_params = pedersen::dot_product::PublicParams {
        vec_gens: params.vec_gens.clone(),
        scalar_gens: params.scalar_gens,
    };
    let dp_stmt = pedersen::dot_product::Statement {
        a: y,
        comm_x: comm_pi,
        comm_result: cres,
    };
    pedersen::dot_product::verify(&dp_params, &dp_stmt, &proof.dp_proof, trans)?;

    Ok((proof.comm_final, challenges))
}
