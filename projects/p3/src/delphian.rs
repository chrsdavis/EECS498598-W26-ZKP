#![allow(non_snake_case)]
use crate::{
    pedersen::{self, CommittedValue},
    quokka, sumcheck,
    transcript::Transcript,
};
use anyhow::Result;
use p1::{Random, Zero, poly::Multilinear};
use p2::{
    combined::CombinedMLE,
    ec::EllipticCurve,
    sparsemat::{SparseMatrix, SparseVector},
};
use serde::{Deserialize, Serialize};

/// Public parameters for the ZK Delphian protocol.
///
/// Holds independent generator sets for each subprotocol, plus the shared
/// blinding generator `h` and scalar-commitment base `g`.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(deserialize = "E: serde::de::DeserializeOwned"))]
pub struct PublicParams<E> {
    /// Vector generators for Quokka row commitments (`sqrt(witness_size)` elements).
    pub quokka_gens: Vec<E>,
    /// Vector generators for the main (degree-3) sumcheck (`log_rows * 4` elements).
    pub phase1_sc_gens: Vec<E>,
    /// Vector generators for the matrix-vector (degree-2) sumchecks (`log_cols * 3` elements).
    pub phase2_sc_gens: Vec<E>,
    /// Scalar generators `[g, h]` shared across subprotocols.
    pub scalar_gens: [E; 2],
}

impl<E: EllipticCurve> PublicParams<E> {
    /// Creates new public parameters with generator count validation.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - `quokka_gens` is empty or its length is not a power of two
    /// - `phase1_sc_gens` length is not a positive multiple of 4 (`max_degree + 1`)
    /// - `phase2_sc_gens` length is not a positive multiple of 3 (`max_degree + 1`)
    pub fn new(
        quokka_gens: Vec<E>,
        phase1_sc_gens: Vec<E>,
        phase2_sc_gens: Vec<E>,
        scalar_gens: [E; 2],
    ) -> Self {
        assert!(!quokka_gens.is_empty(), "quokka_gens must be non-empty");
        assert!(
            quokka_gens.len().is_power_of_two(),
            "quokka_gens.len() must be power of two"
        );
        assert!(
            !phase1_sc_gens.is_empty() && phase1_sc_gens.len() % 4 == 0,
            "phase1_sc_gens.len() must be a positive multiple of 4 (max_degree+1), got {}",
            phase1_sc_gens.len()
        );
        assert!(
            !phase2_sc_gens.is_empty() && phase2_sc_gens.len() % 3 == 0,
            "phase2_sc_gens.len() must be a positive multiple of 3 (max_degree+1), got {}",
            phase2_sc_gens.len()
        );
        Self {
            quokka_gens,
            phase1_sc_gens,
            phase2_sc_gens,
            scalar_gens,
        }
    }

    fn quokka_params(&self) -> quokka::PublicParams<E> {
        pedersen::dot_product::PublicParams {
            vec_gens: self.quokka_gens.clone(),
            scalar_gens: self.scalar_gens,
        }
    }

    fn sc_phase1_params(&self) -> sumcheck::PublicParams<E> {
        sumcheck::PublicParams {
            vec_gens: self.phase1_sc_gens.clone(),
            scalar_gens: self.scalar_gens,
        }
    }

    fn sc_phase2_params(&self) -> sumcheck::PublicParams<E> {
        sumcheck::PublicParams {
            vec_gens: self.phase2_sc_gens.clone(),
            scalar_gens: self.scalar_gens,
        }
    }

    fn sigma_params(&self) -> pedersen::equals::PublicParams<E> {
        pedersen::equals::PublicParams {
            generators: self.scalar_gens,
        }
    }

    fn product_params(&self) -> pedersen::product::PublicParams<E> {
        pedersen::product::PublicParams {
            generators: self.scalar_gens,
        }
    }

    fn open_params(&self) -> pedersen::open::PublicParams<E> {
        pedersen::open::PublicParams {
            generators: self.scalar_gens,
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(deserialize = ""))]
pub struct Statement<E: EllipticCurve> {
    /// R1CS constraint matrix `A`.
    pub A: SparseMatrix<E::Scalar>,
    /// R1CS constraint matrix `B`.
    pub B: SparseMatrix<E::Scalar>,
    /// R1CS constraint matrix `C`.
    pub C: SparseMatrix<E::Scalar>,
    /// Public input vector.
    pub x: SparseVector<E::Scalar>,
}

impl<E: EllipticCurve> Statement<E> {
    /// Creates a new R1CS statement with dimension validation.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The matrices have mismatched dimensions
    /// - Rows or columns are not powers of two
    /// - The public input size is not a power of two
    pub fn new(
        A: SparseMatrix<E::Scalar>,
        B: SparseMatrix<E::Scalar>,
        C: SparseMatrix<E::Scalar>,
        x: SparseVector<E::Scalar>,
    ) -> Self {
        assert_eq!(A.rows, B.rows, "A.rows != B.rows");
        assert_eq!(A.rows, C.rows, "A.rows != C.rows");
        assert_eq!(A.cols, B.cols, "A.cols != B.cols");
        assert_eq!(A.cols, C.cols, "A.cols != C.cols");
        assert!(A.rows.is_power_of_two(), "rows must be power of two");
        assert!(A.cols.is_power_of_two(), "cols must be power of two");
        assert!(x.size.is_power_of_two(), "x.size must be power of two");
        Self { A, B, C, x }
    }
}

/// The prover's private witness.
#[derive(Clone, Debug)]
pub struct Witness<E: EllipticCurve> {
    /// Private witness vector.
    pub w: SparseVector<E::Scalar>,
}

impl<E: EllipticCurve> Witness<E> {
    /// Creates a new witness with validation.
    ///
    /// # Panics
    ///
    /// Panics if the witness size is not a power of two.
    pub fn new(w: SparseVector<E::Scalar>) -> Self {
        assert!(w.size.is_power_of_two(), "w.size must be power of two");
        Self { w }
    }
}

/// Proof elements for a single matrix-vector sumcheck (A, B, or C).
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(deserialize = ""))]
pub struct MatrixProof<E: EllipticCurve> {
    /// Commitment to `v_M = sum_x M_tilde(r', x) * z_tilde(x)`.
    pub comm_v: E,
    /// ZK sumcheck proof for `p_M(x) = M_tilde(r', x) * z_tilde(x)`.
    pub sc_proof: sumcheck::Proof<E>,
    /// Commitment to `w_tilde(r''_partial)`.
    pub comm_w_m: E,
    /// Quokka opening proof for `w_tilde` at `r''_partial`.
    pub quokka_proof: quokka::Proof<E>,
    /// Proves `sc_proof.comm_final` and the homomorphically-derived commitment
    /// to `p_M(r'')` agree on the same underlying value.
    pub equals_proof: pedersen::equals::Proof<E>,
}

/// Full ZK Delphian proof.
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(bound(deserialize = ""))]
pub struct Proof<E: EllipticCurve> {
    /// Quokka commitment to `w_tilde`.
    pub comm_w: quokka::Commitment<E>,
    /// Main sumcheck proof (degree-3, over `log_rows` variables).
    pub sc_phase1_proof: sumcheck::Proof<E>,
    /// Per-matrix proofs for `A`, `B`, `C` (in that order).
    pub matrix_proofs: [MatrixProof<E>; 3],
    /// Commitment to `v_A * v_B`.
    pub comm_v_ab: E,
    /// Proves `v_A * v_B = v_AB` under the committed values.
    pub product_proof: pedersen::product::Proof<E>,
    // Proves that prover has an opening for Comm(v_C),
    pub open_proof: pedersen::open::Proof<E>,
    /// Proves the main sumcheck's final evaluation equals
    /// `eq_tilde(tau, r') * (v_AB - v_C)`.
    pub final_equals_proof: pedersen::equals::Proof<E>,
}

fn matrix_entry_poly<F: p1::Field>(m_eval: &[F]) -> F {
    m_eval[0] * m_eval[1]
}

fn phase1_combiner<F: p1::Field>(evals: &[F]) -> F {
    evals[0] * (evals[1] * evals[2] - evals[3])
}

/// Produce a ZK proof that the witness satisfies the R1CS instance `(A, B, C, x)`.
///
/// The protocol has several phases, each building on the previous:
///
/// **Setup**: Concatenate the public input `x` and private witness `w` into `z`.
/// Compute multilinear extensions `z_tilde`, `w_tilde`, and `x_tilde`. Commit
/// to `w_tilde` using Quokka (generating random blinding factors for the row
/// commitments). Append the commitment and derive the random challenge vector
/// `tau` from the transcript.
///
/// **Phase 1 -- Main sumcheck**: Construct the degree-3 polynomial
/// `h(r') = eq_tilde(tau, r') * (Az_tilde(r') * Bz_tilde(r') - Cz_tilde(r'))`,
/// which sums to zero over the hypercube when the R1CS is satisfied. Run ZK
/// sumcheck on `h` with claimed sum = 0.
///
/// **Phase 2 -- Matrix-vector sumchecks**: For each matrix `M` in `{A, B, C}`:
///   - Define `p_M(x) = M_tilde(r', x) * z_tilde(x)`, a degree-2 polynomial
///     whose hypercube sum equals `v_M`. Commit to `v_M` and run ZK sumcheck.
///   - The sumcheck produces a random point `r''`. Use Quokka to open `w_tilde`
///     at a derived sub-point of `r''` (all but the last coordinate), proving
///     consistency between the committed polynomial and the sumcheck output.
///   - Use an equals proof to show that the sumcheck's final evaluation
///     commitment matches the value derivable from the Quokka opening and
///     the public input.
///
/// **Phase 3 -- Final consistency**: Prove `v_A * v_B = v_AB` via the product
/// sigma protocol. Then prove that the main sumcheck's final evaluation
/// commitment matches `eq_tilde(tau, r') * (v_AB - v_C)` via an equals proof.
#[must_use]
pub fn prove<E: EllipticCurve>(
    params: &PublicParams<E>,
    statement: &Statement<E>,
    witness: &Witness<E>,
    trans: &mut Transcript,
    mut rng: impl rand::Rng,
) -> Proof<E> {
    // Cross-consistency between statement and witness.
    assert_eq!(statement.x.size, witness.w.size, "x.size != w.size");
    assert_eq!(
        statement.x.size + witness.w.size,
        statement.A.cols,
        "|x| + |w| != cols"
    );

    let num_rows = statement.A.rows;
    let num_cols = statement.A.cols;
    let log_rows = num_rows.ilog2() as usize;
    let log_cols = num_cols.ilog2() as usize;

    let z = statement.x.concat(&witness.w);
    let z_tilde = z.multilinear_extension();
    let w_tilde = witness.w.multilinear_extension();
    let x_tilde = statement.x.multilinear_extension();

    let num_rows_w = 1 << (w_tilde.num_vars() / 2);

    // common transcript prefix and witness commitment
    trans.append_message("zk_delphian_protocol", ());
    trans.append_message("zk_delphian_params", params);
    trans.append_message("zk_delphian_statement", statement);

    let quokka_params = params.quokka_params();
    let w_openings = (0..num_rows_w)
        .map(|_| E::Scalar::random(&mut rng))
        .collect::<Vec<_>>();
    let comm_w = quokka::commit(&w_tilde, &w_openings, &quokka_params);
    trans.append_message("zk_delphian_witness_commitment", &comm_w);

    let tau = (0..log_rows)
        .map(|_| trans.get_challenge("zk_delphian_tau"))
        .collect::<Vec<E::Scalar>>();

    // phase 1: main sumcheck on h(X) = eq(tau, X) * (Az(X) * Bz(X) - Cz(X))
    let Az_tilde = statement.A.mul_sparse(&z).multilinear_extension();
    let Bz_tilde = statement.B.mul_sparse(&z).multilinear_extension();
    let Cz_tilde = statement.C.mul_sparse(&z).multilinear_extension();
    let eq_tau = Multilinear::eq_tilde(&tau);

    let h_poly = CombinedMLE::new(
        3,
        phase1_combiner::<E::Scalar>,
        vec![eq_tau, Az_tilde, Bz_tilde, Cz_tilde],
    );

    let zero = E::Scalar::zero();
    let comm_zero = pedersen::commit(zero, zero, &params.scalar_gens);
    let h_stmt = sumcheck::Statement {
        comm_sum: comm_zero,
        num_vars: log_rows,
        max_degree: 3,
    };
    let h_wit = sumcheck::Witness {
        polynomial: h_poly,
        sum: zero,
        r_sum: zero,
    };
    let (sc_phase1_proof, r_prime, r_h_final) =
        sumcheck::prove(&params.sc_phase1_params(), &h_stmt, &h_wit, trans, &mut rng);

    // precompute the 3 phase 2 polynomials and their committed sums
    let mut matrix_polys = Vec::with_capacity(3);
    let mut v_comms = Vec::with_capacity(3);
    for M in [&statement.A, &statement.B, &statement.C] {
        let M_tilde = M.multilinear_extension();
        let M_at_r_prime = M_tilde.partial_eval(&r_prime);
        let p_M = CombinedMLE::new(
            2,
            matrix_entry_poly::<E::Scalar>,
            vec![M_at_r_prime, z_tilde.clone()],
        );
        let v_M = p_M.sum_over_hypercube();
        matrix_polys.push(p_M);
        v_comms.push(CommittedValue::new(v_M, &mut rng, &params.scalar_gens));
    }

    // Transition check between phase 1 and phase 2
    let cv_A = v_comms[0].clone();
    let cv_B = v_comms[1].clone();
    let cv_C = v_comms[2].clone();
    let cv_AB = CommittedValue::new(cv_A.val * cv_B.val, &mut rng, &params.scalar_gens);

    let product_stmt = pedersen::product::Statement {
        comm_x: cv_A.comm,
        comm_y: cv_B.comm,
        comm_z: cv_AB.comm,
    };
    let product_wit = pedersen::product::Witness {
        x: cv_A.val,
        y: cv_B.val,
        z: cv_AB.val,
        rx: cv_A.r,
        ry: cv_B.r,
        rz: cv_AB.r,
    };
    let product_proof = pedersen::product::prove(
        &params.product_params(),
        &product_stmt,
        &product_wit,
        trans,
        &mut rng,
    );

    let open_stmt = pedersen::open::Statement {
        commitment: cv_C.comm,
    };
    let open_wit = pedersen::open::Witness {
        x: cv_C.val,
        r: cv_C.r,
    };
    let open_proof =
        pedersen::open::prove(&params.open_params(), &open_stmt, &open_wit, trans, &mut rng);

    let e = Multilinear::eq_tilde(&tau).evaluate(&r_prime);
    let expected_h = (cv_AB - cv_C) * e;
    let final_eq_stmt = pedersen::equals::Statement {
        comm1: sc_phase1_proof.comm_final,
        comm2: expected_h.comm,
    };
    let final_eq_wit = pedersen::equals::Witness {
        x: expected_h.val,
        r1: r_h_final,
        r2: expected_h.r,
    };
    let final_equals_proof = pedersen::equals::prove(
        &params.sigma_params(),
        &final_eq_stmt,
        &final_eq_wit,
        trans,
        &mut rng,
    );

    // phase 2: mat/vec sumcheck per matrix
    let mut matrix_proofs = Vec::with_capacity(3);
    for ((M, p_M), cv_M) in [&statement.A, &statement.B, &statement.C]
        .into_iter()
        .zip(matrix_polys.into_iter())
        .zip(v_comms.iter())
    {
        let sc_stmt = sumcheck::Statement {
            comm_sum: cv_M.comm,
            num_vars: log_cols,
            max_degree: 2,
        };
        let sc_wit = sumcheck::Witness {
            polynomial: p_M.clone(),
            sum: cv_M.val,
            r_sum: cv_M.r,
        };
        let (sc_proof, r_double, r_p_final) =
            sumcheck::prove(&params.sc_phase2_params(), &sc_stmt, &sc_wit, trans, &mut rng);

        let w_point = r_double[..(log_cols - 1)].to_vec();
        let selector = r_double[log_cols - 1];
        let one = E::Scalar::from(1u64);

        let w_m = w_tilde.evaluate(&w_point);
        let cv_w_m = CommittedValue::new(w_m, &mut rng, &params.scalar_gens);
        let quokka_stmt = quokka::Statement {
            comm_rows: comm_w.clone(),
            point: w_point.clone(),
            comm_eval: cv_w_m.comm,
        };
        let quokka_wit = quokka::Witness {
            poly: w_tilde.clone(),
            openings: w_openings.clone(),
            eval: cv_w_m.val,
            r_eval: cv_w_m.r,
        };
        let quokka_proof =
            quokka::prove(&quokka_params, &quokka_stmt, &quokka_wit, trans, &mut rng);

        let M_tilde = M.multilinear_extension();
        let M_at_r_prime = M_tilde.partial_eval(&r_prime);
        let x_eval = x_tilde.evaluate(&w_point);
        let z_eval = (one - selector) * x_eval + selector * w_m;
        let m_eval = M_at_r_prime.evaluate(&r_double);
        let p_eval = m_eval * z_eval;

        let comm_z = params.scalar_gens[0] * ((one - selector) * x_eval) + cv_w_m.comm * selector;
        let comm_e = comm_z * m_eval;
        let r_comm_e = m_eval * (selector * cv_w_m.r);

        let eq_stmt = pedersen::equals::Statement {
            comm1: sc_proof.comm_final,
            comm2: comm_e,
        };
        let eq_wit = pedersen::equals::Witness {
            x: p_eval,
            r1: r_p_final,
            r2: r_comm_e,
        };
        let equals_proof =
            pedersen::equals::prove(&params.sigma_params(), &eq_stmt, &eq_wit, trans, &mut rng);

        matrix_proofs.push(MatrixProof {
            comm_v: cv_M.comm,
            sc_proof,
            comm_w_m: cv_w_m.comm,
            quokka_proof,
            equals_proof,
        });
    }

    let matrix_proofs: [MatrixProof<E>; 3] = matrix_proofs
        .try_into()
        .expect("expected exactly three matrix proofs");

    Proof {
        comm_w,
        sc_phase1_proof,
        matrix_proofs,
        comm_v_ab: cv_AB.comm,
        product_proof,
        open_proof,
        final_equals_proof,
    }
}

/// Verify a ZK Delphian proof.
///
/// The verifier mirrors the prover's transcript interactions without access
/// to the witness. It re-derives all challenges, runs the sumcheck and
/// sigma-protocol verifiers, and checks consistency at each stage:
///
/// 1. Derive `tau` from the transcript (after absorbing params, statement,
///    `comm_w`).
/// 2. Verify the main sumcheck (claimed sum = 0, degree 3, `log_rows`
///    variables).
/// 3. For each matrix `M` in `{A, B, C}`:
///    - Verify the matrix-vector sumcheck.
///    - Verify the Quokka opening of `w_tilde` at the derived point.
///    - Derive the expected final-evaluation commitment homomorphically from
///      the public matrix value `M_tilde(r', r'')`, the public input
///      `x_tilde(r'')`, and the Quokka-verified commitment to
///      `w_tilde(r''_partial)`. Verify it equals the sumcheck's final
///      commitment via the equals proof.
/// 4. Verify the product proof (`v_A * v_B = v_AB`).
/// 5. Verify the final equals proof linking the main sumcheck's output to
///    `eq_tilde(tau, r') * (v_AB - v_C)`.
pub fn verify<E: EllipticCurve>(
    params: &PublicParams<E>,
    statement: &Statement<E>,
    proof: Proof<E>,
    trans: &mut Transcript,
) -> Result<()> {
    let num_rows = statement.A.rows;
    let num_cols = statement.A.cols;
    let log_rows = num_rows.ilog2() as usize;
    let log_cols = num_cols.ilog2() as usize;

    let comm_w = proof.comm_w;

    anyhow::ensure!(statement.x.size.is_power_of_two(), "x.size must be power of two");
    anyhow::ensure!(num_rows.is_power_of_two(), "rows must be power of two");
    anyhow::ensure!(num_cols.is_power_of_two(), "cols must be power of two");

    trans.append_message("zk_delphian_protocol", ());
    trans.append_message("zk_delphian_params", params);
    trans.append_message("zk_delphian_statement", statement);
    trans.append_message("zk_delphian_witness_commitment", &comm_w);

    let tau = (0..log_rows)
        .map(|_| trans.get_challenge("zk_delphian_tau"))
        .collect::<Vec<E::Scalar>>();

    let zero = E::Scalar::zero();
    let comm_zero = pedersen::commit(zero, zero, &params.scalar_gens);
    let h_stmt = sumcheck::Statement {
        comm_sum: comm_zero,
        num_vars: log_rows,
        max_degree: 3,
    };
    let (comm_h_rprime, r_prime) =
        sumcheck::verify(&params.sc_phase1_params(), &h_stmt, &proof.sc_phase1_proof, trans)?;

    // Transition check between phases 1 and 2
    let product_stmt = pedersen::product::Statement {
        comm_x: proof.matrix_proofs[0].comm_v,
        comm_y: proof.matrix_proofs[1].comm_v,
        comm_z: proof.comm_v_ab,
    };
    pedersen::product::verify(
        &params.product_params(),
        &product_stmt,
        &proof.product_proof,
        trans,
    )?;

    let open_stmt = pedersen::open::Statement {
        commitment: proof.matrix_proofs[2].comm_v,
    };
    pedersen::open::verify(&params.open_params(), &open_stmt, &proof.open_proof, trans)?;

    let e = Multilinear::eq_tilde(&tau).evaluate(&r_prime);
    let final_eq_stmt = pedersen::equals::Statement {
        comm1: comm_h_rprime,
        comm2: (proof.comm_v_ab - proof.matrix_proofs[2].comm_v) * e,
    };
    pedersen::equals::verify(
        &params.sigma_params(),
        &final_eq_stmt,
        &proof.final_equals_proof,
        trans,
    )?;

    // Complete each phase 2 proof per matrix
    let x_tilde = statement.x.multilinear_extension();
    for (M, matrix_pf) in [&statement.A, &statement.B, &statement.C]
        .into_iter()
        .zip(&proof.matrix_proofs)
    {
        let sc_stmt = sumcheck::Statement {
            comm_sum: matrix_pf.comm_v,
            num_vars: log_cols,
            max_degree: 2,
        };
        let (comm_p_eval, r_double) =
            sumcheck::verify(&params.sc_phase2_params(), &sc_stmt, &matrix_pf.sc_proof, trans)?;

        let w_point = r_double[..(log_cols - 1)].to_vec();
        let selector = r_double[log_cols - 1];
        let quokka_stmt = quokka::Statement {
            comm_rows: comm_w.clone(),
            point: w_point.clone(),
            comm_eval: matrix_pf.comm_w_m,
        };
        quokka::verify(&params.quokka_params(), &quokka_stmt, &matrix_pf.quokka_proof, trans)?;

        let M_tilde = M.multilinear_extension();
        let M_at_r_prime = M_tilde.partial_eval(&r_prime);
        let m_eval = M_at_r_prime.evaluate(&r_double);

        let one = E::Scalar::from(1u64);
        let x_eval = x_tilde.evaluate(&w_point);
        let comm_z =
            params.scalar_gens[0] * ((one - selector) * x_eval) + matrix_pf.comm_w_m * selector;
        let comm_e = comm_z * m_eval;

        let eq_stmt = pedersen::equals::Statement {
            comm1: comm_p_eval,
            comm2: comm_e,
        };
        pedersen::equals::verify(
            &params.sigma_params(),
            &eq_stmt,
            &matrix_pf.equals_proof,
            trans,
        )?;
    }

    Ok(())
}
