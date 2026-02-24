//! # Delphian Protocol
//!
//! This module implements the **Delphian SNARK**, an interactive proof system for
//! **R1CS (Rank-1 Constraint System)** satisfiability. R1CS is a standard representation
//! for NP statements used in many SNARK constructions.
//!
//! ## R1CS Background
//!
//! An R1CS instance consists of three sparse matrices `A`, `B`, `C` over a field `F`,
//! each of dimension `m × n`. A witness vector `w` satisfies the R1CS if there exists
//! a combined vector `z = (x || w)` (public input concatenated with private witness) such that:
//!
//! ```text
//! (A · z) ∘ (B · z) = C · z
//! ```
//!
//! where `∘` denotes the Hadamard (element-wise) product. This means for each row `i`:
//!
//! ```text
//! (A_i · z) × (B_i · z) = C_i · z
//! ```
//!
//! ## Protocol Overview
//!
//! The Delphian protocol reduces R1CS verification to polynomial evaluations using:
//!
//! 1. **Multilinear Extensions (MLEs)**: Convert the discrete R1CS check into polynomial
//!    identities over the field. The matrices `A`, `B`, `C` and vector `z` are lifted to
//!    their multilinear extensions `Ã`, `B̃`, `C̃`, `z̃`.
//!
//! 2. **Sumcheck Protocol**: The R1CS constraint `(Az) ∘ (Bz) = Cz` is equivalent to:
//!    ```text
//!    Σ_{τ ∈ {0,1}^log(m)} eq̃(r, τ) · [(Ãz̃)(τ) · (B̃z̃)(τ) - (C̃z̃)(τ)] = 0
//!    ```
//!    for a random `r`. This sum is verified using the sumcheck protocol.
//!
//! 3. **Polynomial Commitments**: The prover commits to `w̃` (the witness MLE) and later
//!    opens it at random points chosen by the verifier. This uses the Quokka commitment scheme.
#![allow(non_snake_case)]
use crate::{
    combined::CombinedMLE,
    ec::{EllipticCurve, ScalarOf},
    ip::{self, Comms, InteractiveProof},
    quokka,
    sparsemat::{SparseMatrix, SparseVector},
    sumcheck,
};
use anyhow::bail;
use p1::{One, Random, Zero, poly::Multilinear, poly::Univariate};
use std::marker::PhantomData;

/// Messages sent from the prover to the verifier in the Delphian protocol.
///
/// The prover sends two types of messages:
/// - **Polynomial commitments**: Used to commit to the witness polynomial `w̃` at the start.
/// - **Field values**: Used to send claimed evaluations (e.g., `v_A`, `v_B`, `v_C`) that
///   the verifier will later check via sumcheck and polynomial openings.
#[derive(Clone, Debug)]
pub enum ProverMessage<E: EllipticCurve> {
    /// A commitment to a multilinear polynomial using the Quokka scheme.
    PolyComm(quokka::Commitment<E>),
    /// A field element, typically a claimed polynomial evaluation.
    Value(E::Scalar),
}

/// Messages sent from the verifier to the prover.
///
/// The verifier sends vectors of random field elements as challenges. The primary use is
/// sending the random point `τ = (τ_1, ..., τ_log(m))` that determines which "row" of the
/// R1CS constraint is being checked (in a randomized, aggregated sense).
pub type VerifierMessage<E> = Vec<ScalarOf<E>>;

/// The public statement for the Delphian R1CS protocol.
///
/// This contains the R1CS instance (matrices `A`, `B`, `C`) and the public input `x`.
/// Both the prover and verifier have access to this information.
///
/// # R1CS Structure
///
/// The matrices define constraints of the form `(A·z) ∘ (B·z) = C·z` where:
/// - `z = (x || w)` is the concatenation of public input and private witness
/// - Each row represents one multiplicative constraint
/// - Columns correspond to variables in `z`
///
/// # Dimension Requirements
///
/// - All matrices must have the same dimensions: `m × n` where `m` is the number of
///   constraints and `n` is the number of variables.
/// - `m` and `n` must be powers of two (for MLE compatibility).
/// - `|x| + |w| = n` (public input size + witness size = number of columns).
#[derive(Clone, Debug)]
pub struct Statement<E: EllipticCurve> {
    /// The "left input" matrix. `A·z` gives the left operand of each multiplication gate.
    pub A: SparseMatrix<E::Scalar>,
    /// The "right input" matrix. `B·z` gives the right operand of each multiplication gate.
    pub B: SparseMatrix<E::Scalar>,
    /// The "output" matrix. `C·z` gives the expected result of each multiplication gate.
    pub C: SparseMatrix<E::Scalar>,
    /// The public input vector. This is known to both prover and verifier.
    pub x: SparseVector<E::Scalar>,
}

impl<E: EllipticCurve> Statement<E> {
    /// Creates a new R1CS statement with validation.
    ///
    /// # Panics
    ///
    /// Panics if:
    /// - The matrices have mismatched dimensions
    /// - The public input size is not a power of two
    pub fn new(
        A: SparseMatrix<E::Scalar>,
        B: SparseMatrix<E::Scalar>,
        C: SparseMatrix<E::Scalar>,
        x: SparseVector<E::Scalar>,
    ) -> Self {
        // Check the A/B/C dimensions match up.
        assert_eq!(A.rows, B.rows);
        assert_eq!(A.rows, C.rows);
        assert_eq!(A.cols, B.cols);
        assert_eq!(A.cols, C.cols);
        assert!(x.size.is_power_of_two());

        Self { A, B, C, x }
    }

    /// Combines the public input `x` with the private witness `w` to form `z = (x || w)`.
    ///
    /// This is the full variable assignment vector used in the R1CS equations.
    /// The first `|x|` entries are the public input, and the remaining `|w|` entries
    /// are the private witness.
    ///
    /// # Panics
    ///
    /// Panics if `|x| + |w| ≠ number of columns` in the matrices.
    fn z(&self, wit: &Witness<E>) -> SparseVector<E::Scalar> {
        assert_eq!(self.x.size, wit.w.size);
        assert_eq!(self.x.size + wit.w.size, self.A.cols);

        SparseVector::from_entries(
            self.x.size + wit.w.size,
            self.x
                .iter()
                .map(|(i, v)| (i, *v))
                .chain(wit.w.iter().map(|(i, v)| (i + self.x.size, *v))),
        )
    }
}

/// The private witness for the Delphian R1CS protocol.
///
/// This contains the private portion of the variable assignment `z = (x || w)`.
/// Only the prover has access to the witness; the verifier never sees `w` directly
/// (though they can verify properties about it through the protocol).
///
/// # Structure
///
/// The witness `w` is a sparse vector of field elements. Combined with the public
/// input `x` from the statement, it forms the complete variable assignment `z`
/// that must satisfy the R1CS constraints.
#[derive(Clone, Debug)]
pub struct Witness<E: EllipticCurve> {
    /// The private witness vector. This is concatenated with the public input `x`
    /// to form the full assignment `z = (x || w)`.
    w: SparseVector<E::Scalar>,
}

impl<E: EllipticCurve> Witness<E> {
    /// Creates a new witness from a sparse vector.
    ///
    /// # Panics
    ///
    /// Panics if the witness size is not a power of two.
    pub fn new(w: SparseVector<E::Scalar>) -> Self {
        assert!(w.size.is_power_of_two());
        Self { w }
    }
}

/// Marker struct for the Delphian R1CS protocol.
///
/// This implements [`InteractiveProof`] and serves as a namespace for the
/// prover and verifier algorithms. The protocol is parameterized by an
/// elliptic curve `E` which determines the field and commitment scheme used.
pub struct Protocol<E>(PhantomData<E>);

/// Implementation of the Delphian R1CS interactive proof.
///
/// This protocol allows a prover to convince a verifier that they know a witness `w`
/// such that `z = (x || w)` satisfies the R1CS constraints `(A·z) ∘ (B·z) = C·z`.
///
/// # Protocol Flow
///
/// 1. **Commitment Phase**: Prover commits to the witness MLE `w̃`.
///
/// 2. **Main Sumcheck**: Verifier sends random `τ`, prover and verifier run sumcheck on
///    `h̃(X) = eq̃(τ, X) · [(Ãz̃)(X) · (B̃z̃)(X) - (C̃z̃)(X)]` to verify the R1CS constraint
///    holds at all points (in aggregate).
///
/// 3. **Matrix-Vector Sumchecks**: For each matrix `M ∈ {A, B, C}`:
///    - Prover claims `v_M = (M̃·z̃)(r')` where `r'` is from the main sumcheck.
///    - Run sumcheck on `p_M(X) = M̃(r', X) · z̃(X)` to verify the matrix-vector product.
///    - Prover opens `w̃` at the sumcheck's random point via Quokka.
///
/// 4. **Final Check**: Verifier confirms `h̃(r') = eq̃(τ, r') · (v_A · v_B - v_C)`.
impl<E: EllipticCurve> InteractiveProof for Protocol<E> {
    type ProverMessage = ProverMessage<E>;
    type VerifierMessage = VerifierMessage<E>;
    type Statement = Statement<E>;
    type Witness = Witness<E>;

    /// The prover produces no additional output beyond completing the protocol.
    ///
    /// All necessary information (the validity of the R1CS) is conveyed through
    /// the interactive messages themselves.
    type ProverOutput = ();

    /// The verifier produces no additional output beyond accepting/rejecting.
    ///
    /// If the protocol completes without error, the verifier is convinced that
    /// the prover knows a valid witness. Errors indicate rejection.
    type VerifierOutput = ();

    /// The Delphian prover algorithm.
    ///
    /// # Algorithm Overview
    ///
    /// 1. Compute the full assignment `z = (x || w)` and its MLE `z̃`.
    /// 2. Commit to the witness MLE `w̃` using Quokka and send the commitment.
    /// 3. Receive random challenge `τ` from the verifier.
    /// 4. Construct the combined polynomial `h̃ = eq̃(τ, ·) · [(Ãz̃) · (B̃z̃) - (C̃z̃)]`.
    /// 5. Run sumcheck on `h̃` with claimed sum 0 (the R1CS constraint).
    /// 6. For each matrix M ∈ {A, B, C}:
    ///    - Compute `v_M = Σ M̃(r', X) · z̃(X)` over the hypercube.
    ///    - Send `v_M` and run sumcheck to prove the matrix-vector product.
    ///    - Open the witness commitment at the resulting random point.
    ///
    /// # Arguments
    ///
    /// * `stmt` - The R1CS statement (matrices A, B, C and public input x).
    /// * `wit` - The private witness w.
    /// * `comms` - The communication channel.
    async fn prover(
        stmt: Self::Statement,
        wit: Self::Witness,
        mut comms: Comms<Self::ProverMessage, Self::VerifierMessage>,
    ) -> ip::Result<()> {
        let sub_comms = comms.establish_subprotocol::<String, i32>("").await?;
        let value = comms.recv().await?;

        let m = stmt.A.rows;
        let n = stmt.A.cols;
    
        // TODO: add asserts to make sure input is valid

        // z has `col_vars`` vars, and w has `(col_vars-1)`` vars
        let row_vars = m.ilog2() as usize;
        let col_vars = n.ilog2() as usize;

        let w_vars = col_vars - 1;

        // Compute z and is MLE
        let z = stmt.z(&wit);
        let z_tilde: Multilinear<E::Scalar> = z.multilinear_extension();

        // Commit to witness MLE w_tilde with quokka and send
        let w_tilde: Multilinear<E::Scalar> = wit.w.multilinear_extension();
        let (w_comm, w_opening) = quokka::commit::<E>(&w_tilde);
        comms.send(ProverMessage::PolyComm(w_comm.clone()))?;

        // Get random challenge tau from verifier
        let tau = comms.recv().await?;
        if tau.len() != row_vars {
            // TODO: bail
        }

        // Main sumcheck witness h~(X) = eq~(tau,X) * (Az~(X)*Bz~(X) - Cz~(X))
        // TODO: this is kind of clunky
        let Az_dense = stmt.A.mul_sparse(&z).to_dense();
        let Bz_dense = stmt.B.mul_sparse(&z).to_dense();
        let Cz_dense = stmt.C.mul_sparse(&z).to_dense();

        let eq_tilde_tau = Multilinear::<E::Scalar>::eq_tilde(&tau);
        let Az_tilde = Multilinear::new(row_vars, Az_dense);
        let Bz_tilde = Multilinear::new(row_vars, Bz_dense);
        let Cz_tilde = Multilinear::new(row_vars, Cz_dense);

        // Note that max degree is 3 because of the eq*Az*Bz term
        let max_h_degree = 3;
        let combiner = |vals: &[E::Scalar]| vals[0] * (vals[1] * vals[2] - vals[3]);
        let h = CombinedMLE::new(
            max_h_degree, combiner, vec![eq_tilde_tau, Az_tilde, Bz_tilde, Cz_tilde],
        );

        // Run sumcheck as subprotocol
        let sumcheck_comms = comms
            .establish_subprotocol::<Univariate<E::Scalar>, E::Scalar>("main_sumcheck")
            .await?;
        let sumcheck_stmt = sumcheck::Statement {
            claimed_sum: E::Scalar::zero(),
            num_vars: row_vars,
            max_degree: max_h_degree,
        };

        let r_row = sumcheck::Protocol::<E::Scalar>::prover(sumcheck_stmt, h, sumcheck_comms).await?;
        if r_row.len() != row_vars {
            // TODO: bail
        }

        // Public matrix MLEs
        let A_tilde = stmt.A.multilinear_extension();
        let B_tilde = stmt.B.multilinear_extension();
        let C_tilde = stmt.C.multilinear_extension();

        // Process each matrix
        let cases: [(&'static str, &Multilinear<E::Scalar>, &'static str, &'static str); 3] = [
            ("A", &A_tilde, "mv_sumcheck_A", "open_w_A"),
            ("B", &B_tilde, "mv_sumcheck_B", "open_w_B"),
            ("C", &C_tilde, "mv_sumcheck_C", "open_w_C"),
        ];

        for (label, M_tilde, mv_sumcheck_name, open_name) in cases {
            // M_row(X) = M~(r_row, X)
            let M_row = M_tilde.partial_eval(&r_row);

            // p_M(X) = M_row(X) * z~(X)
            let p = CombinedMLE::new(
                2,
                |v: &[E::Scalar]| v[0] * v[1],
                vec![M_row, z_tilde.clone()],
            );

            // Compute and send claimed sum v_M
            let v_M = p.sum_over_hypercube();
            comms.send(ProverMessage::Value(v_M))?;

            // Run sumcheck for p_M
            let mv_sumcheck_comms = comms
                .establish_subprotocol::<Univariate<E::Scalar>, E::Scalar>(mv_sumcheck_name)
                .await?;
            let mv_sumcheck_stmt = sumcheck::Statement {
                claimed_sum: v_M,
                num_vars: col_vars,
                max_degree: 2,
            };

            let r_col = sumcheck::Protocol::<E::Scalar>::prover(mv_sumcheck_stmt, p, mv_sumcheck_comms).await?;
            if r_col.len() != col_vars {
                // TODO: bail
            }

            // Open witness w_tilde for low coords of r_col
            let r_low = r_col[..w_vars].to_vec();
            let w_eval = w_tilde.evaluate(&r_low);

            // Send claimed evaluation (so verifier can make Quokkka.stmt)
            comms.send(ProverMessage::Value(w_eval))?;

            // Quokka.open
            let open_comms = comms
                .establish_subprotocol::<quokka::ProverMessage<E>, quokka::VerifierMessage>(open_name)
                .await?;
            let open_stmt = quokka::Statement::<E> {
                comm: w_comm.clone(),
                point: r_low,
                value: w_eval,
            };
            let open_wit = quokka::Witness::<E> {
                poly: w_tilde.clone(),
                _opening: w_opening,
            };
            quokka::OpenProtocol::<E>::prover(open_stmt, open_wit, open_comms).await?;
        }

        Ok(())
    }

    /// The Delphian verifier algorithm.
    ///
    /// # Algorithm Overview
    ///
    /// 1. Receive the prover's commitment to `w̃`.
    /// 2. Sample random challenge `τ` and send to prover.
    /// 3. Run sumcheck verification for the main R1CS polynomial `h̃` (claimed sum = 0).
    /// 4. For each matrix M ∈ {A, B, C}:
    ///    - Receive claimed value `v_M` from prover.
    ///    - Run sumcheck verification for `p_M(X) = M̃(r', X) · z̃(X)`.
    ///    - Verify the Quokka opening of `w̃` at the random point.
    ///    - Reconstruct `z̃(r'')` from the opened `w̃` value and the public input.
    ///    - Check that `p_M(r'') = M̃(r', r'') · z̃(r'')`.
    /// 5. Final check: verify `h̃(r') = eq̃(τ, r') · (v_A · v_B - v_C)`.
    ///
    /// # Arguments
    ///
    /// * `stmt` - The R1CS statement (matrices A, B, C and public input x).
    /// * `comms` - The communication channel.
    /// * `rng` - Random number generator for sampling challenges.
    ///
    /// # Errors
    ///
    /// Returns an error (rejection) if:
    /// - Any message has an unexpected type
    /// - Any sumcheck verification fails
    /// - Any polynomial commitment opening fails
    /// - The matrix-vector product check fails: `p_M(r'') ≠ M̃(r', r'') · z̃(r'')`
    /// - The final consistency check fails: `h̃(r') ≠ eq̃(τ, r') · (v_A · v_B - v_C)`
    async fn verifier<R: rand::Rng>(
        stmt: Self::Statement,
        mut comms: Comms<Self::VerifierMessage, Self::ProverMessage>,
        rng: &mut R,
    ) -> ip::Result<()> {
        let m = stmt.A.rows;
        let n = stmt.A.cols;

        // TODO: bounds checking and bailing

        let row_vars = m.ilog2() as usize;
        let col_vars = n.ilog2() as usize;
        if col_vars == 0 {
            // TODO: bail
        }
        let w_vars = col_vars - 1;
        if w_vars % 2 != 0 {
           // TODO: bail
        }

        // Receive witness commitment
        let w_comm = match comms.recv().await? {
            ProverMessage::PolyComm(c) => c,
            other => bail!("Delphian: expected ProverMessage::PolyComm, got {:?}", other),
        };

        // Sample and send challenge tau
        let mut tau = Vec::with_capacity(row_vars);
        for _ in 0..row_vars {
            tau.push(E::Scalar::random(rng));
        }
        comms.send(tau.clone())?;

        // Main sumcheck verifier for R1CS poly
        let sumcheck_comms = comms
            .establish_subprotocol::<E::Scalar, Univariate<E::Scalar>>("main_sumcheck")
            .await?;
        let sumcheck_stmt = sumcheck::Statement {
            claimed_sum: E::Scalar::zero(),
            num_vars: row_vars,
            max_degree: 3,
        };
        let (h_eval, r_row) =
            sumcheck::Protocol::<E::Scalar>::verifier(sumcheck_stmt, sumcheck_comms, rng).await?;
        if r_row.len() != row_vars {
            // TODO: bail
        }

        // misc. pub objects for checks
        let A_tilde = stmt.A.multilinear_extension();
        let B_tilde = stmt.B.multilinear_extension();
        let C_tilde = stmt.C.multilinear_extension();

        let x_tilde = stmt.x.multilinear_extension();
        if x_tilde.num_vars() != w_vars {
            // TODO: bail
        }

        // Sumchecks and opens for each matrix
        let mut v = [E::Scalar::zero(); 3];

        let cases: [(&'static str, &Multilinear<E::Scalar>, &'static str, &'static str, usize); 3] = [
            ("A", &A_tilde, "mv_sumcheck_A", "open_w_A", 0),
            ("B", &B_tilde, "mv_sumcheck_B", "open_w_B", 1),
            ("C", &C_tilde, "mv_sumcheck_C", "open_w_C", 2),
        ];

        for (label, M_tilde, mv_sumcheck_name, open_name, idx) in cases {
            // receive v_M from prover
            let v_M = match comms.recv().await? {
                ProverMessage::Value(x) => x,
                other => bail!("Delphian: expected Value(v_M) for {}, got {:?}", label, other),
            };
            v[idx] = v_M;

            // mv sumcheck verifier
            let mv_sumcheck_comms = comms
                .establish_subprotocol::<E::Scalar, Univariate<E::Scalar>>(mv_sumcheck_name)
                .await?;
            let mv_sumcheck_stmt = sumcheck::Statement {
                claimed_sum: v_M,
                num_vars: col_vars,
                max_degree: 2,
            };
            let (p_eval, r_col) =
                sumcheck::Protocol::<E::Scalar>::verifier(mv_sumcheck_stmt, mv_sumcheck_comms, rng).await?;
            if r_col.len() != col_vars {
                // TODO: bail
            }

            // Receive w_eval to form Quokka stmt
            let w_eval = match comms.recv().await? {
                ProverMessage::Value(x) => x,
                other => bail!("Delphian: expected ProverMessage::Value(w_eval) for {}, got {:?}", label, other),
            };

            // Verify Quokka.open(w_comm, r_low, w_eval)
            let r_low = r_col[..w_vars].to_vec();
            let open_comms = comms
                .establish_subprotocol::<quokka::VerifierMessage, quokka::ProverMessage<E>>(open_name)
                .await?;
            let open_stmt = quokka::Statement::<E> {
                comm: w_comm.clone(),
                point: r_low.clone(),
                value: w_eval,
            };
            quokka::OpenProtocol::<E>::verifier(open_stmt, open_comms, rng).await?;

            // Reconstruct z~(r_col) = (1-t)*x~(r_low) + t*w~(r_low)
            // t is the last coordinate
            let t = r_col[w_vars];
            let x_eval = x_tilde.evaluate(&r_low);
            let z_eval = (E::Scalar::one() - t) * x_eval + t * w_eval;

            // compute M~(r_row, r_col)
            let mut pt = Vec::with_capacity(r_row.len() + r_col.len());
            pt.extend_from_slice(&r_row);
            pt.extend_from_slice(&r_col);
            let M_val = M_tilde.evaluate(&pt);

            // point check: p_eval == M_val * z_eval
            if p_eval != M_val * z_eval {
                bail!(
                    "Delphian: {} mv point check failed (p_eval != M(r_row,r_col)*z(r_col), i.e., `p_M(r'') ≠ M̃(r', r'') · z̃(r'')`)",
                    label
                );
            }
        }

        // Final consistency check: h~(r_row) == eq~(tau, r_row) * (vA*vB - vC)
        let eq_eval = Multilinear::<E::Scalar>::eq_tilde(&tau).evaluate(&r_row);
        let rhs = eq_eval * (v[0] * v[1] - v[2]); // vA*vB - vC
        if h_eval != rhs {
            bail!("Delphian: final check failed; `h̃(r') ≠ eq̃(τ, r') · (v_A · v_B - v_C)`");
        }

        Ok(())
    }
}
