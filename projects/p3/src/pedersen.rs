#![allow(non_snake_case)]
use crate::transcript::Transcript;
use anyhow::Result;
use itertools::Itertools;
use p1::Random;
use p2::ec::EllipticCurve;
use serde::{Deserialize, Serialize};

/// Pedersen commitment: Com(m, r) = g^m * h^r
#[must_use]
pub fn commit<E: EllipticCurve>(value: E::Scalar, randomness: E::Scalar, generators: &[E; 2]) -> E {
    E::msm(&[value, randomness], generators)
}

/// Vector Pedersen commitment: Com(m₁,...,mₙ; r) = h^r * ∏ᵢ gᵢ^mᵢ
#[must_use]
pub fn vector_commit<E: EllipticCurve>(
    values: &[E::Scalar],
    randomness: E::Scalar,
    generators: &[E],
    h: E,
) -> E {
    assert_eq!(
        values.len(),
        generators.len(),
        "vector_commit: values and generators must have equal length"
    );
    E::msm(values, generators) + h * randomness
}

pub fn inner_product<F: p1::Field>(a: &[F], b: &[F]) -> F {
    assert_eq!(
        a.len(),
        b.len(),
        "inner_product: vectors must have equal length"
    );
    a.iter().zip(b.iter()).map(|(x, y)| *x * *y).sum()
}

/// Sigma protocol for proving knowledge of a Pedersen commitment opening.
///
/// Given `C = g^x * h^r`, prove knowledge of `(x, r)`.
pub mod open {
    use super::*;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    #[serde(bound(deserialize = "E: serde::de::DeserializeOwned"))]
    pub struct PublicParams<E> {
        /// Generators `[g, h]` for the Pedersen commitment scheme.
        pub generators: [E; 2],
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Statement<E> {
        /// The commitment `C` whose opening the prover claims to know.
        pub commitment: E,
    }

    /// The secret opening `(x, r)` such that `C = g^x * h^r`.
    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Witness<E: EllipticCurve> {
        /// The committed value.
        pub x: E::Scalar,
        /// The commitment randomness.
        pub r: E::Scalar,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Proof<E: EllipticCurve> {
        /// The Fiat-Shamir challenge.
        pub c: E::Scalar,
        /// Blinded `x`.
        pub z_x: E::Scalar,
        /// Blinded `r`.
        pub z_r: E::Scalar,
    }

    /// Produce a non-interactive proof of knowledge of an opening `(x, r)` for
    /// the commitment `C = g^x * h^r`.
    #[must_use]
    pub fn prove<E: EllipticCurve>(
        params: &PublicParams<E>,
        statement: &Statement<E>,
        witness: &Witness<E>,
        trans: &mut Transcript,
        mut rng: impl rand::Rng,
    ) -> Proof<E> {
        assert_eq!(statement.commitment, commit(witness.x, witness.r, &params.generators));

        let t_x = E::Scalar::random(&mut rng);
        let t_r = E::Scalar::random(&mut rng);
        let R = commit(t_x, t_r, &params.generators);

        trans.append_message("pedersen_open_protocol", ());
        trans.append_message("pedersen_open_params", params);
        trans.append_message("pedersen_open_statement", statement);
        trans.append_message("pedersen_open_mask", &R);
        let c: E::Scalar = trans.get_challenge("pedersen_open_challenge");

        Proof {
            c,
            z_x: t_x + c * witness.x,
            z_r: t_r + c * witness.r,
        }
    }

    /// Verify a proof of knowledge of a Pedersen commitment opening.
    pub fn verify<E: EllipticCurve>(
        params: &PublicParams<E>,
        statement: &Statement<E>,
        pf: &Proof<E>,
        trans: &mut Transcript,
    ) -> Result<()> {
        let R = commit(pf.z_x, pf.z_r, &params.generators) - statement.commitment * pf.c;

        trans.append_message("pedersen_open_protocol", ());
        trans.append_message("pedersen_open_params", params);
        trans.append_message("pedersen_open_statement", statement);
        trans.append_message("pedersen_open_mask", &R);
        let expected_c: E::Scalar = trans.get_challenge("pedersen_open_challenge");

        anyhow::ensure!(pf.c == expected_c, "invalid Pedersen opening proof");
        Ok(())
    }
}

/// Sigma protocol for proving two Pedersen commitments hide the same value.
///
/// Given `C_1 = g^x * h^r_1` and `C_2 = g^x * h^r_2`, prove knowledge of `(x, r_1, r_2)`.
pub mod equals {
    use super::*;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    #[serde(bound(deserialize = "E: serde::de::DeserializeOwned"))]
    pub struct PublicParams<E> {
        /// Generators `[g, h]`.
        pub generators: [E; 2],
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Statement<E> {
        /// First commitment `C_1 = g^x * h^r_1`.
        pub comm1: E,
        /// Second commitment `C_2 = g^x * h^r_2`.
        pub comm2: E,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Witness<E: EllipticCurve> {
        /// The shared committed value.
        pub x: E::Scalar,
        /// Blinding factor for `comm1`.
        pub r1: E::Scalar,
        /// Blinding factor for `comm2`.
        pub r2: E::Scalar,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Proof<E: EllipticCurve> {
        /// The Fiat-Shamir challenge.
        pub c: E::Scalar,
        /// Blinded `x`.
        pub z_x: E::Scalar,
        /// Blinded `r1`.
        pub z_r1: E::Scalar,
        /// Blinded `r2`.
        pub z_r2: E::Scalar,
    }

    /// Produce a non-interactive proof that two Pedersen commitments `C_1` and
    /// `C_2` hide the same value `x` (possibly with different blinding factors
    /// `r_1`, `r_2`).
    #[must_use]
    pub fn prove<E: EllipticCurve>(
        params: &PublicParams<E>,
        statement: &Statement<E>,
        witness: &Witness<E>,
        trans: &mut Transcript,
        mut rng: impl rand::Rng,
    ) -> Proof<E> {
        assert_eq!(statement.comm1, commit(witness.x, witness.r1, &params.generators));
        assert_eq!(statement.comm2, commit(witness.x, witness.r2, &params.generators));

        let t_x = E::Scalar::random(&mut rng);
        let t_r1 = E::Scalar::random(&mut rng);
        let t_r2 = E::Scalar::random(&mut rng);
        let R1 = commit(t_x, t_r1, &params.generators);
        let R2 = commit(t_x, t_r2, &params.generators);

        trans.append_message("pedersen_equals_protocol", ());
        trans.append_message("pedersen_equals_params", params);
        trans.append_message("pedersen_equals_statement", statement);
        trans.append_message("pedersen_equals_mask_left", &R1);
        trans.append_message("pedersen_equals_mask_right", &R2);
        let c: E::Scalar = trans.get_challenge("pedersen_equals_challenge");

        Proof {
            c,
            z_x: t_x + c * witness.x,
            z_r1: t_r1 + c * witness.r1,
            z_r2: t_r2 + c * witness.r2,
        }
    }

    /// Verify a proof that two commitments hide the same value.
    pub fn verify<E: EllipticCurve>(
        params: &PublicParams<E>,
        statement: &Statement<E>,
        pf: &Proof<E>,
        trans: &mut Transcript,
    ) -> Result<()> {
        let R1 = commit(pf.z_x, pf.z_r1, &params.generators) - statement.comm1 * pf.c;
        let R2 = commit(pf.z_x, pf.z_r2, &params.generators) - statement.comm2 * pf.c;

        trans.append_message("pedersen_equals_protocol", ());
        trans.append_message("pedersen_equals_params", params);
        trans.append_message("pedersen_equals_statement", statement);
        trans.append_message("pedersen_equals_mask_left", &R1);
        trans.append_message("pedersen_equals_mask_right", &R2);
        let expected_c: E::Scalar = trans.get_challenge("pedersen_equals_challenge");

        anyhow::ensure!(pf.c == expected_c, "invalid Pedersen equality proof");
        Ok(())
    }
}

/// Sigma protocol for proving a multiplicative relation between committed values.
///
/// Given `C_x = g^x * h^r_x`, `C_y = g^y * h^r_y`, `C_z = g^z * h^r_z`,
/// prove knowledge of `(x, y, z, r_x, r_y, r_z)` such that `z = x * y`.
pub mod product {
    use super::*;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    #[serde(bound(deserialize = "E: serde::de::DeserializeOwned"))]
    pub struct PublicParams<E> {
        /// Generators `[g, h]`.
        pub generators: [E; 2],
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Statement<E> {
        /// Commitment to `x`.
        pub comm_x: E,
        /// Commitment to `y`.
        pub comm_y: E,
        /// Commitment to `z = x * y`.
        pub comm_z: E,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Witness<E: EllipticCurve> {
        /// First multiplicand.
        pub x: E::Scalar,
        /// Second multiplicand.
        pub y: E::Scalar,
        /// Product `z = x * y`.
        pub z: E::Scalar,
        /// Blinding factor for `comm_x`.
        pub rx: E::Scalar,
        /// Blinding factor for `comm_y`.
        pub ry: E::Scalar,
        /// Blinding factor for `comm_z`.
        pub rz: E::Scalar,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Proof<E: EllipticCurve> {
        /// The Fiat-Shamir challenge.
        pub c: E::Scalar,
        /// Blinded `x`.
        pub z_x: E::Scalar,
        /// Blinded `rx`.
        pub z_rx: E::Scalar,
        /// Blinded `y`.
        pub z_y: E::Scalar,
        /// Blinded `ry`.
        pub z_ry: E::Scalar,
        /// Blinded `rz - rx * y`.
        pub z_prod: E::Scalar,
    }

    /// Produce a non-interactive proof that `z = x * y` for three committed values.
    #[must_use]
    pub fn prove<E: EllipticCurve>(
        params: &PublicParams<E>,
        statement: &Statement<E>,
        witness: &Witness<E>,
        trans: &mut Transcript,
        mut rng: impl rand::Rng,
    ) -> Proof<E> {
        assert_eq!(witness.z, witness.x * witness.y, "witness.z must equal witness.x * witness.y");
        assert_eq!(statement.comm_x, commit(witness.x, witness.rx, &params.generators));
        assert_eq!(statement.comm_y, commit(witness.y, witness.ry, &params.generators));
        assert_eq!(statement.comm_z, commit(witness.z, witness.rz, &params.generators));

        let b1 = E::Scalar::random(&mut rng);
        let b2 = E::Scalar::random(&mut rng);
        let b3 = E::Scalar::random(&mut rng);
        let b4 = E::Scalar::random(&mut rng);
        let b5 = E::Scalar::random(&mut rng);

        let R = commit(b1, b2, &params.generators);
        let beta = commit(b3, b4, &params.generators);
        let gamma = statement.comm_x * b3 + params.generators[1] * b5;

        trans.append_message("pedersen_product_protocol", ());
        trans.append_message("pedersen_product_params", params);
        trans.append_message("pedersen_product_statement", statement);
        trans.append_message("pedersen_product_mask_r", &R);
        trans.append_message("pedersen_product_mask_beta", &beta);
        trans.append_message("pedersen_product_mask_gamma", &gamma);
        let c: E::Scalar = trans.get_challenge("pedersen_product_challenge");

        Proof {
            c,
            z_x: b1 + c * witness.x,
            z_rx: b2 + c * witness.rx,
            z_y: b3 + c * witness.y,
            z_ry: b4 + c * witness.ry,
            z_prod: b5 + c * (witness.rz - witness.rx * witness.y),
        }
    }

    /// Verify a proof that `z = x * y` for three committed values.
    pub fn verify<E: EllipticCurve>(
        params: &PublicParams<E>,
        statement: &Statement<E>,
        pf: &Proof<E>,
        trans: &mut Transcript,
    ) -> Result<()> {
        let R = commit(pf.z_x, pf.z_rx, &params.generators) - statement.comm_x * pf.c;
        let beta = commit(pf.z_y, pf.z_ry, &params.generators) - statement.comm_y * pf.c;
        let gamma = statement.comm_x * pf.z_y + params.generators[1] * pf.z_prod - statement.comm_z * pf.c;

        trans.append_message("pedersen_product_protocol", ());
        trans.append_message("pedersen_product_params", params);
        trans.append_message("pedersen_product_statement", statement);
        trans.append_message("pedersen_product_mask_r", &R);
        trans.append_message("pedersen_product_mask_beta", &beta);
        trans.append_message("pedersen_product_mask_gamma", &gamma);
        let expected_c: E::Scalar = trans.get_challenge("pedersen_product_challenge");

        anyhow::ensure!(pf.c == expected_c, "invalid Pedersen product proof");
        Ok(())
    }
}

/// Sigma protocol for proving a dot product relation between a public vector and a committed vector.
///
/// Given public vector `a`, a vector commitment `xi = VecCom(x; r_xi)`, and a
/// scalar commitment `tau = Com(y; r_tau)`, prove knowledge of `(x, y, r_xi, r_tau)`
/// such that `y = <a, x>`.
pub mod dot_product {
    use super::*;

    #[derive(Clone, Debug, Serialize, Deserialize)]
    #[serde(bound(deserialize = "E: serde::de::DeserializeOwned"))]
    pub struct PublicParams<E> {
        /// Vector generators `[g_1, ..., g_n]` for `VecCom`.
        pub vec_gens: Vec<E>,
        /// Scalar generators `[g, h]` for `Com`.
        pub scalar_gens: [E; 2],
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    #[serde(bound(deserialize = ""))]
    pub struct Statement<E: EllipticCurve> {
        /// The public vector `a`.
        pub a: Vec<E::Scalar>,
        /// Vector commitment to `x`: `VecCom(x; r_x)`.
        pub comm_x: E,
        /// Scalar commitment to `<a, x>`: `Com(y; r_y)`.
        pub comm_result: E,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Witness<E: EllipticCurve> {
        /// The committed vector.
        pub x: Vec<E::Scalar>,
        /// Blinding factor for `comm_x`.
        pub r_x: E::Scalar,
        /// The dot product `<a, x>`.
        pub result: E::Scalar,
        /// Blinding factor for `comm_result`.
        pub r_result: E::Scalar,
    }

    #[derive(Clone, Debug, Serialize, Deserialize)]
    pub struct Proof<E: EllipticCurve> {
        /// The Fiat-Shamir challenge.
        pub c: E::Scalar,
        /// Element-wise blinded `x`.
        pub z_vec: Vec<E::Scalar>,
        /// Blinded vector commitment randomness.
        pub z_delta: E::Scalar,
        /// Blinded scalar commitment randomness.
        pub z_beta: E::Scalar,
    }

    /// Produce a non-interactive proof that `y = <a, x>`, where `a` is public,
    /// `x` is committed via a vector Pedersen commitment, and `y` is committed
    /// via a scalar Pedersen commitment.
    #[must_use]
    pub fn prove<E: EllipticCurve>(
        params: &PublicParams<E>,
        statement: &Statement<E>,
        witness: &Witness<E>,
        trans: &mut Transcript,
        mut rng: impl rand::Rng,
    ) -> Proof<E> {
        assert_eq!(statement.a.len(), params.vec_gens.len(), "a.len() != vec_gens.len()");
        assert_eq!(witness.x.len(), params.vec_gens.len(), "x.len() != vec_gens.len()");
        assert_eq!(statement.a.len(), witness.x.len(), "a.len() != x.len()");
        assert_eq!(witness.result, inner_product(&statement.a, &witness.x));
        assert_eq!(statement.comm_x, vector_commit(&witness.x, witness.r_x, &params.vec_gens, params.scalar_gens[1]));
        assert_eq!(statement.comm_result, commit(witness.result, witness.r_result, &params.scalar_gens));

        let d = (0..witness.x.len())
            .map(|_| E::Scalar::random(&mut rng))
            .collect_vec();
        let o1 = E::Scalar::random(&mut rng);
        let o2 = E::Scalar::random(&mut rng);

        let C1 = vector_commit(&d, o1, &params.vec_gens, params.scalar_gens[1]);
        let C2 = commit(inner_product(&d, &statement.a), o2, &params.scalar_gens);

        trans.append_message("pedersen_dot_product_protocol", ());
        trans.append_message("pedersen_dot_product_params", params);
        trans.append_message("pedersen_dot_product_statement", statement);
        trans.append_message("pedersen_dot_product_mask_vec", &C1);
        trans.append_message("pedersen_dot_product_mask_result", &C2);
        let c: E::Scalar = trans.get_challenge("pedersen_dot_product_challenge");

        Proof {
            c,
            z_vec: d
                .into_iter()
                .zip(&witness.x)
                .map(|(d_i, &x_i)| d_i + c * x_i)
                .collect(),
            z_delta: o1 + c * witness.r_x,
            z_beta: o2 + c * witness.r_result,
        }
    }

    /// Verify a proof that `y = <a, x>` for a committed vector `x` and committed
    /// scalar `y`.
    pub fn verify<E: EllipticCurve>(
        params: &PublicParams<E>,
        statement: &Statement<E>,
        pf: &Proof<E>,
        trans: &mut Transcript,
    ) -> Result<()> {
        anyhow::ensure!(statement.a.len() == params.vec_gens.len(), "a.len() != vec_gens.len()");
        anyhow::ensure!(pf.z_vec.len() == params.vec_gens.len(), "z_vec.len() != vec_gens.len()");
        anyhow::ensure!(statement.a.len() == pf.z_vec.len(), "a.len() != z_vec.len()");

        let C1 = vector_commit(&pf.z_vec, pf.z_delta, &params.vec_gens, params.scalar_gens[1])
            - statement.comm_x * pf.c;
        let z_result = inner_product(&pf.z_vec, &statement.a);
        let C2 = commit(z_result, pf.z_beta, &params.scalar_gens) - statement.comm_result * pf.c;

        trans.append_message("pedersen_dot_product_protocol", ());
        trans.append_message("pedersen_dot_product_params", params);
        trans.append_message("pedersen_dot_product_statement", statement);
        trans.append_message("pedersen_dot_product_mask_vec", &C1);
        trans.append_message("pedersen_dot_product_mask_result", &C2);
        let expected_c: E::Scalar = trans.get_challenge("pedersen_dot_product_challenge");

        anyhow::ensure!(pf.c == expected_c, "invalid Pedersen dot-product proof");
        Ok(())
    }
}

/// A convenience struct that packages together a value, a commitment to it, and the commitment
/// randomness, and implements addition, subtraction, negation and scalar multiplication
/// component-wise
#[derive(Default, Clone, Copy)]
#[must_use]
pub struct CommittedValue<E: EllipticCurve> {
    pub val: E::Scalar,
    pub r: E::Scalar,
    pub comm: E,
}

impl<E: EllipticCurve> CommittedValue<E> {
    /// construct a CommittedValue by committing to a value (r is sampled from rng)
    pub fn new(val: E::Scalar, mut rng: impl rand::Rng, generators: &[E; 2]) -> Self {
        let r = E::Scalar::random(&mut rng);
        Self {
            val,
            r,
            comm: commit(val, r, generators),
        }
    }

    /// Construct a committed value from its individual components
    /// equivalent to CommittedValue {val, r, comm}
    pub fn from_parts(val: E::Scalar, r: E::Scalar, comm: E) -> Self {
        Self { val, r, comm }
    }
}

/// Pedersen commmitments are additively homomorphic...
impl<E: EllipticCurve> std::ops::Add for CommittedValue<E> {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Self {
            val: self.val + rhs.val,
            r: self.r + rhs.r,
            comm: self.comm + rhs.comm,
        }
    }
}

impl<E: EllipticCurve> std::ops::Neg for CommittedValue<E> {
    type Output = Self;
    fn neg(self) -> Self::Output {
        Self {
            val: -self.val,
            r: -self.r,
            comm: -self.comm,
        }
    }
}

impl<E: EllipticCurve> std::ops::Sub for CommittedValue<E> {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self::Output {
        Self {
            val: self.val - rhs.val,
            r: self.r - rhs.r,
            comm: self.comm - rhs.comm,
        }
    }
}

impl<E: EllipticCurve> std::ops::Mul<E::Scalar> for CommittedValue<E> {
    type Output = Self;
    fn mul(self, rhs: E::Scalar) -> Self::Output {
        Self {
            val: self.val * rhs,
            r: self.r * rhs,
            comm: self.comm * rhs,
        }
    }
}
