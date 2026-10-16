import Mathlib

open scoped BigOperators

namespace AugmentedRisk

universe uG uX uY uTheta uOmega

variable
    {G : Type uG}
    {X : Type uX}
    {Y : Type uY}
    {Theta : Type uTheta}
    {Omega : Type uOmega}

/--
The covariance property of a neural network under the parameter action.
-/
def NetworkCovariant
    [Group G]
    [MulAction G X]
    [MulAction G Y]
    (paramAct : G → Theta → Theta)
    (f : Theta → X → Y) : Prop :=
  ∀ (g : G) (theta : Theta) (x : X),
    f (paramAct g theta) x =
      g • f theta (g⁻¹ • x)

/--
Simultaneous invariance of the loss under the group action.
-/
def InvariantLoss
    (G : Type uG)
    [Group G]
    [MulAction G Y]
    (ell : Y → Y → ℝ) : Prop :=
  ∀ (g : G) (a b : Y),
    ell (g • a) (g • b) = ell a b

/--
The augmented loss at `g` equals the nominal loss evaluated at the
inverse-transformed parameters.
-/
theorem augmented_loss_eq_transformed_loss
    [Group G]
    [MulAction G X]
    [MulAction G Y]
    (paramAct : G → Theta → Theta)
    (f : Theta → X → Y)
    (ell : Y → Y → ℝ)
    (hcov : NetworkCovariant paramAct f)
    (hinv : InvariantLoss G ell)
    (g : G)
    (theta : Theta)
    (x : X)
    (y : Y) :
    ell (f theta (g • x)) (g • y) =
      ell (f (paramAct g⁻¹ theta) x) y := by
  calc
    ell (f theta (g • x)) (g • y) =
        ell
          (g⁻¹ • f theta (g • x))
          (g⁻¹ • (g • y)) := by
      symm
      exact hinv g⁻¹ (f theta (g • x)) (g • y)
    _ = ell (g⁻¹ • f theta (g • x)) y := by
      simp
    _ = ell (f (paramAct g⁻¹ theta) x) y := by
      rw [hcov g⁻¹ theta x]
      simp

/--
An abstract expectation operation only needs to respect pointwise
equality for this result.
-/
def ExpectationExtensional
    (expect : (Omega → ℝ) → ℝ) : Prop :=
  ∀ u v : Omega → ℝ,
    (∀ omega : Omega, u omega = v omega) →
      expect u = expect v

/--
The nominal risk.
-/
def nominalRisk
    (expect : (Omega → ℝ) → ℝ)
    (sampleX : Omega → X)
    (sampleY : Omega → Y)
    (f : Theta → X → Y)
    (ell : Y → Y → ℝ)
    (theta : Theta) : ℝ :=
  expect fun omega =>
    ell
      (f theta (sampleX omega))
      (sampleY omega)

/--
The risk obtained by averaging the augmented losses over the group.
-/
noncomputable def augmentedRisk
    (G : Type uG)
    [Group G]
    [Fintype G]
    [MulAction G X]
    [MulAction G Y]
    (expect : (Omega → ℝ) → ℝ)
    (sampleX : Omega → X)
    (sampleY : Omega → Y)
    (f : Theta → X → Y)
    (ell : Y → Y → ℝ)
    (theta : Theta) : ℝ :=
  ((Fintype.card G : ℝ)⁻¹) *
    ∑ g : G,
      expect fun omega =>
        ell
          (f theta (g • sampleX omega))
          (g • sampleY omega)

/--
The augmented risk is the average of the nominal risks of the
group-transformed parameters.
-/
theorem augmented_risk_eq_average_transformed_risk
    [Group G]
    [Fintype G]
    [MulAction G X]
    [MulAction G Y]
    (paramAct : G → Theta → Theta)
    (f : Theta → X → Y)
    (ell : Y → Y → ℝ)
    (expect : (Omega → ℝ) → ℝ)
    (sampleX : Omega → X)
    (sampleY : Omega → Y)
    (hcov : NetworkCovariant paramAct f)
    (hinv : InvariantLoss G ell)
    (hexpect : ExpectationExtensional expect)
    (theta : Theta) :
    augmentedRisk
        G
        expect
        sampleX
        sampleY
        f
        ell
        theta =
      ((Fintype.card G : ℝ)⁻¹) *
        ∑ g : G,
          nominalRisk
            expect
            sampleX
            sampleY
            f
            ell
            (paramAct g theta) := by
  unfold augmentedRisk
  congr 1
  calc
    (∑ g : G,
        expect fun omega =>
          ell
            (f theta (g • sampleX omega))
            (g • sampleY omega)) =
      ∑ g : G,
        nominalRisk
          expect
          sampleX
          sampleY
          f
          ell
          (paramAct g⁻¹ theta) := by
      apply Finset.sum_congr rfl
      intro g _
      unfold nominalRisk
      apply hexpect
      intro omega
      exact
        augmented_loss_eq_transformed_loss
          paramAct
          f
          ell
          hcov
          hinv
          g
          theta
          (sampleX omega)
          (sampleY omega)
    _ =
      ∑ g : G,
        nominalRisk
          expect
          sampleX
          sampleY
          f
          ell
          (paramAct g theta) := by
      let F : G → ℝ :=
        fun g =>
          nominalRisk
            expect
            sampleX
            sampleY
            f
            ell
            (paramAct g theta)
      change (∑ g : G, F g⁻¹) = ∑ g : G, F g
      exact
        Fintype.sum_equiv
          (Equiv.inv G)
          (fun g : G => F g⁻¹)
          F
          (fun g : G => rfl)
end AugmentedRisk
