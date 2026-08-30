import Mathlib.RepresentationTheory.Basic
import Mathlib.MeasureTheory.Integral.Bochner.Basic
import Mathlib.Analysis.Calculus.Gradient.Basic
import Mathlib.Analysis.InnerProductSpace.LinearMap
import Mathlib.Analysis.InnerProductSpace.PiL2
import Mathlib.Analysis.InnerProductSpace.ProdL2
import Mathlib.Analysis.Calculus.FDeriv.Comp
import Mathlib.Analysis.Calculus.Deriv.Comp
import Mathlib.Analysis.Calculus.FDeriv.Equiv
import Mathlib.Dynamics.Ergodic.MeasurePreserving
import Mathlib.MeasureTheory.Integral.Bochner.ContinuousLinearMap
import Mathlib.LinearAlgebra.AffineSpace.AffineSubspace.Basic

/-!
# Fixed neural-network architectures with Hilbert parameter spaces

A network architecture is fixed independently of its parameter value.

The parameter space `P` is a real Hilbert space, and `G` acts on `P`
by surjective linear isometries. Thus, the parameter action is
orthogonal, which is the real analogue of a unitary representation.

The central architecture property is

    eval θ (ρX(g)x) = ρY(g) (eval (T(g⁻¹)θ) x).

This is Lemma 2.1 in a form suitable for gradients and gradient flow.
-/

universe uG uP uV

namespace NeuralNetworkEquivariance

variable {G : Type uG} [Group G]

/--
A finite-dimensional real Hilbert space.
-/
structure FDHilbert where
  carrier : Type uV
  instNormedAddCommGroup : NormedAddCommGroup carrier
  instInnerProductSpace : InnerProductSpace ℝ carrier
  instFiniteDimensional : FiniteDimensional ℝ carrier

attribute [instance]
  FDHilbert.instNormedAddCommGroup
  FDHilbert.instInnerProductSpace
  FDHilbert.instFiniteDimensional

/--
Coerce a bundled finite-dimensional Hilbert space to its underlying
type.
-/
instance : CoeSort (FDHilbert.{uV}) (Type uV) where
  coe V := V.carrier

/--
Every finite-dimensional real inner-product space is complete.
-/
instance (V : FDHilbert.{uV}) : CompleteSpace V :=
  FiniteDimensional.complete ℝ V


/-!
## Unitary parameter representations
-/

/--
A unitary representation on a real Hilbert parameter space.

Over `ℝ`, these are technically orthogonal representations. Each group
element acts by a bijective linear isometry.
-/
structure UnitaryRepresentation
    (G : Type uG)
    [Group G]
    (V : FDHilbert) where
  act : G → V ≃ₗᵢ[ℝ] V

  act_one :
    ∀ x : V,
      act 1 x = x

  act_mul :
    ∀ (g h : G) (x : V),
      act (g * h) x =
        act g (act h x)

def IsEquivariant
    {G : Type uG}
    [Group G]
    {V : FDHilbert}
    (ρV : UnitaryRepresentation G V)
    (σ : V → V) : Prop :=
  ∀ (g : G) (x : V),
    ρV.act g (σ x) =
      σ (ρV.act g x)

/--
A function between two spaces carrying unitary representations is
equivariant if it commutes with the group actions.
-/
def IsEquivariantMap
    {X Y : FDHilbert.{uV}}
    (ρX : UnitaryRepresentation G X)
    (ρY : UnitaryRepresentation G Y)
    (f : X → Y) : Prop :=
  ∀ (g : G) (x : X),
    ρY.act g (f x) =
      f (ρX.act g x)
/--
A fixed feed-forward architecture between finite-dimensional real
Hilbert spaces.

The architecture contains no parameter values.
-/
inductive Architecture
    (G : Type uG)
    [Group G] :
    (X : FDHilbert.{uV}) →
    UnitaryRepresentation G X →
    (Y : FDHilbert.{uV}) →
    UnitaryRepresentation G Y →
    Type (max uG (uV + 1))
  /--
  The empty architecture. It will evaluate as the identity function.
  -/
  | identity
      (X : FDHilbert.{uV})
      (ρX : UnitaryRepresentation G X) :
      Architecture G X ρX X ρX
  /--
  Add one parameterized linear layer and one equivariant nonlinearity
  in front of an existing architecture.
  -/
  | layer
      {X H Y : FDHilbert.{uV}}
      {ρX : UnitaryRepresentation G X}
      (ρH : UnitaryRepresentation G H)
      {ρY : UnitaryRepresentation G Y}
      /- Hilbert parameter space of the current layer. -/
      (Q : FDHilbert.{uV})
      /- Unitary action on the current layer parameter. -/
      (ρQ : UnitaryRepresentation G Q)
      /- The linear layer associated with parameter `q`. -/
      (linearLayer : Q → X →ₗ[ℝ] H)
      /-
      Local parameter transformation law:

          L_{ρQ(g)q}(x)
            =
          ρH(g)(L_q(ρX(g⁻¹)x)).
      -/
      (linearLayer_transform :
        ∀ (g : G) (q : Q) (x : X),
          linearLayer (ρQ.act g q) x =
            ρH.act g
              (linearLayer q (ρX.act g⁻¹ x)))
      /- Equivariant nonlinearity on the hidden space. -/
      (σ : H → H)
      (hσ : IsEquivariant ρH σ)
      /- Remaining fixed architecture. -/
      (tail : Architecture G H ρH Y ρY) :
      Architecture G X ρX Y ρY

namespace Architecture

/--
The finite-dimensional Hilbert parameter space associated with a fixed
architecture.

* The identity architecture uses the zero-dimensional space
  `Fin 0 → X`.
* A layer with current parameter space `Q` and tail parameter space
  `P_tail` uses their `L²` product.
-/
noncomputable def paramSpace
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY) :
    FDHilbert.{uV} := by
  induction arch with
  | identity X ρX =>
      exact
        { carrier := WithLp 2 (Fin 0 → X)
          instNormedAddCommGroup := inferInstance
          instInnerProductSpace := inferInstance
          instFiniteDimensional := inferInstance }
  | layer ρH Q ρQ linearLayer linearLayer_transform
      sigma hσ tail ih =>
      exact
        { carrier :=
            WithLp 2 (Q × ih)
          instNormedAddCommGroup := inferInstance
          instInnerProductSpace := inferInstance
          instFiniteDimensional := inferInstance }

/--
The underlying parameter type associated with an architecture.
-/
abbrev Param
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY) :
    Type uV :=
  paramSpace arch


/--
The identity architecture has the zero-dimensional parameter space.
-/
@[simp]
theorem paramSpace_identity
    (X : FDHilbert.{uV})
    (ρX : UnitaryRepresentation G X) :
    (paramSpace
      (Architecture.identity
        (G := G)
        X
        ρX)).carrier =
      WithLp 2 (Fin 0 → X) := by
  rfl

/--
The parameter space of a layer architecture is the product of the
current layer parameter space and the parameter space of the tail.
-/
@[simp]
theorem paramSpace_layer
    {X H Y Q : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    (ρH : UnitaryRepresentation G H)
    {ρY : UnitaryRepresentation G Y}
    (ρQ : UnitaryRepresentation G Q)
    (linearLayer : Q → X →ₗ[ℝ] H)
    (linearLayer_transform :
      ∀ (g : G) (q : Q) (x : X),
        linearLayer (ρQ.act g q) x =
          ρH.act g
            (linearLayer q (ρX.act g⁻¹ x)))
    (sigma : H → H)
    (hσ : IsEquivariant ρH sigma)
    (tail : Architecture G H ρH Y ρY) :
    (paramSpace
      (Architecture.layer
        (ρX := ρX)
        ρH
        Q
        ρQ
        linearLayer
        linearLayer_transform
        sigma
        hσ
        tail)).carrier =
      WithLp 2 (Q × (paramSpace tail).carrier) := by
  rfl

/-!
## Recursive parameter actions

The parameter action is derived recursively from the architecture.

* The identity architecture has a zero-dimensional parameter space, so
  the action is the identity.
* For a layer, the group acts on the current layer parameter through
  `ρQ` and recursively on the tail parameter.
-/

/--
The action of `g : G` on the full parameter space of an architecture.
-/
noncomputable def paramAct
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (g : G) :
    Param arch → Param arch := by
  induction arch with
  | identity X ρX =>
      exact id
  | layer ρH Q ρQ linearLayer linearLayer_transform
      sigma hσ tail ih =>
      exact fun θ =>
        WithLp.toLp 2
          (ρQ.act g θ.ofLp.1,
           ih θ.ofLp.2)

/--
The parameter action on the identity architecture is the identity.
-/
@[simp]
theorem paramAct_identity
    (X : FDHilbert.{uV})
    (ρX : UnitaryRepresentation G X)
    (g : G)
    (θ :
      Param
        (Architecture.identity
          (G := G)
          X
          ρX)) :
    paramAct
        (Architecture.identity
          (G := G)
          X
          ρX)
        g
        θ =
      θ := by
  rfl

/--
The parameter action on a layer acts on the current layer parameter and
recursively on the tail parameter.
-/
@[simp]
theorem paramAct_layer
    {X H Y Q : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    (ρH : UnitaryRepresentation G H)
    {ρY : UnitaryRepresentation G Y}
    (ρQ : UnitaryRepresentation G Q)
    (linearLayer : Q → X →ₗ[ℝ] H)
    (linearLayer_transform :
      ∀ (g : G) (q : Q) (x : X),
        linearLayer (ρQ.act g q) x =
          ρH.act g
            (linearLayer q (ρX.act g⁻¹ x)))
    (sigma : H → H)
    (hσ : IsEquivariant ρH sigma)
    (tail : Architecture G H ρH Y ρY)
    (g : G)
    (θ :
      Param
        (Architecture.layer
          (ρX := ρX)
          ρH
          Q
          ρQ
          linearLayer
          linearLayer_transform
          sigma
          hσ
          tail)) :
    paramAct
        (Architecture.layer
          (ρX := ρX)
          ρH
          Q
          ρQ
          linearLayer
          linearLayer_transform
          sigma
          hσ
          tail)
        g
        θ =
      WithLp.toLp 2
        (ρQ.act g θ.ofLp.1,
         paramAct tail g θ.ofLp.2) := by
  rfl

/--
The identity element acts trivially on the full parameter space.
-/
@[simp]
theorem paramAct_one
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (θ : Param arch) :
    paramAct arch 1 θ = θ := by
  induction arch with
  | identity X ρX =>
      rfl
  | layer ρH Q ρQ linearLayer linearLayer_transform
      sigma hσ tail ih =>
      cases θ using WithLp.rec with
      | toLp θ' =>
          rcases θ' with ⟨q, θtail⟩
          change
            WithLp.toLp 2
                (ρQ.act 1 q, paramAct tail 1 θtail) =
              WithLp.toLp 2 (q, θtail)
          rw [ρQ.act_one]
          rw [ih θtail]
          rfl

/--
The recursively defined parameter action respects group
multiplication:

    paramAct arch (g * h) θ
      =
    paramAct arch g (paramAct arch h θ).
-/
theorem paramAct_mul
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (g h : G)
    (θ : Param arch) :
    paramAct arch (g * h) θ =
      paramAct arch g
        (paramAct arch h θ) := by
  induction arch with
  | identity X ρX =>
      rfl
  | layer ρH Q ρQ linearLayer linearLayer_transform
      sigma hσ tail ih =>
      cases θ using WithLp.rec with
      | toLp θ' =>
          rcases θ' with ⟨q, θtail⟩
          change
            WithLp.toLp 2
                (ρQ.act (g * h) q,
                 paramAct tail (g * h) θtail) =
              WithLp.toLp 2
                (ρQ.act g (ρQ.act h q),
                 paramAct tail g
                   (paramAct tail h θtail))
          rw [ρQ.act_mul]
          rw [ih θtail]

/--
Acting by `g⁻¹` after acting by `g` returns the original full
parameter.
-/
theorem paramAct_inv_apply
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (g : G)
    (θ : Param arch) :
    paramAct arch g⁻¹
        (paramAct arch g θ) =
      θ := by
  rw [← paramAct_mul]
  simp

/--
Acting by `g` after acting by `g⁻¹` returns the original full
parameter.
-/
theorem paramAct_apply_inv
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (g : G)
    (θ : Param arch) :
    paramAct arch g
        (paramAct arch g⁻¹ θ) =
      θ := by
  rw [← paramAct_mul]
  simp

/-!
## Recursive evaluation

Evaluation takes a fixed architecture, a parameter value in its
recursively derived parameter space, and an input vector.

* The identity architecture returns its input.
* A layer extracts the current parameter and the tail parameters,
  applies the current linear layer and nonlinearity, then recursively
  evaluates the tail.
-/

/--
Evaluate a fixed architecture at a parameter value.
-/
noncomputable def eval
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY) :
    Param arch → X → Y := by
  induction arch with
  | identity X ρX =>
      exact fun _ x => x
  | layer ρH Q ρQ linearLayer linearLayer_transform
      sigma hσ tail ih =>
      exact fun θ x =>
        ih θ.ofLp.2
          (sigma
            (linearLayer θ.ofLp.1 x))

/--
Evaluation of the identity architecture returns its input.
-/
@[simp]
theorem eval_identity
    (X : FDHilbert.{uV})
    (ρX : UnitaryRepresentation G X)
    (θ :
      Param
        (Architecture.identity
          (G := G)
          X
          ρX))
    (x : X) :
    eval
        (Architecture.identity
          (G := G)
          X
          ρX)
        θ
        x =
      x := by
  rfl

/--
Evaluation of a layer first evaluates the current linear layer, then
the nonlinearity, and finally the tail architecture.
-/
@[simp]
theorem eval_layer
    {X H Y Q : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    (ρH : UnitaryRepresentation G H)
    {ρY : UnitaryRepresentation G Y}
    (ρQ : UnitaryRepresentation G Q)
    (linearLayer : Q → X →ₗ[ℝ] H)
    (linearLayer_transform :
      ∀ (g : G) (q : Q) (x : X),
        linearLayer (ρQ.act g q) x =
          ρH.act g
            (linearLayer q (ρX.act g⁻¹ x)))
    (sigma : H → H)
    (hσ : IsEquivariant ρH sigma)
    (tail : Architecture G H ρH Y ρY)
    (θ :
      Param
        (Architecture.layer
          (ρX := ρX)
          ρH
          Q
          ρQ
          linearLayer
          linearLayer_transform
          sigma
          hσ
          tail))
    (x : X) :
    eval
        (Architecture.layer
          (ρX := ρX)
          ρH
          Q
          ρQ
          linearLayer
          linearLayer_transform
          sigma
          hσ
          tail)
        θ
        x =
      eval tail θ.ofLp.2
        (sigma
          (linearLayer θ.ofLp.1 x)) := by
  rfl

/-!
## The layer transformation identity

The local transformation law stored in the architecture has the form

    L_{ρQ(g)q}(x)
      =
    ρH(g)(L_q(ρX(g⁻¹)x)).

The following theorem puts it into the orientation needed in the
inductive proof of Lemma 2.1:

    L_q(ρX(g)x)
      =
    ρH(g)(L_{ρQ(g⁻¹)q}(x)).
-/

/--
The current linear layer transforms compatibly with a transformed input
and an inversely transformed parameter.
-/
theorem linearLayer_input_transform
    {X H Q : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    (ρH : UnitaryRepresentation G H)
    (ρQ : UnitaryRepresentation G Q)
    (linearLayer : Q → X →ₗ[ℝ] H)
    (linearLayer_transform :
      ∀ (g : G) (q : Q) (x : X),
        linearLayer (ρQ.act g q) x =
          ρH.act g
            (linearLayer q (ρX.act g⁻¹ x)))
    (g : G)
    (q : Q)
    (x : X) :
    linearLayer q (ρX.act g x) =
      ρH.act g
        (linearLayer (ρQ.act g⁻¹ q) x) := by
  symm
  calc
    ρH.act g
        (linearLayer (ρQ.act g⁻¹ q) x) =
      ρH.act g
        (ρH.act g⁻¹
          (linearLayer q
            (ρX.act (g⁻¹)⁻¹ x))) := by
      rw [linearLayer_transform g⁻¹ q x]
    _ =
      ρH.act g
        (ρH.act g⁻¹
          (linearLayer q
            (ρX.act g x))) := by
      rw [inv_inv]
    _ =
      ρH.act (g * g⁻¹)
        (linearLayer q
          (ρX.act g x)) := by
      symm
      exact ρH.act_mul
        g
        g⁻¹
        (linearLayer q (ρX.act g x))
    _ =
      ρH.act 1
        (linearLayer q
          (ρX.act g x)) := by
      rw [mul_inv_cancel]
    _ =
      linearLayer q
        (ρX.act g x) := by
      exact ρH.act_one
        (linearLayer q (ρX.act g x))

/-!
## Lemma 2.1

The full-network transformation law is now proved by induction over the
recursive architecture. It is not assumed as a field of the
architecture.
-/

/--
For every fixed architecture, parameter, group element, and input,

    f_θ(ρX(g)x)
      =
    ρY(g)f_{T(g⁻¹)θ}(x).

The parameter action `T` is the recursively defined `paramAct`.
-/
theorem lemma_2_1
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY) :
    ∀ (θ : Param arch) (g : G) (x : X),
      eval arch θ (ρX.act g x) =
        ρY.act g
          (eval arch
            (paramAct arch g⁻¹ θ)
            x) := by
  induction arch with
  | identity X ρX =>
      intro θ g x
      change ρX.act g x = ρX.act g x
      rfl
  | @layer X H Y ρX ρH ρY Q ρQ
      linearLayer linearLayer_transform
      sigma hσ tail ih =>
      intro θ g x
      cases θ using WithLp.rec with
      | toLp θ' =>
          rcases θ' with ⟨q, θtail⟩
          change
            eval tail θtail
                (sigma
                  (linearLayer q
                    (ρX.act g x))) =
              ρY.act g
                (eval tail
                  (paramAct tail g⁻¹ θtail)
                  (sigma
                    (linearLayer
                      (ρQ.act g⁻¹ q)
                      x)))
          rw [
            linearLayer_input_transform
              ρH
              ρQ
              linearLayer
              linearLayer_transform
              g
              q
              x
          ]
          rw [← hσ g]
          exact ih
            θtail
            g
            (sigma
              (linearLayer
                (ρQ.act g⁻¹ q)
                x))

/-!
## True risk and augmented risk

The data distribution is represented by a measure

    ν : Measure (X × Y).

Expectation with respect to the data distribution is therefore the
actual Bochner integral

    ∫ z, F z ∂ν.

The augmented risk additionally integrates over the group using the
measure

    μ : Measure G.
-/

open MeasureTheory

/--
The loss is invariant under the unitary output representation:

    loss (ρY(g)y) (ρY(g)y') = loss y y'.
-/
def IsInvariantLoss
    {Y : FDHilbert.{uV}}
    (ρY : UnitaryRepresentation G Y)
    (loss : Y → Y → ℝ) : Prop :=
  ∀ (g : G) (y y' : Y),
    loss (ρY.act g y) (ρY.act g y') =
      loss y y'

/--
The true, non-augmented risk of a fixed architecture at parameter `θ`:

    R(θ) = ∫ z, loss (f_θ(z.1)) z.2 ∂ν.

Here `ν` is the data distribution on `X × Y`.
-/
noncomputable def trueRisk
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace X]
    [MeasurableSpace Y]
    (arch : Architecture G X ρX Y ρY)
    (ν : Measure (X × Y))
    (loss : Y → Y → ℝ)
    (θ : Param arch) : ℝ :=
  ∫ z,
    loss
      (eval arch θ z.1)
      z.2
    ∂ν

/--
The true augmented risk:

    R_aug(θ)
      =
    ∫ g,
      ∫ z,
        loss
          (f_θ(ρX(g)z.1))
          (ρY(g)z.2)
        ∂ν
      ∂μ.

The inner integral is expectation over the data distribution `ν`.
The outer integral is averaging over the group using `μ`.
-/
noncomputable def augmentedRisk
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace G]
    [MeasurableSpace X]
    [MeasurableSpace Y]
    (arch : Architecture G X ρX Y ρY)
    (μ : Measure G)
    (ν : Measure (X × Y))
    (loss : Y → Y → ℝ)
    (θ : Param arch) : ℝ :=
  ∫ g,
    ∫ z,
      loss
        (eval arch θ (ρX.act g z.1))
        (ρY.act g z.2)
      ∂ν
    ∂μ

/--
The orbit form of the augmented risk:

    orbitRisk(θ)
      =
    ∫ g, R(paramAct(g⁻¹, θ)) ∂μ.

Lemma 2.1 and invariance of the loss show that this equals the original
definition of augmented risk.
-/
noncomputable def orbitRisk
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace G]
    [MeasurableSpace X]
    [MeasurableSpace Y]
    (arch : Architecture G X ρX Y ρY)
    (μ : Measure G)
    (ν : Measure (X × Y))
    (loss : Y → Y → ℝ)
    (θ : Param arch) : ℝ :=
  ∫ g,
    trueRisk arch ν loss
      (paramAct arch g⁻¹ θ)
    ∂μ

/--
The integral form of left invariance needed from Haar measure:

    ∫ g, F(hg) ∂μ = ∫ g, F(g) ∂μ.

For the normalized Haar measure of a compact group, this is the required
change-of-variables property.
-/
def IsLeftInvariantIntegral
    [MeasurableSpace G]
    (μ : Measure G) : Prop :=
  ∀ (F : G → ℝ) (h : G),
    (∫ g, F (h * g) ∂μ) =
      ∫ g, F g ∂μ

/--
Lemma 2.1 and loss invariance rewrite the true augmented risk as an
integral over the parameter orbit:

    R_aug(θ)
      =
    ∫ g, R(paramAct(g⁻¹, θ)) ∂μ.
-/
theorem augmentedRisk_eq_orbitRisk
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace G]
    [MeasurableSpace X]
    [MeasurableSpace Y]
    (arch : Architecture G X ρX Y ρY)
    (μ : Measure G)
    (ν : Measure (X × Y))
    (loss : Y → Y → ℝ)
    (hloss : IsInvariantLoss ρY loss)
    (θ : Param arch) :
    augmentedRisk arch μ ν loss θ =
      orbitRisk arch μ ν loss θ := by
  unfold augmentedRisk orbitRisk trueRisk
  apply integral_congr_ae
  filter_upwards with g
  apply integral_congr_ae
  filter_upwards with z
  rw [lemma_2_1 arch θ g z.1]
  exact hloss
    g
    (eval arch
      (paramAct arch g⁻¹ θ)
      z.1)
    z.2

/--
The orbit risk is invariant under the recursively defined parameter
action:

    orbitRisk(paramAct(h, θ)) = orbitRisk(θ).
-/
theorem orbitRisk_paramAct
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace G]
    [MeasurableSpace X]
    [MeasurableSpace Y]
    (arch : Architecture G X ρX Y ρY)
    (μ : Measure G)
    (hμ : IsLeftInvariantIntegral μ)
    (ν : Measure (X × Y))
    (loss : Y → Y → ℝ)
    (h : G)
    (θ : Param arch) :
    orbitRisk arch μ ν loss
        (paramAct arch h θ) =
      orbitRisk arch μ ν loss θ := by
  unfold orbitRisk
  let F : G → ℝ :=
    fun g =>
      trueRisk arch ν loss
        (paramAct arch g⁻¹ θ)
  have hpoint :
      (fun g =>
        trueRisk arch ν loss
          (paramAct arch g⁻¹
            (paramAct arch h θ))) =
        fun g => F (h⁻¹ * g) := by
    funext g
    unfold F
    rw [mul_inv_rev]
    rw [inv_inv]
    rw [paramAct_mul]
  rw [hpoint]
  exact hμ F h⁻¹

/--
The true augmented risk is invariant under the recursively defined
unitary action on the full parameter space:

    R_aug(paramAct(h, θ)) = R_aug(θ).
-/
theorem augmentedRisk_paramAct
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace G]
    [MeasurableSpace X]
    [MeasurableSpace Y]
    (arch : Architecture G X ρX Y ρY)
    (μ : Measure G)
    (hμ : IsLeftInvariantIntegral μ)
    (ν : Measure (X × Y))
    (loss : Y → Y → ℝ)
    (hloss : IsInvariantLoss ρY loss)
    (h : G)
    (θ : Param arch) :
    augmentedRisk arch μ ν loss
        (paramAct arch h θ) =
      augmentedRisk arch μ ν loss θ := by
  calc
    augmentedRisk arch μ ν loss
        (paramAct arch h θ) =
      orbitRisk arch μ ν loss
        (paramAct arch h θ) := by
      exact augmentedRisk_eq_orbitRisk
        arch
        μ
        ν
        loss
        hloss
        (paramAct arch h θ)
    _ =
      orbitRisk arch μ ν loss θ := by
      exact orbitRisk_paramAct
        arch
        μ
        hμ
        ν
        loss
        h
        θ
    _ =
      augmentedRisk arch μ ν loss θ := by
      symm
      exact augmentedRisk_eq_orbitRisk
        arch
        μ
        ν
        loss
        hloss
        θ


/-!
## Unitary structure of the recursive parameter action

The recursively defined parameter action is linear and preserves the
inner product. These facts are needed to derive gradient equivariance
from invariance of the augmented risk.
-/

/--
The recursive parameter action preserves addition.
-/
theorem paramAct_add
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (g : G)
    (θ η : Param arch) :
    paramAct arch g (θ + η) =
      paramAct arch g θ +
        paramAct arch g η := by
  induction arch with
  | identity X ρX =>
      rfl
  | layer ρH Q ρQ linearLayer linearLayer_transform
      sigma hσ tail ih =>
      cases θ using WithLp.rec with
      | toLp θ' =>
          rcases θ' with ⟨q, θtail⟩
          cases η using WithLp.rec with
          | toLp η' =>
              rcases η' with ⟨r, ηtail⟩
              change
                WithLp.toLp 2
                    (ρQ.act g (q + r),
                     paramAct tail g (θtail + ηtail)) =
                  WithLp.toLp 2
                    (ρQ.act g q + ρQ.act g r,
                     paramAct tail g θtail +
                       paramAct tail g ηtail)
              rw [(ρQ.act g).map_add]
              exact congrArg
                (fun p =>
                  WithLp.toLp 2
                    (ρQ.act g q + ρQ.act g r, p))
                (ih θtail ηtail)

/--
The recursive parameter action preserves real scalar multiplication.
-/
theorem paramAct_smul
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (g : G)
    (a : ℝ)
    (θ : Param arch) :
    paramAct arch g (a • θ) =
      a • paramAct arch g θ := by
  induction arch with
  | identity X ρX =>
      rfl
  | layer ρH Q ρQ linearLayer linearLayer_transform
      sigma hσ tail ih =>
      cases θ using WithLp.rec with
      | toLp θ' =>
          rcases θ' with ⟨q, θtail⟩
          change
            WithLp.toLp 2
                (ρQ.act g (a • q),
                 paramAct tail g (a • θtail)) =
              WithLp.toLp 2
                (a • ρQ.act g q,
                 a • paramAct tail g θtail)
          rw [(ρQ.act g).map_smul]
          exact congrArg
            (fun p =>
              WithLp.toLp 2
                (a • ρQ.act g q, p))
            (ih θtail)

/--
The recursive parameter action preserves negation.
-/
theorem paramAct_neg
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (g : G)
    (θ : Param arch) :
    paramAct arch g (-θ) =
      -paramAct arch g θ := by
  simpa only [neg_one_smul] using
    paramAct_smul arch g (-1 : ℝ) θ

/--
The recursive parameter action preserves the Hilbert inner product.
-/
theorem paramAct_inner
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (g : G)
    (θ η : Param arch) :
    inner ℝ
        (paramAct arch g θ)
        (paramAct arch g η) =
      inner ℝ θ η := by
  induction arch with
  | identity X ρX =>
      rfl
  | layer ρH Q ρQ linearLayer linearLayer_transform
      sigma hσ tail ih =>
      cases θ using WithLp.rec with
      | toLp θ' =>
          rcases θ' with ⟨q, θtail⟩
          cases η using WithLp.rec with
          | toLp η' =>
              rcases η' with ⟨r, ηtail⟩
              change
                inner ℝ
                    (WithLp.toLp 2
                      (ρQ.act g q,
                       paramAct tail g θtail))
                    (WithLp.toLp 2
                      (ρQ.act g r,
                       paramAct tail g ηtail)) =
                  inner ℝ
                    (WithLp.toLp 2 (q, θtail))
                    (WithLp.toLp 2 (r, ηtail))
              simp only [WithLp.prod_inner_apply]
              rw [(ρQ.act g).inner_map_map]
              rw [ih θtail ηtail]
              rfl

/--
The recursive parameter action preserves the norm.
-/
theorem paramAct_norm
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (g : G)
    (θ : Param arch) :
    ‖paramAct arch g θ‖ = ‖θ‖ := by
  have hsq :
      ‖paramAct arch g θ‖ ^ 2 =
        ‖θ‖ ^ 2 := by
    rw [← @inner_self_eq_norm_sq ℝ]
    rw [← @inner_self_eq_norm_sq ℝ]
    exact paramAct_inner arch g θ θ
  have hleft :
      0 ≤ ‖paramAct arch g θ‖ :=
    norm_nonneg _
  have hright :
      0 ≤ ‖θ‖ :=
    norm_nonneg _
  nlinarith

/--
The recursive parameter action as a linear map.
-/
noncomputable def paramActLinear
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (g : G) :
    Param arch →ₗ[ℝ] Param arch where
  toFun := paramAct arch g
  map_add' := paramAct_add arch g
  map_smul' := paramAct_smul arch g

@[simp]
theorem paramActLinear_apply
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (g : G)
    (θ : Param arch) :
    paramActLinear arch g θ =
      paramAct arch g θ :=
  rfl

/--
The recursive parameter action as a linear equivalence.
-/
noncomputable def paramActLinearEquiv
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (g : G) :
    Param arch ≃ₗ[ℝ] Param arch where
  toFun := paramAct arch g
  invFun := paramAct arch g⁻¹
  left_inv := paramAct_inv_apply arch g
  right_inv := paramAct_apply_inv arch g
  map_add' := paramAct_add arch g
  map_smul' := paramAct_smul arch g

@[simp]
theorem paramActLinearEquiv_apply
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (g : G)
    (θ : Param arch) :
    paramActLinearEquiv arch g θ =
      paramAct arch g θ :=
  rfl

/--
The recursive parameter action as a linear isometric equivalence.
-/
noncomputable def paramActIsometryEquiv
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (g : G) :
    Param arch ≃ₗᵢ[ℝ] Param arch :=
  (paramActLinearEquiv arch g).isometryOfInner
    (fun θ η => paramAct_inner arch g θ η)

@[simp]
theorem paramActIsometryEquiv_apply
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (g : G)
    (θ : Param arch) :
    paramActIsometryEquiv arch g θ =
      paramAct arch g θ :=
  rfl

/-!
## Gradient equivariance from risk invariance
-/

/--
A scalar function on the parameter space is invariant under the
recursive parameter action.
-/
def IsInvariantParameterFunction
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (risk : Param arch → ℝ) : Prop :=
  ∀ (g : G) (θ : Param arch),
    risk (paramAct arch g θ) =
      risk θ

/--
The gradient of a scalar function is equivariant under the recursive
parameter action.
-/
def IsEquivariantParameterGradient
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (risk : Param arch → ℝ) : Prop :=
  ∀ (g : G) (θ : Param arch),
    gradient risk (paramAct arch g θ) =
      paramAct arch g (gradient risk θ)

/--
Transport a gradient witness through the unitary parameter action.

If `risk` is invariant and has gradient `grad` at `θ`, then it has
gradient `paramAct arch g grad` at `paramAct arch g θ`.
-/
theorem hasGradientAt_paramAct
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (risk : Param arch → ℝ)
    (hrisk : IsInvariantParameterFunction arch risk)
    (g : G)
    (θ grad : Param arch)
    (hgrad : HasGradientAt risk grad θ) :
    HasGradientAt risk
      (paramAct arch g grad)
      (paramAct arch g θ) := by
  let e :
      Param arch ≃ₗᵢ[ℝ] Param arch :=
    paramActIsometryEquiv arch g⁻¹
  let eL :
      Param arch →L[ℝ] Param arch :=
    e.toContinuousLinearEquiv.toContinuousLinearMap
  have he_apply :
      ∀ η : Param arch,
        e η = paramAct arch g⁻¹ η := by
    intro η
    rfl
  have he_point :
      e (paramAct arch g θ) = θ := by
    change
      paramAct arch g⁻¹
        (paramAct arch g θ) =
      θ
    exact paramAct_inv_apply arch g θ
  have hgrad_at_transformed_point :
      HasFDerivAt risk
        (InnerProductSpace.toDual ℝ (Param arch) grad)
        (e (paramAct arch g θ)) := by
    rw [he_point]
    exact hgrad.hasFDerivAt
  have he_deriv :
      HasFDerivAt
        (fun η : Param arch => e η)
        eL
        (paramAct arch g θ) := by
    exact e.toContinuousLinearEquiv.hasFDerivAt
  have hcomp :
      HasFDerivAt
        (fun η : Param arch => risk (e η))
        ((InnerProductSpace.toDual ℝ (Param arch) grad).comp eL)
        (paramAct arch g θ) := by
    exact hgrad_at_transformed_point.comp
      (paramAct arch g θ)
      he_deriv
  have hdual :
      ((InnerProductSpace.toDual ℝ (Param arch) grad).comp eL) =
        InnerProductSpace.toDual ℝ (Param arch)
          (paramAct arch g grad) := by
    ext v
    simp only [
      ContinuousLinearMap.comp_apply,
      InnerProductSpace.toDual_apply_apply
    ]
    change
      inner ℝ grad (paramAct arch g⁻¹ v) =
        inner ℝ (paramAct arch g grad) v
    calc
      inner ℝ grad (paramAct arch g⁻¹ v) =
          inner ℝ
            (paramAct arch g grad)
            (paramAct arch g
              (paramAct arch g⁻¹ v)) := by
        symm
        exact paramAct_inner
          arch
          g
          grad
          (paramAct arch g⁻¹ v)
      _ =
          inner ℝ
            (paramAct arch g grad)
            v := by
        rw [paramAct_apply_inv arch g v]
  have hfun :
      (fun η : Param arch => risk (e η)) =
        risk := by
    funext η
    change
      risk (paramAct arch g⁻¹ η) =
        risk η
    exact hrisk g⁻¹ η
  rw [hfun] at hcomp
  rw [hdual] at hcomp
  have hgradient :
      HasGradientAt risk
        ((InnerProductSpace.toDual ℝ (Param arch)).symm
          ((InnerProductSpace.toDual ℝ (Param arch))
            (paramAct arch g grad)))
        (paramAct arch g θ) :=
    hcomp.hasGradientAt
  have hcancel :
      (InnerProductSpace.toDual ℝ (Param arch)).symm
          ((InnerProductSpace.toDual ℝ (Param arch))
            (paramAct arch g grad)) =
        paramAct arch g grad := by
    exact
      (InnerProductSpace.toDual ℝ (Param arch)).symm_apply_apply
        (paramAct arch g grad)
  rw [hcancel] at hgradient
  exact hgradient

/--
Invariance of a differentiable scalar function under the unitary
parameter action implies equivariance of its gradient.
-/
theorem gradient_equivariant_of_invariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (risk : Param arch → ℝ)
    (hrisk : IsInvariantParameterFunction arch risk)
    (hdiff :
      ∀ θ : Param arch,
        DifferentiableAt ℝ risk θ) :
    IsEquivariantParameterGradient arch risk := by
  intro g θ
  have hθ :
      HasGradientAt risk
        (gradient risk θ)
        θ :=
    (hdiff θ).hasGradientAt
  have htransformed :
      HasGradientAt risk
        (paramAct arch g (gradient risk θ))
        (paramAct arch g θ) :=
    hasGradientAt_paramAct
      arch
      risk
      hrisk
      g
      θ
      (gradient risk θ)
      hθ
  exact htransformed.gradient

/--
The augmented risk is an invariant scalar function on the recursively
derived parameter Hilbert space.
-/
theorem augmentedRisk_isInvariantParameterFunction
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace G]
    [MeasurableSpace X]
    [MeasurableSpace Y]
    (arch : Architecture G X ρX Y ρY)
    (μ : Measure G)
    (hμ : IsLeftInvariantIntegral μ)
    (ν : Measure (X × Y))
    (loss : Y → Y → ℝ)
    (hloss : IsInvariantLoss ρY loss) :
    IsInvariantParameterFunction arch
      (augmentedRisk arch μ ν loss) := by
  intro g θ
  exact augmentedRisk_paramAct
    arch μ hμ ν loss hloss g θ

/--
The augmented-risk gradient is equivariant under the recursive unitary
parameter action.
-/
theorem augmentedRisk_gradient_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace G]
    [MeasurableSpace X]
    [MeasurableSpace Y]
    (arch : Architecture G X ρX Y ρY)
    (μ : Measure G)
    (hμ : IsLeftInvariantIntegral μ)
    (ν : Measure (X × Y))
    (loss : Y → Y → ℝ)
    (hloss : IsInvariantLoss ρY loss)
    (hdiff :
      ∀ θ : Param arch,
        DifferentiableAt ℝ
          (augmentedRisk arch μ ν loss)
          θ) :
    IsEquivariantParameterGradient arch
      (augmentedRisk arch μ ν loss) := by
  exact gradient_equivariant_of_invariant
    arch
    (augmentedRisk arch μ ν loss)
    (augmentedRisk_isInvariantParameterFunction
      arch μ hμ ν loss hloss)
    hdiff

/-!
## Negative-gradient flow
-/

/--
The negative gradient vector field.
-/
noncomputable def negativeGradientField
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (risk : Param arch → ℝ) :
    Param arch → Param arch :=
  fun θ => -gradient risk θ

/--
Equivariance of the gradient implies equivariance of the negative
gradient field.
-/
theorem negativeGradientField_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (risk : Param arch → ℝ)
    (hgrad :
      IsEquivariantParameterGradient arch risk) :
    ∀ (g : G) (θ : Param arch),
      negativeGradientField arch risk
          (paramAct arch g θ) =
        paramAct arch g
          (negativeGradientField arch risk θ) := by
  intro g θ
  unfold negativeGradientField
  rw [hgrad g θ]
  rw [paramAct_neg]

/--
A gradient-flow solution starts at `θ₀` and satisfies

    θ'(t) = -∇risk(θ(t)).
-/
def IsGradientFlowSolution
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (risk : Param arch → ℝ)
    (curve : ℝ → Param arch)
    (θ₀ : Param arch) : Prop :=
  curve 0 = θ₀ ∧
    ∀ t : ℝ,
      HasDerivAt curve
        (negativeGradientField arch risk (curve t))
        t

/--
The gradient-flow equation is equivariant when transforming a solution
curve gives another solution from the transformed initial parameter.
-/
def IsEquivariantGradientFlow
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (risk : Param arch → ℝ) : Prop :=
  ∀ (g : G)
      (curve : ℝ → Param arch)
      (θ₀ : Param arch),
    IsGradientFlowSolution arch risk curve θ₀ →
      IsGradientFlowSolution arch risk
        (fun t => paramAct arch g (curve t))
        (paramAct arch g θ₀)

/--
Uniqueness of gradient-flow solutions.
-/
def HasUniqueGradientFlows
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (risk : Param arch → ℝ) : Prop :=
  ∀ (curve₁ curve₂ : ℝ → Param arch)
      (θ₀ : Param arch),
    IsGradientFlowSolution arch risk curve₁ θ₀ →
    IsGradientFlowSolution arch risk curve₂ θ₀ →
    curve₁ = curve₂

/--
A flow map realizes the gradient-flow solution from each initial
parameter.
-/
def IsGradientFlowMap
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (risk : Param arch → ℝ)
    (flow : ℝ → Param arch → Param arch) : Prop :=
  ∀ θ₀ : Param arch,
    IsGradientFlowSolution arch risk
      (fun t => flow t θ₀)
      θ₀

/--
An equivariant vector field makes the gradient-flow solution predicate
equivariant.
-/
theorem gradientFlowSolution_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (risk : Param arch → ℝ)
    (hfield :
      ∀ (g : G) (θ : Param arch),
        negativeGradientField arch risk
            (paramAct arch g θ) =
          paramAct arch g
            (negativeGradientField arch risk θ)) :
    IsEquivariantGradientFlow arch risk := by
  intro g curve θ₀ hcurve
  constructor
  · change
      paramAct arch g (curve 0) =
        paramAct arch g θ₀
    rw [hcurve.1]
  · intro t
    let e : Param arch ≃ₗᵢ[ℝ] Param arch :=
      paramActIsometryEquiv arch g
    let L : Param arch →L[ℝ] Param arch :=
      e.toLinearIsometry.toContinuousLinearMap
    have hL_apply :
        ∀ θ : Param arch,
          L θ = paramAct arch g θ := by
      intro θ
      rfl
    have hL :
        HasFDerivAt
          (fun θ : Param arch => L θ)
          L
          (curve t) :=
      L.hasFDerivAt
    have hcomp' :=
      hL.comp_hasDerivAt t (hcurve.2 t)
    change
      HasDerivAt
        (fun s : ℝ => L (curve s))
        (L
          (negativeGradientField
            arch risk (curve t)))
        t at hcomp'
    have hderiv :
        HasDerivAt
          (fun s : ℝ =>
            paramAct arch g (curve s))
          (paramAct arch g
            (negativeGradientField
              arch risk (curve t)))
          t := by
      convert hcomp' using 1
      · funext s
        exact hL_apply (curve s)
      · exact hL_apply
          (negativeGradientField
            arch risk (curve t))
    change
      HasDerivAt
        (fun s : ℝ =>
          paramAct arch g (curve s))
        (negativeGradientField arch risk
          (paramAct arch g (curve t)))
        t
    rw [hfield g (curve t)]
    exact hderiv
/--
If the gradient is equivariant, then the gradient-flow equation is
equivariant.
-/
theorem gradientFlowEquation_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (risk : Param arch → ℝ)
    (hgrad :
      IsEquivariantParameterGradient arch risk) :
    IsEquivariantGradientFlow arch risk := by
  exact gradientFlowSolution_equivariant
    arch
    risk
    (negativeGradientField_equivariant
      arch risk hgrad)

/--
Equivariance of the gradient-flow equation and uniqueness of solutions
imply equivariance of the flow map.
-/
theorem gradientFlowMap_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (risk : Param arch → ℝ)
    (flow : ℝ → Param arch → Param arch)
    (hEquation :
      IsEquivariantGradientFlow arch risk)
    (hUnique :
      HasUniqueGradientFlows arch risk)
    (hFlow :
      IsGradientFlowMap arch risk flow) :
    ∀ (t : ℝ) (g : G) (θ₀ : Param arch),
      flow t (paramAct arch g θ₀) =
        paramAct arch g (flow t θ₀) := by
  intro t g θ₀
  have hDirect :
      IsGradientFlowSolution arch risk
        (fun s => flow s (paramAct arch g θ₀))
        (paramAct arch g θ₀) :=
    hFlow (paramAct arch g θ₀)
  have hTransformed :
      IsGradientFlowSolution arch risk
        (fun s => paramAct arch g (flow s θ₀))
        (paramAct arch g θ₀) := by
    exact hEquation
      g
      (fun s => flow s θ₀)
      θ₀
      (hFlow θ₀)
  have hCurves :
      (fun s => flow s (paramAct arch g θ₀)) =
        fun s => paramAct arch g (flow s θ₀) := by
    exact hUnique
      (fun s => flow s (paramAct arch g θ₀))
      (fun s => paramAct arch g (flow s θ₀))
      (paramAct arch g θ₀)
      hDirect
      hTransformed
  exact congrFun hCurves t

/--
The flow map for augmented-risk gradient flow is equivariant, assuming
differentiability and uniqueness of solutions.
-/
theorem augmentedRisk_gradientFlowMap_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace G]
    [MeasurableSpace X]
    [MeasurableSpace Y]
    (arch : Architecture G X ρX Y ρY)
    (μ : Measure G)
    (hμ : IsLeftInvariantIntegral μ)
    (ν : Measure (X × Y))
    (loss : Y → Y → ℝ)
    (hloss : IsInvariantLoss ρY loss)
    (hdiff :
      ∀ θ : Param arch,
        DifferentiableAt ℝ
          (augmentedRisk arch μ ν loss)
          θ)
    (flow : ℝ → Param arch → Param arch)
    (hUnique :
      HasUniqueGradientFlows arch
        (augmentedRisk arch μ ν loss))
    (hFlow :
      IsGradientFlowMap arch
        (augmentedRisk arch μ ν loss)
        flow) :
    ∀ (t : ℝ) (g : G) (θ₀ : Param arch),
      flow t (paramAct arch g θ₀) =
        paramAct arch g (flow t θ₀) := by
  have hgrad :
      IsEquivariantParameterGradient arch
        (augmentedRisk arch μ ν loss) :=
    augmentedRisk_gradient_equivariant
      arch μ hμ ν loss hloss hdiff
  have hEquation :
      IsEquivariantGradientFlow arch
        (augmentedRisk arch μ ν loss) :=
    gradientFlowEquation_equivariant
      arch
      (augmentedRisk arch μ ν loss)
      hgrad
  exact gradientFlowMap_equivariant
    arch
    (augmentedRisk arch μ ν loss)
    flow
    hEquation
    hUnique
    hFlow

/-!
## Invariant parameter distributions

A parameter distribution is represented by a measure on the recursively
derived parameter space `Param arch`.

Invariance under the group action means that pushing the measure forward
through `paramAct arch g` leaves the measure unchanged:

    Measure.map (paramAct arch g) ν = ν.

We include measurability of every parameter action because it is needed
to use the API for pushforward measures and measure-preserving maps.
-/

/--
A parameter distribution `ν` is invariant under the recursively defined
parameter action if every group action is measurable and preserves `ν`.
-/
structure IsInvariantParameterDistribution
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν : MeasureTheory.Measure (Param arch)) :
    Prop where

  /--
  Every parameter transformation is measurable.
  -/
  measurable_paramAct :
    ∀ g : G,
      Measurable (paramAct arch g)

  /--
  The pushforward of `ν` under every parameter transformation equals
  `ν` itself.
  -/
  map_paramAct :
    ∀ g : G,
      MeasureTheory.Measure.map
          (paramAct arch g)
          ν =
        ν

/--
An invariant parameter distribution makes every parameter action
measure preserving.
-/
theorem IsInvariantParameterDistribution.measurePreserving
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν : MeasureTheory.Measure (Param arch))
    (hν : IsInvariantParameterDistribution arch ν)
    (g : G) :
    MeasureTheory.MeasurePreserving
      (paramAct arch g)
      ν
      ν where
  measurable := hν.measurable_paramAct g
  map_eq := hν.map_paramAct g

/--
The identity parameter action preserves every parameter measure.
-/
theorem map_paramAct_one
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν : MeasureTheory.Measure (Param arch)) :
    MeasureTheory.Measure.map
        (paramAct arch 1)
        ν =
      ν := by
  have hid :
      paramAct arch 1 =
        id := by
    funext θ
    exact paramAct_one arch θ
  rw [hid]
  exact MeasureTheory.Measure.map_id

/--
If a parameter measure is invariant, it is in particular invariant
under the inverse action.
-/
theorem map_paramAct_inv
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν : MeasureTheory.Measure (Param arch))
    (hν : IsInvariantParameterDistribution arch ν)
    (g : G) :
    MeasureTheory.Measure.map
        (paramAct arch g⁻¹)
        ν =
      ν :=
  hν.map_paramAct g⁻¹

/--
Invariant parameter distributions are closed under composition of group
actions at the level of pushforward measures.
-/
theorem map_paramAct_mul
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν : MeasureTheory.Measure (Param arch))
    (hν : IsInvariantParameterDistribution arch ν)
    (g h : G) :
    MeasureTheory.Measure.map
        (paramAct arch (g * h))
        ν =
      ν :=
  hν.map_paramAct (g * h)

/--
Pointwise equality between the action associated with `g * h` and the
composition of the actions associated with `h` and `g`.
-/
theorem paramAct_mul_function
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (g h : G) :
    paramAct arch (g * h) =
      (paramAct arch g) ∘
        (paramAct arch h) := by
  funext θ
  exact paramAct_mul arch g h θ

/--
A probability measure that is invariant under the parameter action is
an invariant initialization distribution.

The probability assumption is represented separately by the
`IsProbabilityMeasure` typeclass.
-/
def IsInvariantInitialization
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν : MeasureTheory.Measure (Param arch))
    [MeasureTheory.IsProbabilityMeasure ν] :
    Prop :=
  IsInvariantParameterDistribution arch ν

/--
The pushforward of an initial parameter distribution through a
deterministic map.
-/
noncomputable def pushforwardParameterDistribution
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν : MeasureTheory.Measure (Param arch))
    (f : Param arch → Param arch) :
    MeasureTheory.Measure (Param arch) :=
  MeasureTheory.Measure.map f ν

/--
The parameter distribution at time `t` is the pushforward of the
initial distribution through the time-`t` flow map.
-/
noncomputable def parameterDistributionAtTime
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν₀ : MeasureTheory.Measure (Param arch))
    (flow : ℝ → Param arch → Param arch)
    (t : ℝ) :
    MeasureTheory.Measure (Param arch) :=
  MeasureTheory.Measure.map
    (flow t)
    ν₀

/-!
## Preservation of invariant distributions by equivariant flows

Let

    νₜ = Measure.map (flow t) ν₀.

If the initial distribution `ν₀` is invariant under every parameter
action and the flow map is equivariant, then `νₜ` is invariant under
every parameter action.
-/

/--
The parameter distribution at time `t` is invariant if the initial
distribution is invariant and the time-`t` flow map is measurable and
equivariant.
-/
theorem parameterDistributionAtTime_invariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν₀ : MeasureTheory.Measure (Param arch))
    (flow : ℝ → Param arch → Param arch)
    (hν₀ :
      IsInvariantParameterDistribution arch ν₀)
    (hflow_measurable :
      ∀ t : ℝ,
        Measurable (flow t))
    (hflow_equivariant :
      ∀ (t : ℝ) (g : G) (θ : Param arch),
        flow t (paramAct arch g θ) =
          paramAct arch g (flow t θ))
    (t : ℝ) :
    IsInvariantParameterDistribution arch
      (parameterDistributionAtTime arch ν₀ flow t) := by
  constructor
  · intro g
    exact hν₀.measurable_paramAct g
  · intro g
    unfold parameterDistributionAtTime
    have hact :
        Measurable (paramAct arch g) :=
      hν₀.measurable_paramAct g
    have hflow :
        Measurable (flow t) :=
      hflow_measurable t
    have hcommute :
        (paramAct arch g) ∘ (flow t) =
          (flow t) ∘ (paramAct arch g) := by
      funext θ
      exact (hflow_equivariant t g θ).symm
    calc
      MeasureTheory.Measure.map
          (paramAct arch g)
          (MeasureTheory.Measure.map
            (flow t)
            ν₀) =
        MeasureTheory.Measure.map
          ((paramAct arch g) ∘ (flow t))
          ν₀ := by
        exact MeasureTheory.Measure.map_map
          hact
          hflow
      _ =
        MeasureTheory.Measure.map
          ((flow t) ∘ (paramAct arch g))
          ν₀ := by
        rw [hcommute]
      _ =
        MeasureTheory.Measure.map
          (flow t)
          (MeasureTheory.Measure.map
            (paramAct arch g)
            ν₀) := by
        symm
        exact MeasureTheory.Measure.map_map
          hflow
          hact
      _ =
        MeasureTheory.Measure.map
          (flow t)
          ν₀ := by
        rw [hν₀.map_paramAct g]

/--
Direct `Measure.map` formulation of invariance of the parameter
distribution at time `t`.
-/
theorem map_paramAct_parameterDistributionAtTime
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν₀ : MeasureTheory.Measure (Param arch))
    (flow : ℝ → Param arch → Param arch)
    (hν₀ :
      IsInvariantParameterDistribution arch ν₀)
    (hflow_measurable :
      ∀ t : ℝ,
        Measurable (flow t))
    (hflow_equivariant :
      ∀ (t : ℝ) (g : G) (θ : Param arch),
        flow t (paramAct arch g θ) =
          paramAct arch g (flow t θ))
    (t : ℝ)
    (g : G) :
    MeasureTheory.Measure.map
        (paramAct arch g)
        (parameterDistributionAtTime
          arch ν₀ flow t) =
      parameterDistributionAtTime
        arch ν₀ flow t := by
  exact
    (parameterDistributionAtTime_invariant
      arch
      ν₀
      flow
      hν₀
      hflow_measurable
      hflow_equivariant
      t).map_paramAct g

/--
At every time, each parameter action is measure preserving with respect
to the time-`t` parameter distribution.
-/
theorem paramAct_measurePreserving_atTime
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν₀ : MeasureTheory.Measure (Param arch))
    (flow : ℝ → Param arch → Param arch)
    (hν₀ :
      IsInvariantParameterDistribution arch ν₀)
    (hflow_measurable :
      ∀ t : ℝ,
        Measurable (flow t))
    (hflow_equivariant :
      ∀ (t : ℝ) (g : G) (θ : Param arch),
        flow t (paramAct arch g θ) =
          paramAct arch g (flow t θ))
    (t : ℝ)
    (g : G) :
    MeasureTheory.MeasurePreserving
      (paramAct arch g)
      (parameterDistributionAtTime
        arch ν₀ flow t)
      (parameterDistributionAtTime
        arch ν₀ flow t) := by
  exact
    IsInvariantParameterDistribution.measurePreserving
      arch
      (parameterDistributionAtTime
        arch ν₀ flow t)
      (parameterDistributionAtTime_invariant
        arch
        ν₀
        flow
        hν₀
        hflow_measurable
        hflow_equivariant
        t)
      g

/--
The parameter distribution remains invariant under the gradient flow of
the augmented risk.

The flow equivariance used here is obtained from:

* invariance of augmented risk;
* unitarity of the parameter action;
* equivariance of the augmented-risk gradient;
* uniqueness of gradient-flow solutions.
-/
theorem augmentedRisk_gradientFlow_preserves_invariantDistribution
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace G]
    [MeasurableSpace X]
    [MeasurableSpace Y]
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (μ : MeasureTheory.Measure G)
    (hμ : IsLeftInvariantIntegral μ)
    (ν : MeasureTheory.Measure (X × Y))
    (loss : Y → Y → ℝ)
    (hloss : IsInvariantLoss ρY loss)
    (hdiff :
      ∀ θ : Param arch,
        DifferentiableAt ℝ
          (augmentedRisk arch μ ν loss)
          θ)
    (flow : ℝ → Param arch → Param arch)
    (hflow_measurable :
      ∀ t : ℝ,
        Measurable (flow t))
    (hUnique :
      HasUniqueGradientFlows arch
        (augmentedRisk arch μ ν loss))
    (hFlow :
      IsGradientFlowMap arch
        (augmentedRisk arch μ ν loss)
        flow)
    (ν₀ : MeasureTheory.Measure (Param arch))
    (hν₀ :
      IsInvariantParameterDistribution arch ν₀) :
    ∀ t : ℝ,
      IsInvariantParameterDistribution arch
        (parameterDistributionAtTime
          arch ν₀ flow t) := by
  have hflow_equivariant :
      ∀ (t : ℝ) (g : G) (θ₀ : Param arch),
        flow t (paramAct arch g θ₀) =
          paramAct arch g (flow t θ₀) := by
    exact augmentedRisk_gradientFlowMap_equivariant
      arch
      μ
      hμ
      ν
      loss
      hloss
      hdiff
      flow
      hUnique
      hFlow
  intro t
  exact parameterDistributionAtTime_invariant
    arch
    ν₀
    flow
    hν₀
    hflow_measurable
    hflow_equivariant
    t

/-!
## Bochner means of parameters and network outputs

The parameter mean is the Bochner integral of the identity function on
the recursively derived parameter Hilbert space.

The output mean at an input `x` is the Bochner integral of the map

    θ ↦ eval arch θ x

with respect to the parameter distribution.
-/

/--
The mean parameter associated with a parameter distribution `ν`:

    Eν[θ] = ∫ θ, θ ∂ν.
-/
noncomputable def parameterMean
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν : MeasureTheory.Measure (Param arch)) :
    Param arch :=
  ∫ θ, θ ∂ν

/--
The pointwise mean of network outputs under a parameter distribution
`ν`:

    outputMean(x) = ∫ θ, f_θ(x) ∂ν.
-/
noncomputable def outputMean
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν : MeasureTheory.Measure (Param arch)) :
    X → Y :=
  fun x =>
    ∫ θ,
      eval arch θ x
    ∂ν

/--
The network obtained by first taking the mean parameter and then
evaluating the architecture:

    parameterMeanNetwork(x) = f_{Eν[θ]}(x).
-/
noncomputable def parameterMeanNetwork
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν : MeasureTheory.Measure (Param arch)) :
    X → Y :=
  fun x =>
    eval arch (parameterMean arch ν) x

/--
Integrability of the identity map on the parameter space.

This is the condition ensuring that the parameter mean is represented
by an ordinary Bochner-integrable random parameter.
-/
def HasIntegrableParameterMean
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν : MeasureTheory.Measure (Param arch)) :
    Prop :=
  MeasureTheory.Integrable
    (fun θ : Param arch => θ)
    ν

/--
Integrability of the network output over the parameter distribution at
a fixed input `x`.
-/
def HasIntegrableOutputMeanAt
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν : MeasureTheory.Measure (Param arch))
    (x : X) :
    Prop :=
  MeasureTheory.Integrable
    (fun θ : Param arch =>
      eval arch θ x)
    ν

/--
Integrability of the network output over the parameter distribution at
every input.
-/
def HasIntegrableOutputMean
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν : MeasureTheory.Measure (Param arch)) :
    Prop :=
  ∀ x : X,
    HasIntegrableOutputMeanAt arch ν x

/-!
## Means along a parameter flow
-/

/--
The mean parameter at time `t`, where the time-`t` distribution is the
pushforward of `ν₀` through `flow t`:

    E[θ(t)] = ∫ θ, θ ∂νₜ.
-/
noncomputable def parameterMeanAtTime
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν₀ : MeasureTheory.Measure (Param arch))
    (flow : ℝ → Param arch → Param arch)
    (t : ℝ) :
    Param arch :=
  parameterMean arch
    (parameterDistributionAtTime
      arch ν₀ flow t)

/--
The pointwise ensemble mean at time `t`:

    x ↦ ∫ θ, f_θ(x) ∂νₜ.
-/
noncomputable def outputMeanAtTime
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν₀ : MeasureTheory.Measure (Param arch))
    (flow : ℝ → Param arch → Param arch)
    (t : ℝ) :
    X → Y :=
  outputMean arch
    (parameterDistributionAtTime
      arch ν₀ flow t)

/--
The network evaluated at the mean parameter at time `t`:

    x ↦ f_{E[θ(t)]}(x).
-/
noncomputable def parameterMeanNetworkAtTime
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν₀ : MeasureTheory.Measure (Param arch))
    (flow : ℝ → Param arch → Param arch)
    (t : ℝ) :
    X → Y :=
  parameterMeanNetwork arch
    (parameterDistributionAtTime
      arch ν₀ flow t)

/--
The mean parameter at time `t` equals the Bochner integral of the
time-`t` flow map with respect to the initial distribution.
-/
theorem parameterMeanAtTime_eq_integral_flow
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    [BorelSpace (Param arch)]
    (ν₀ : MeasureTheory.Measure (Param arch))
    (flow : ℝ → Param arch → Param arch)
    (t : ℝ)
    (hflow : Measurable (flow t)) :
    parameterMeanAtTime arch ν₀ flow t =
      ∫ θ₀, flow t θ₀ ∂ν₀ := by
  unfold parameterMeanAtTime
  unfold parameterMean
  unfold parameterDistributionAtTime
  exact MeasureTheory.integral_map
    hflow.aemeasurable
    measurable_id.aestronglyMeasurable

/--
The pointwise output mean at time `t` equals the Bochner integral of
the network output evaluated along the parameter flow, with respect to
the initial parameter distribution:

    outputMeanAtTime(t, x)
      =
    ∫ θ₀, f_{flow(t, θ₀)}(x) ∂ν₀.

This is the change-of-variables formula for the pushforward measure

    νₜ = Measure.map (flow t) ν₀.
-/
theorem outputMeanAtTime_eq_integral_flow
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    [BorelSpace (Param arch)]
    (ν₀ : MeasureTheory.Measure (Param arch))
    (flow : ℝ → Param arch → Param arch)
    (t : ℝ)
    (x : X)
    (hflow : Measurable (flow t))
    (heval :
      AEStronglyMeasurable
        (fun θ : Param arch =>
          eval arch θ x)
        (MeasureTheory.Measure.map
          (flow t)
          ν₀)) :
    outputMeanAtTime arch ν₀ flow t x =
      ∫ θ₀,
        eval arch (flow t θ₀) x
      ∂ν₀ := by
  unfold outputMeanAtTime
  unfold outputMean
  unfold parameterDistributionAtTime
  exact MeasureTheory.integral_map
    hflow.aemeasurable
    heval

/-!
## Equivariance of the pointwise output mean

The proof has three steps:

1. Lemma 2.1 moves the input action into the output and parameter
   actions.
2. Linearity of the Bochner integral moves the output action outside
   the integral.
3. Invariance of the parameter measure removes the inverse parameter
   action.
-/

/--
The Bochner integral is unchanged when the integrand is precomposed
with a parameter action preserving the parameter distribution.
-/
theorem integral_eval_paramAct_eq
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    [BorelSpace (Param arch)]
    [MeasurableSpace Y]
    [BorelSpace Y]
    (ν : MeasureTheory.Measure (Param arch))
    (hν : IsInvariantParameterDistribution arch ν)
    (g : G)
    (x : X)
    (heval :
      Measurable
        (fun θ : Param arch =>
          eval arch θ x)) :
    (∫ θ,
        eval arch
          (paramAct arch g θ)
          x
      ∂ν) =
      ∫ θ,
        eval arch θ x
      ∂ν := by
  have hact :
      Measurable (paramAct arch g) :=
    hν.measurable_paramAct g
  have hmap :
      MeasureTheory.Measure.map
          (paramAct arch g)
          ν =
        ν :=
    hν.map_paramAct g
  have heval_map :
      MeasureTheory.AEStronglyMeasurable
        (fun θ : Param arch =>
          eval arch θ x)
        (MeasureTheory.Measure.map
          (paramAct arch g)
          ν) := by
    rw [hmap]
    exact heval.aestronglyMeasurable
  have hchange :=
    MeasureTheory.integral_map
      hact.aemeasurable
      heval_map
  rw [hmap] at hchange
  exact hchange.symm

/--
Pointwise equivariance of the output mean:

    ∫ θ, f_θ(ρX(g)x) ∂ν
      =
    ρY(g)(∫ θ, f_θ(x) ∂ν).

The required integrability hypothesis is for the inversely transformed
network output. It is needed to commute the continuous linear map
`ρY(g)` with the Bochner integral.
-/
theorem outputMean_equivariant_pointwise
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    [BorelSpace (Param arch)]
    [MeasurableSpace Y]
    [BorelSpace Y]
    (ν : MeasureTheory.Measure (Param arch))
    (hν : IsInvariantParameterDistribution arch ν)
    (g : G)
    (x : X)
    (heval :
      Measurable
        (fun θ : Param arch =>
          eval arch θ x))
    (hintegrable :
      MeasureTheory.Integrable
        (fun θ : Param arch =>
          eval arch
            (paramAct arch g⁻¹ θ)
            x)
        ν) :
    outputMean arch ν (ρX.act g x) =
      ρY.act g (outputMean arch ν x) := by
  let L : Y →L[ℝ] Y :=
    (ρY.act g).toLinearIsometry.toContinuousLinearMap
  have hL_apply :
      ∀ y : Y,
        L y = ρY.act g y := by
    intro y
    rfl
  unfold outputMean
  calc
    (∫ θ,
        eval arch θ (ρX.act g x)
      ∂ν) =
      ∫ θ,
        ρY.act g
          (eval arch
            (paramAct arch g⁻¹ θ)
            x)
      ∂ν := by
        apply MeasureTheory.integral_congr_ae
        filter_upwards with θ
        exact lemma_2_1 arch θ g x
    _ =
      ρY.act g
        (∫ θ,
          eval arch
            (paramAct arch g⁻¹ θ)
            x
        ∂ν) := by
        change
          (∫ θ,
            L
              (eval arch
                (paramAct arch g⁻¹ θ)
                x)
          ∂ν) =
            L
              (∫ θ,
                eval arch
                  (paramAct arch g⁻¹ θ)
                  x
              ∂ν)
        exact L.integral_comp_comm hintegrable
    _ =
      ρY.act g
        (∫ θ,
          eval arch θ x
        ∂ν) := by
        apply congrArg (fun y : Y => ρY.act g y)
        exact integral_eval_paramAct_eq
          arch
          ν
          hν
          g⁻¹
          x
          heval

/--
The pointwise output mean is an equivariant map:

    ρY(g)(outputMean(x))
      =
    outputMean(ρX(g)x).
-/
theorem outputMean_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    [BorelSpace (Param arch)]
    [MeasurableSpace Y]
    [BorelSpace Y]
    (ν : MeasureTheory.Measure (Param arch))
    (hν : IsInvariantParameterDistribution arch ν)
    (heval :
      ∀ x : X,
        Measurable
          (fun θ : Param arch =>
            eval arch θ x))
    (hintegrable :
      ∀ (g : G) (x : X),
        MeasureTheory.Integrable
          (fun θ : Param arch =>
            eval arch
              (paramAct arch g⁻¹ θ)
              x)
          ν) :
    IsEquivariantMap ρX ρY
      (outputMean arch ν) := by
  intro g x
  symm
  exact outputMean_equivariant_pointwise
    arch
    ν
    hν
    g
    x
    (heval x)
    (hintegrable g x)

/--
At every time `t`, invariance of the time-dependent parameter
distribution implies equivariance of the pointwise output mean.
-/
theorem outputMeanAtTime_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    [BorelSpace (Param arch)]
    [MeasurableSpace Y]
    [BorelSpace Y]
    (ν₀ : MeasureTheory.Measure (Param arch))
    (flow : ℝ → Param arch → Param arch)
    (hν₀ :
      IsInvariantParameterDistribution arch ν₀)
    (hflow_measurable :
      ∀ t : ℝ,
        Measurable (flow t))
    (hflow_equivariant :
      ∀ (t : ℝ) (g : G) (θ : Param arch),
        flow t (paramAct arch g θ) =
          paramAct arch g (flow t θ))
    (heval :
      ∀ x : X,
        Measurable
          (fun θ : Param arch =>
            eval arch θ x))
    (hintegrable :
      ∀ (t : ℝ) (g : G) (x : X),
        MeasureTheory.Integrable
          (fun θ : Param arch =>
            eval arch
              (paramAct arch g⁻¹ θ)
              x)
          (parameterDistributionAtTime
            arch ν₀ flow t)) :
    ∀ t : ℝ,
      IsEquivariantMap ρX ρY
        (outputMeanAtTime
          arch ν₀ flow t) := by
  intro t
  have hνt :
      IsInvariantParameterDistribution arch
        (parameterDistributionAtTime
          arch ν₀ flow t) :=
    parameterDistributionAtTime_invariant
      arch
      ν₀
      flow
      hν₀
      hflow_measurable
      hflow_equivariant
      t
  unfold outputMeanAtTime
  exact outputMean_equivariant
    arch
    (parameterDistributionAtTime
      arch ν₀ flow t)
    hνt
    heval
    (hintegrable t)

/--
Under gradient flow of the augmented risk, the pointwise output mean is
equivariant at every time.

Flow equivariance is derived from augmented-risk invariance, unitarity
of the parameter action, gradient equivariance, and uniqueness of the
gradient-flow solution. It is not assumed separately.
-/
theorem augmentedRisk_outputMeanAtTime_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace G]
    [MeasurableSpace X]
    [MeasurableSpace Y]
    [BorelSpace Y]
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    [BorelSpace (Param arch)]
    (μ : MeasureTheory.Measure G)
    (hμ : IsLeftInvariantIntegral μ)
    (νdata : MeasureTheory.Measure (X × Y))
    (loss : Y → Y → ℝ)
    (hloss : IsInvariantLoss ρY loss)
    (hdiff :
      ∀ θ : Param arch,
        DifferentiableAt ℝ
          (augmentedRisk arch μ νdata loss)
          θ)
    (flow : ℝ → Param arch → Param arch)
    (hflow_measurable :
      ∀ t : ℝ,
        Measurable (flow t))
    (hUnique :
      HasUniqueGradientFlows arch
        (augmentedRisk arch μ νdata loss))
    (hFlow :
      IsGradientFlowMap arch
        (augmentedRisk arch μ νdata loss)
        flow)
    (ν₀ : MeasureTheory.Measure (Param arch))
    (hν₀ :
      IsInvariantParameterDistribution arch ν₀)
    (heval :
      ∀ x : X,
        Measurable
          (fun θ : Param arch =>
            eval arch θ x))
    (hintegrable :
      ∀ (t : ℝ) (g : G) (x : X),
        MeasureTheory.Integrable
          (fun θ : Param arch =>
            eval arch
              (paramAct arch g⁻¹ θ)
              x)
          (parameterDistributionAtTime
            arch ν₀ flow t)) :
    ∀ t : ℝ,
      IsEquivariantMap ρX ρY
        (outputMeanAtTime
          arch ν₀ flow t) := by
  have hflow_equivariant :
      ∀ (t : ℝ) (g : G) (θ₀ : Param arch),
        flow t (paramAct arch g θ₀) =
          paramAct arch g (flow t θ₀) := by
    exact augmentedRisk_gradientFlowMap_equivariant
      (arch := arch)
      (μ := μ)
      (hμ := hμ)
      (ν := νdata)
      (loss := loss)
      (hloss := hloss)
      (hdiff := hdiff)
      (flow := flow)
      (hUnique := hUnique)
      (hFlow := hFlow)
  intro t
  have hνt :
      IsInvariantParameterDistribution arch
        (parameterDistributionAtTime
          arch ν₀ flow t) := by
    exact parameterDistributionAtTime_invariant
      arch
      ν₀
      flow
      hν₀
      hflow_measurable
      hflow_equivariant
      t
  unfold outputMeanAtTime
  exact outputMean_equivariant
    arch
    (parameterDistributionAtTime
      arch ν₀ flow t)
    hνt
    heval
    (hintegrable t)


/-!
## Equivariance of the parameter-mean network

The proof has three steps:

1. The unitary parameter action commutes with the Bochner integral.
2. Invariance of the parameter distribution implies that the mean
   parameter is fixed by every parameter action.
3. Lemma 2.1, applied at the mean parameter, implies equivariance of the
   corresponding network.
-/

/--
The Bochner integral of the transformed parameter equals the Bochner
integral of the original parameter under an invariant distribution:

    ∫ θ, paramAct arch g θ ∂ν
      =
    ∫ θ, θ ∂ν.
-/
theorem integral_paramAct_eq_integral_id
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    [BorelSpace (Param arch)]
    (ν : MeasureTheory.Measure (Param arch))
    (hν : IsInvariantParameterDistribution arch ν)
    (g : G) :
    (∫ θ : Param arch,
        paramAct arch g θ
      ∂ν) =
      ∫ θ : Param arch,
        θ
      ∂ν := by
  have hact :
      Measurable (paramAct arch g) :=
    hν.measurable_paramAct g
  have hmap :
      MeasureTheory.Measure.map
          (paramAct arch g)
          ν =
        ν :=
    hν.map_paramAct g
  have hid_map :
      MeasureTheory.AEStronglyMeasurable
        (fun θ : Param arch => θ)
        (MeasureTheory.Measure.map
          (paramAct arch g)
          ν) := by
    rw [hmap]
    exact measurable_id.aestronglyMeasurable
  have hchange :=
    MeasureTheory.integral_map
      hact.aemeasurable
      hid_map
  rw [hmap] at hchange
  exact hchange.symm

/--
The unitary parameter action commutes with the parameter mean:

    paramAct arch g (∫ θ, θ ∂ν)
      =
    ∫ θ, paramAct arch g θ ∂ν.

This is linearity of the Bochner integral.
-/
theorem paramAct_parameterMean_eq_integral
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (ν : MeasureTheory.Measure (Param arch))
    (g : G)
    (hintegrable :
      MeasureTheory.Integrable
        (fun θ : Param arch => θ)
        ν) :
    paramAct arch g (parameterMean arch ν) =
      ∫ θ : Param arch,
        paramAct arch g θ
      ∂ν := by
  let e : Param arch ≃ₗᵢ[ℝ] Param arch :=
    paramActIsometryEquiv arch g
  let i : Param arch →ₗᵢ[ℝ] Param arch :=
    e.toLinearIsometry
  let L : Param arch →L[ℝ] Param arch :=
    i.toContinuousLinearMap
  have hL_apply :
      ∀ θ : Param arch,
        L θ = paramAct arch g θ := by
    intro θ
    change
      paramActIsometryEquiv arch g θ =
        paramAct arch g θ
    exact paramActIsometryEquiv_apply arch g θ
  unfold parameterMean
  have hcommute :
      (∫ θ : Param arch,
          L θ
        ∂ν) =
        L
          (∫ θ : Param arch,
            θ
          ∂ν) := by
    exact L.integral_comp_comm hintegrable
  calc
    paramAct arch g
        (∫ θ : Param arch, θ ∂ν) =
      L
        (∫ θ : Param arch, θ ∂ν) := by
      symm
      exact hL_apply
        (∫ θ : Param arch, θ ∂ν)
    _ =
      ∫ θ : Param arch,
        L θ
      ∂ν := by
      exact hcommute.symm
    _ =
      ∫ θ : Param arch,
        paramAct arch g θ
      ∂ν := by
      apply MeasureTheory.integral_congr_ae
      filter_upwards with θ
      exact hL_apply θ

/--
The mean parameter of an invariant parameter distribution is fixed by
the unitary parameter action:

    paramAct arch g (parameterMean arch ν)
      =
    parameterMean arch ν.
-/
theorem parameterMean_invariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    [BorelSpace (Param arch)]
    (ν : MeasureTheory.Measure (Param arch))
    (hν : IsInvariantParameterDistribution arch ν)
    (hintegrable :
      MeasureTheory.Integrable
        (fun θ : Param arch => θ)
        ν)
    (g : G) :
    paramAct arch g (parameterMean arch ν) =
      parameterMean arch ν := by
  calc
    paramAct arch g
        (parameterMean arch ν) =
      ∫ θ : Param arch,
        paramAct arch g θ
      ∂ν := by
      exact paramAct_parameterMean_eq_integral
        arch
        ν
        g
        hintegrable
    _ =
      ∫ θ : Param arch,
        θ
      ∂ν := by
      exact integral_paramAct_eq_integral_id
        arch
        ν
        hν
        g
    _ =
      parameterMean arch ν := by
      rfl

/--
Pointwise equivariance of the network evaluated at the mean parameter:

    f_{E[θ]}(ρX(g)x)
      =
    ρY(g)f_{E[θ]}(x).
-/
theorem parameterMeanNetwork_equivariant_pointwise
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    [BorelSpace (Param arch)]
    (ν : MeasureTheory.Measure (Param arch))
    (hν : IsInvariantParameterDistribution arch ν)
    (hintegrable :
      MeasureTheory.Integrable
        (fun θ : Param arch => θ)
        ν)
    (g : G)
    (x : X) :
    parameterMeanNetwork arch ν
        (ρX.act g x) =
      ρY.act g
        (parameterMeanNetwork arch ν x) := by
  unfold parameterMeanNetwork
  calc
    eval arch
        (parameterMean arch ν)
        (ρX.act g x) =
      ρY.act g
        (eval arch
          (paramAct arch g⁻¹
            (parameterMean arch ν))
          x) := by
      exact lemma_2_1
        arch
        (parameterMean arch ν)
        g
        x
    _ =
      ρY.act g
        (eval arch
          (parameterMean arch ν)
          x) := by
      rw [
        parameterMean_invariant
          arch
          ν
          hν
          hintegrable
          g⁻¹
      ]

/--
The network obtained by evaluating the architecture at the mean
parameter is equivariant.
-/
theorem parameterMeanNetwork_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    [BorelSpace (Param arch)]
    (ν : MeasureTheory.Measure (Param arch))
    (hν : IsInvariantParameterDistribution arch ν)
    (hintegrable :
      MeasureTheory.Integrable
        (fun θ : Param arch => θ)
        ν) :
    IsEquivariantMap ρX ρY
      (parameterMeanNetwork arch ν) := by
  intro g x
  symm
  exact parameterMeanNetwork_equivariant_pointwise
    arch
    ν
    hν
    hintegrable
    g
    x

/--
At every time `t`, the mean parameter is fixed by the group action,
provided the time-dependent parameter distribution is invariant and has
an integrable identity map.
-/
theorem parameterMeanAtTime_invariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    [BorelSpace (Param arch)]
    (ν₀ : MeasureTheory.Measure (Param arch))
    (flow : ℝ → Param arch → Param arch)
    (hν₀ :
      IsInvariantParameterDistribution arch ν₀)
    (hflow_measurable :
      ∀ t : ℝ,
        Measurable (flow t))
    (hflow_equivariant :
      ∀ (t : ℝ) (g : G) (θ : Param arch),
        flow t (paramAct arch g θ) =
          paramAct arch g (flow t θ))
    (hintegrable :
      ∀ t : ℝ,
        MeasureTheory.Integrable
          (fun θ : Param arch => θ)
          (parameterDistributionAtTime
            arch ν₀ flow t))
    (t : ℝ)
    (g : G) :
    paramAct arch g
        (parameterMeanAtTime
          arch ν₀ flow t) =
      parameterMeanAtTime
        arch ν₀ flow t := by
  have hνt :
      IsInvariantParameterDistribution arch
        (parameterDistributionAtTime
          arch ν₀ flow t) :=
    parameterDistributionAtTime_invariant
      arch
      ν₀
      flow
      hν₀
      hflow_measurable
      hflow_equivariant
      t
  unfold parameterMeanAtTime
  exact parameterMean_invariant
    arch
    (parameterDistributionAtTime
      arch ν₀ flow t)
    hνt
    (hintegrable t)
    g

/--
At every time `t`, the network evaluated at the mean parameter is
equivariant.
-/
theorem parameterMeanNetworkAtTime_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    [BorelSpace (Param arch)]
    (ν₀ : MeasureTheory.Measure (Param arch))
    (flow : ℝ → Param arch → Param arch)
    (hν₀ :
      IsInvariantParameterDistribution arch ν₀)
    (hflow_measurable :
      ∀ t : ℝ,
        Measurable (flow t))
    (hflow_equivariant :
      ∀ (t : ℝ) (g : G) (θ : Param arch),
        flow t (paramAct arch g θ) =
          paramAct arch g (flow t θ))
    (hintegrable :
      ∀ t : ℝ,
        MeasureTheory.Integrable
          (fun θ : Param arch => θ)
          (parameterDistributionAtTime
            arch ν₀ flow t)) :
    ∀ t : ℝ,
      IsEquivariantMap ρX ρY
        (parameterMeanNetworkAtTime
          arch ν₀ flow t) := by
  intro t
  have hνt :
      IsInvariantParameterDistribution arch
        (parameterDistributionAtTime
          arch ν₀ flow t) :=
    parameterDistributionAtTime_invariant
      arch
      ν₀
      flow
      hν₀
      hflow_measurable
      hflow_equivariant
      t
  unfold parameterMeanNetworkAtTime
  exact parameterMeanNetwork_equivariant
    arch
    (parameterDistributionAtTime
      arch ν₀ flow t)
    hνt
    (hintegrable t)

/--
Under gradient flow of the augmented risk, the network evaluated at the
mean parameter is equivariant at every time.

Flow equivariance is derived from augmented-risk invariance, unitarity
of the parameter action, gradient equivariance, and uniqueness of the
gradient-flow solution. It is not assumed separately.
-/
theorem augmentedRisk_parameterMeanNetworkAtTime_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace G]
    [MeasurableSpace X]
    [MeasurableSpace Y]
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    [BorelSpace (Param arch)]
    (μ : MeasureTheory.Measure G)
    (hμ : IsLeftInvariantIntegral μ)
    (νdata : MeasureTheory.Measure (X × Y))
    (loss : Y → Y → ℝ)
    (hloss : IsInvariantLoss ρY loss)
    (hdiff :
      ∀ θ : Param arch,
        DifferentiableAt ℝ
          (augmentedRisk arch μ νdata loss)
          θ)
    (flow : ℝ → Param arch → Param arch)
    (hflow_measurable :
      ∀ t : ℝ,
        Measurable (flow t))
    (hUnique :
      HasUniqueGradientFlows arch
        (augmentedRisk arch μ νdata loss))
    (hFlow :
      IsGradientFlowMap arch
        (augmentedRisk arch μ νdata loss)
        flow)
    (ν₀ : MeasureTheory.Measure (Param arch))
    (hν₀ :
      IsInvariantParameterDistribution arch ν₀)
    (hintegrable :
      ∀ t : ℝ,
        MeasureTheory.Integrable
          (fun θ : Param arch => θ)
          (parameterDistributionAtTime
            arch ν₀ flow t)) :
    ∀ t : ℝ,
      IsEquivariantMap ρX ρY
        (parameterMeanNetworkAtTime
          arch ν₀ flow t) := by
  have hflow_equivariant :
      ∀ (t : ℝ) (g : G) (θ₀ : Param arch),
        flow t (paramAct arch g θ₀) =
          paramAct arch g (flow t θ₀) := by
    exact augmentedRisk_gradientFlowMap_equivariant
      arch
      μ
      hμ
      νdata
      loss
      hloss
      hdiff
      flow
      hUnique
      hFlow
  intro t
  have hνt :
      IsInvariantParameterDistribution arch
        (parameterDistributionAtTime
          arch ν₀ flow t) := by
    exact parameterDistributionAtTime_invariant
      arch
      ν₀
      flow
      hν₀
      hflow_measurable
      hflow_equivariant
      t
  unfold parameterMeanNetworkAtTime
  exact parameterMeanNetwork_equivariant
    arch
    (parameterDistributionAtTime
      arch ν₀ flow t)
    hνt
    (hintegrable t)

/-!
## Gradient flow constrained to an invariant affine subspace

Let `Aff` be an affine subspace of the full parameter Hilbert space.

Its tangent space is the linear subspace `Aff.direction`. Constrained
gradient flow uses the vector field

    θ ↦ -projectTangent (gradient risk θ).

If the affine subspace is invariant under the unitary parameter action,
then its tangent space is invariant and its orthogonal projection
commutes with the parameter action.
-/

/--
Data describing an invariant affine constraint on the full parameter
space.

The continuous linear map `projectTangent` represents orthogonal
projection onto the tangent space `Aff.direction`.

The three projection laws state:

* projected vectors belong to the tangent space;
* tangent vectors are fixed by the projection;
* projection commutes with the unitary parameter action.

In finite-dimensional Hilbert spaces, these properties are satisfied by
orthogonal projection onto the direction of an invariant affine
subspace.
-/
structure InvariantAffineConstraint
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY) where

  /--
  The affine subset in which the parameters are constrained.
  -/
  Aff : AffineSubspace ℝ (Param arch)

  /--
  The affine subspace is invariant under the parameter action.
  -/
  invariant :
    ∀ (g : G) (θ : Param arch),
      θ ∈ Aff →
        paramAct arch g θ ∈ Aff

  /--
  Orthogonal projection onto the tangent space `Aff.direction`.
  -/
  projectTangent :
    Param arch →L[ℝ] Param arch

  /--
  The projected vector lies in the tangent space.
  -/
  projectTangent_mem :
    ∀ v : Param arch,
      projectTangent v ∈ Aff.direction

  /--
  Projection fixes vectors already in the tangent space.
  -/
  projectTangent_eq_self :
    ∀ v : Param arch,
      v ∈ Aff.direction →
        projectTangent v = v

  /--
  Orthogonal projection onto the invariant tangent space commutes with
  the unitary parameter action.
  -/
  projectTangent_commutes :
    ∀ (g : G) (v : Param arch),
      projectTangent (paramAct arch g v) =
        paramAct arch g (projectTangent v)

namespace InvariantAffineConstraint

variable
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    {arch : Architecture G X ρX Y ρY}

/--
The inverse parameter action also preserves the affine constraint.
-/
theorem invariant_inv
    (constraint : InvariantAffineConstraint arch)
    (g : G)
    (θ : Param arch)
    (hθ : θ ∈ constraint.Aff) :
    paramAct arch g⁻¹ θ ∈ constraint.Aff :=
  constraint.invariant g⁻¹ θ hθ

/--
The tangent projection is idempotent.
-/
theorem projectTangent_idempotent
    (constraint : InvariantAffineConstraint arch)
    (v : Param arch) :
    constraint.projectTangent
        (constraint.projectTangent v) =
      constraint.projectTangent v := by
  exact constraint.projectTangent_eq_self
    (constraint.projectTangent v)
    (constraint.projectTangent_mem v)

end InvariantAffineConstraint

/-!
## Projected negative-gradient field
-/

/--
The negative projected-gradient vector field:

    θ ↦ -projectTangent(gradient risk θ).

Since the tangent projection is linear, this is equivalently

    θ ↦ projectTangent(-gradient risk θ).
-/
noncomputable def projectedNegativeGradientField
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (constraint : InvariantAffineConstraint arch)
    (risk : Param arch → ℝ) :
    Param arch → Param arch :=
  fun θ =>
    -constraint.projectTangent
      (gradient risk θ)

/--
The projected negative-gradient vector lies in the tangent space.
-/
theorem projectedNegativeGradientField_mem_direction
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (constraint : InvariantAffineConstraint arch)
    (risk : Param arch → ℝ)
    (θ : Param arch) :
    projectedNegativeGradientField
        arch constraint risk θ ∈
      constraint.Aff.direction := by
  unfold projectedNegativeGradientField
  have hmem :
      constraint.projectTangent
          (gradient risk θ) ∈
        constraint.Aff.direction :=
    constraint.projectTangent_mem
      (gradient risk θ)
  exact constraint.Aff.direction.neg_mem hmem

/--
If the gradient is equivariant and the tangent projection commutes with
the parameter action, then the projected negative-gradient field is
equivariant.
-/
theorem projectedNegativeGradientField_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (constraint : InvariantAffineConstraint arch)
    (risk : Param arch → ℝ)
    (hgrad :
      IsEquivariantParameterGradient arch risk) :
    ∀ (g : G) (θ : Param arch),
      projectedNegativeGradientField
          arch constraint risk
          (paramAct arch g θ) =
        paramAct arch g
          (projectedNegativeGradientField
            arch constraint risk θ) := by
  intro g θ
  unfold projectedNegativeGradientField
  rw [hgrad g θ]
  rw [constraint.projectTangent_commutes g]
  rw [paramAct_neg]

/--
The projected negative-gradient field of the augmented risk is
equivariant.
-/
theorem augmentedRisk_projectedNegativeGradientField_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace G]
    [MeasurableSpace X]
    [MeasurableSpace Y]
    (arch : Architecture G X ρX Y ρY)
    (constraint : InvariantAffineConstraint arch)
    (μ : MeasureTheory.Measure G)
    (hμ : IsLeftInvariantIntegral μ)
    (νdata : MeasureTheory.Measure (X × Y))
    (loss : Y → Y → ℝ)
    (hloss : IsInvariantLoss ρY loss)
    (hdiff :
      ∀ θ : Param arch,
        DifferentiableAt ℝ
          (augmentedRisk arch μ νdata loss)
          θ) :
    ∀ (g : G) (θ : Param arch),
      projectedNegativeGradientField
          arch
          constraint
          (augmentedRisk arch μ νdata loss)
          (paramAct arch g θ) =
        paramAct arch g
          (projectedNegativeGradientField
            arch
            constraint
            (augmentedRisk arch μ νdata loss)
            θ) := by
  have hgrad :
      IsEquivariantParameterGradient arch
        (augmentedRisk arch μ νdata loss) :=
    augmentedRisk_gradient_equivariant
      arch
      μ
      hμ
      νdata
      loss
      hloss
      hdiff
  exact projectedNegativeGradientField_equivariant
    arch
    constraint
    (augmentedRisk arch μ νdata loss)
    hgrad

/-!
## Projected gradient-flow solutions
-/

/--
A solution of gradient flow constrained to the affine subspace.

The curve:

* starts at `θ₀`;
* remains in `constraint.Aff`;
* satisfies the projected gradient-flow equation

      θ'(t) =
        -projectTangent(gradient risk (θ(t))).
-/
def IsProjectedGradientFlowSolution
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (constraint : InvariantAffineConstraint arch)
    (risk : Param arch → ℝ)
    (curve : ℝ → Param arch)
    (θ₀ : Param arch) : Prop :=
  curve 0 = θ₀ ∧
    (∀ t : ℝ, curve t ∈ constraint.Aff) ∧
    ∀ t : ℝ,
      HasDerivAt curve
        (projectedNegativeGradientField
          arch constraint risk (curve t))
        t

/--
The projected gradient-flow equation is equivariant if transforming a
solution curve gives another solution from the transformed initial
parameter.
-/
def IsEquivariantProjectedGradientFlow
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (constraint : InvariantAffineConstraint arch)
    (risk : Param arch → ℝ) : Prop :=
  ∀ (g : G)
      (curve : ℝ → Param arch)
      (θ₀ : Param arch),
    IsProjectedGradientFlowSolution
        arch constraint risk curve θ₀ →
      IsProjectedGradientFlowSolution
        arch
        constraint
        risk
        (fun t => paramAct arch g (curve t))
        (paramAct arch g θ₀)

/--
Uniqueness of projected gradient-flow solutions.
-/
def HasUniqueProjectedGradientFlows
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (constraint : InvariantAffineConstraint arch)
    (risk : Param arch → ℝ) : Prop :=
  ∀ (curve₁ curve₂ : ℝ → Param arch)
      (θ₀ : Param arch),
    IsProjectedGradientFlowSolution
        arch constraint risk curve₁ θ₀ →
    IsProjectedGradientFlowSolution
        arch constraint risk curve₂ θ₀ →
    curve₁ = curve₂

/--
A projected gradient-flow map realizes the projected-gradient solution
from every initial parameter in the affine constraint.
-/
def IsProjectedGradientFlowMap
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (constraint : InvariantAffineConstraint arch)
    (risk : Param arch → ℝ)
    (flow : ℝ → Param arch → Param arch) : Prop :=
  ∀ θ₀ : Param arch,
    θ₀ ∈ constraint.Aff →
      IsProjectedGradientFlowSolution
        arch
        constraint
        risk
        (fun t => flow t θ₀)
        θ₀

/--
An equivariant projected vector field makes the projected-gradient-flow
solution predicate equivariant.
-/
theorem projectedGradientFlowSolution_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (constraint : InvariantAffineConstraint arch)
    (risk : Param arch → ℝ)
    (hfield :
      ∀ (g : G) (θ : Param arch),
        projectedNegativeGradientField
            arch constraint risk
            (paramAct arch g θ) =
          paramAct arch g
            (projectedNegativeGradientField
              arch constraint risk θ)) :
    IsEquivariantProjectedGradientFlow
      arch constraint risk := by
  intro g curve θ₀ hcurve
  constructor
  · change
      paramAct arch g (curve 0) =
        paramAct arch g θ₀
    rw [hcurve.1]
  · constructor
    · intro t
      exact constraint.invariant
        g
        (curve t)
        (hcurve.2.1 t)
    · intro t
      let e : Param arch ≃ₗᵢ[ℝ] Param arch :=
        paramActIsometryEquiv arch g
      let L : Param arch →L[ℝ] Param arch :=
        e.toLinearIsometry.toContinuousLinearMap
      have hL_apply :
          ∀ θ : Param arch,
            L θ = paramAct arch g θ := by
        intro θ
        change
          paramActIsometryEquiv arch g θ =
            paramAct arch g θ
        exact paramActIsometryEquiv_apply
          arch g θ
      have hL :
          HasFDerivAt
            (fun θ : Param arch => L θ)
            L
            (curve t) :=
        L.hasFDerivAt
      have hcomp :=
        hL.comp_hasDerivAt t (hcurve.2.2 t)
      change
        HasDerivAt
          (fun s : ℝ => L (curve s))
          (L
            (projectedNegativeGradientField
              arch constraint risk (curve t)))
          t at hcomp
      have hderiv :
          HasDerivAt
            (fun s : ℝ =>
              paramAct arch g (curve s))
            (paramAct arch g
              (projectedNegativeGradientField
                arch constraint risk (curve t)))
            t := by
        convert hcomp using 1
        · funext s
          exact hL_apply (curve s)
        · exact hL_apply
            (projectedNegativeGradientField
              arch constraint risk (curve t))
      change
        HasDerivAt
          (fun s : ℝ =>
            paramAct arch g (curve s))
          (projectedNegativeGradientField
            arch constraint risk
            (paramAct arch g (curve t)))
          t
      rw [hfield g (curve t)]
      exact hderiv

/--
Equivariance of the augmented-risk gradient implies equivariance of the
projected augmented-risk gradient-flow equation.
-/
theorem augmentedRisk_projectedGradientFlowEquation_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace G]
    [MeasurableSpace X]
    [MeasurableSpace Y]
    (arch : Architecture G X ρX Y ρY)
    (constraint : InvariantAffineConstraint arch)
    (μ : MeasureTheory.Measure G)
    (hμ : IsLeftInvariantIntegral μ)
    (νdata : MeasureTheory.Measure (X × Y))
    (loss : Y → Y → ℝ)
    (hloss : IsInvariantLoss ρY loss)
    (hdiff :
      ∀ θ : Param arch,
        DifferentiableAt ℝ
          (augmentedRisk arch μ νdata loss)
          θ) :
    IsEquivariantProjectedGradientFlow
      arch
      constraint
      (augmentedRisk arch μ νdata loss) := by
  exact projectedGradientFlowSolution_equivariant
    arch
    constraint
    (augmentedRisk arch μ νdata loss)
    (augmentedRisk_projectedNegativeGradientField_equivariant
      arch
      constraint
      μ
      hμ
      νdata
      loss
      hloss
      hdiff)

/--
Uniqueness of projected gradient-flow solutions implies equivariance of
the projected gradient-flow map on the invariant affine subspace.
-/
theorem projectedGradientFlowMap_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    (constraint : InvariantAffineConstraint arch)
    (risk : Param arch → ℝ)
    (flow : ℝ → Param arch → Param arch)
    (hEquation :
      IsEquivariantProjectedGradientFlow
        arch constraint risk)
    (hUnique :
      HasUniqueProjectedGradientFlows
        arch constraint risk)
    (hFlow :
      IsProjectedGradientFlowMap
        arch constraint risk flow) :
    ∀ (t : ℝ) (g : G) (θ₀ : Param arch),
      θ₀ ∈ constraint.Aff →
        flow t (paramAct arch g θ₀) =
          paramAct arch g (flow t θ₀) := by
  intro t g θ₀ hθ₀
  have hgθ₀ :
      paramAct arch g θ₀ ∈ constraint.Aff :=
    constraint.invariant g θ₀ hθ₀
  have hDirect :
      IsProjectedGradientFlowSolution
        arch
        constraint
        risk
        (fun s =>
          flow s (paramAct arch g θ₀))
        (paramAct arch g θ₀) :=
    hFlow
      (paramAct arch g θ₀)
      hgθ₀
  have hOriginal :
      IsProjectedGradientFlowSolution
        arch
        constraint
        risk
        (fun s => flow s θ₀)
        θ₀ :=
    hFlow θ₀ hθ₀
  have hTransformed :
      IsProjectedGradientFlowSolution
        arch
        constraint
        risk
        (fun s =>
          paramAct arch g (flow s θ₀))
        (paramAct arch g θ₀) :=
    hEquation
      g
      (fun s => flow s θ₀)
      θ₀
      hOriginal
  have hCurves :
      (fun s =>
        flow s (paramAct arch g θ₀)) =
        fun s =>
          paramAct arch g (flow s θ₀) :=
    hUnique
      (fun s =>
        flow s (paramAct arch g θ₀))
      (fun s =>
        paramAct arch g (flow s θ₀))
      (paramAct arch g θ₀)
      hDirect
      hTransformed
  exact congrFun hCurves t

/--
The projected augmented-risk gradient-flow map is equivariant on the
invariant affine parameter subspace.
-/
theorem augmentedRisk_projectedGradientFlowMap_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace G]
    [MeasurableSpace X]
    [MeasurableSpace Y]
    (arch : Architecture G X ρX Y ρY)
    (constraint : InvariantAffineConstraint arch)
    (μ : MeasureTheory.Measure G)
    (hμ : IsLeftInvariantIntegral μ)
    (νdata : MeasureTheory.Measure (X × Y))
    (loss : Y → Y → ℝ)
    (hloss : IsInvariantLoss ρY loss)
    (hdiff :
      ∀ θ : Param arch,
        DifferentiableAt ℝ
          (augmentedRisk arch μ νdata loss)
          θ)
    (flow : ℝ → Param arch → Param arch)
    (hUnique :
      HasUniqueProjectedGradientFlows
        arch
        constraint
        (augmentedRisk arch μ νdata loss))
    (hFlow :
      IsProjectedGradientFlowMap
        arch
        constraint
        (augmentedRisk arch μ νdata loss)
        flow) :
    ∀ (t : ℝ) (g : G) (θ₀ : Param arch),
      θ₀ ∈ constraint.Aff →
        flow t (paramAct arch g θ₀) =
          paramAct arch g (flow t θ₀) := by
  have hEquation :
      IsEquivariantProjectedGradientFlow
        arch
        constraint
        (augmentedRisk arch μ νdata loss) :=
    augmentedRisk_projectedGradientFlowEquation_equivariant
      arch
      constraint
      μ
      hμ
      νdata
      loss
      hloss
      hdiff
  exact projectedGradientFlowMap_equivariant
    arch
    constraint
    (augmentedRisk arch μ νdata loss)
    flow
    hEquation
    hUnique
    hFlow

/--
A parameter measure is supported on the affine constraint.
-/
def IsSupportedOnAffineConstraint
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (constraint : InvariantAffineConstraint arch)
    (ν : MeasureTheory.Measure (Param arch)) :
    Prop :=
  ∀ᵐ θ ∂ν, θ ∈ constraint.Aff

/--
If the initial measure is invariant and supported on the invariant
affine constraint, then an equivariant measurable flow preserves
invariance of the parameter distribution.

Flow equivariance is required only for initial parameters belonging to
the affine constraint.
-/
theorem parameterDistributionAtTime_invariant_onAffine
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (constraint : InvariantAffineConstraint arch)
    (ν₀ : MeasureTheory.Measure (Param arch))
    (flow : ℝ → Param arch → Param arch)
    (hν₀ :
      IsInvariantParameterDistribution arch ν₀)
    (hν₀_supported :
      IsSupportedOnAffineConstraint
        arch constraint ν₀)
    (hflow_measurable :
      ∀ t : ℝ,
        Measurable (flow t))
    (hflow_equivariant :
      ∀ (t : ℝ) (g : G) (θ : Param arch),
        θ ∈ constraint.Aff →
          flow t (paramAct arch g θ) =
            paramAct arch g (flow t θ))
    (t : ℝ) :
    IsInvariantParameterDistribution arch
      (parameterDistributionAtTime
        arch ν₀ flow t) := by
  constructor
  · intro g
    exact hν₀.measurable_paramAct g
  · intro g
    unfold parameterDistributionAtTime
    have hact :
        Measurable (paramAct arch g) :=
      hν₀.measurable_paramAct g
    have hflow :
        Measurable (flow t) :=
      hflow_measurable t
    have hcommute_ae :
        (paramAct arch g) ∘ (flow t) =ᵐ[ν₀]
          (flow t) ∘ (paramAct arch g) := by
      filter_upwards [hν₀_supported] with θ hθ
      exact (hflow_equivariant t g θ hθ).symm
    calc
      MeasureTheory.Measure.map
          (paramAct arch g)
          (MeasureTheory.Measure.map
            (flow t)
            ν₀) =
        MeasureTheory.Measure.map
          ((paramAct arch g) ∘ (flow t))
          ν₀ := by
        exact MeasureTheory.Measure.map_map
          hact
          hflow
      _ =
        MeasureTheory.Measure.map
          ((flow t) ∘ (paramAct arch g))
          ν₀ := by
        exact MeasureTheory.Measure.map_congr
          hcommute_ae
      _ =
        MeasureTheory.Measure.map
          (flow t)
          (MeasureTheory.Measure.map
            (paramAct arch g)
            ν₀) := by
        symm
        exact MeasureTheory.Measure.map_map
          hflow
          hact
      _ =
        MeasureTheory.Measure.map
          (flow t)
          ν₀ := by
        rw [hν₀.map_paramAct g]

/--
Under projected gradient flow of the augmented risk, an invariant
initial parameter distribution supported on the invariant affine
constraint remains invariant at every time.
-/
theorem augmentedRisk_projectedGradientFlow_preserves_invariantDistribution
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace G]
    [MeasurableSpace X]
    [MeasurableSpace Y]
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    (constraint : InvariantAffineConstraint arch)
    (μ : MeasureTheory.Measure G)
    (hμ : IsLeftInvariantIntegral μ)
    (νdata : MeasureTheory.Measure (X × Y))
    (loss : Y → Y → ℝ)
    (hloss : IsInvariantLoss ρY loss)
    (hdiff :
      ∀ θ : Param arch,
        DifferentiableAt ℝ
          (augmentedRisk arch μ νdata loss)
          θ)
    (flow : ℝ → Param arch → Param arch)
    (hflow_measurable :
      ∀ t : ℝ,
        Measurable (flow t))
    (hUnique :
      HasUniqueProjectedGradientFlows
        arch
        constraint
        (augmentedRisk arch μ νdata loss))
    (hFlow :
      IsProjectedGradientFlowMap
        arch
        constraint
        (augmentedRisk arch μ νdata loss)
        flow)
    (ν₀ : MeasureTheory.Measure (Param arch))
    (hν₀ :
      IsInvariantParameterDistribution arch ν₀)
    (hν₀_supported :
      IsSupportedOnAffineConstraint
        arch constraint ν₀) :
    ∀ t : ℝ,
      IsInvariantParameterDistribution arch
        (parameterDistributionAtTime
          arch ν₀ flow t) := by
  have hflow_equivariant :
      ∀ (t : ℝ) (g : G) (θ₀ : Param arch),
        θ₀ ∈ constraint.Aff →
          flow t (paramAct arch g θ₀) =
            paramAct arch g (flow t θ₀) := by
    exact augmentedRisk_projectedGradientFlowMap_equivariant
      arch
      constraint
      μ
      hμ
      νdata
      loss
      hloss
      hdiff
      flow
      hUnique
      hFlow
  intro t
  exact parameterDistributionAtTime_invariant_onAffine
    arch
    constraint
    ν₀
    flow
    hν₀
    hν₀_supported
    hflow_measurable
    hflow_equivariant
    t

/--
Under projected gradient flow of the augmented risk, the network
evaluated at the mean parameter is equivariant at every time.

The initial parameter distribution is assumed to be:

* invariant under the parameter action;
* supported on the invariant affine constraint.

Equivariance of the projected gradient-flow map is derived from
augmented-risk invariance, gradient equivariance, commutation of the
tangent projection with the parameter action, and uniqueness of the
projected gradient-flow solution.
-/
theorem augmentedRisk_projectedGradientFlow_parameterMeanNetworkAtTime_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace G]
    [MeasurableSpace X]
    [MeasurableSpace Y]
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    [BorelSpace (Param arch)]
    (constraint : InvariantAffineConstraint arch)
    (μ : MeasureTheory.Measure G)
    (hμ : IsLeftInvariantIntegral μ)
    (νdata : MeasureTheory.Measure (X × Y))
    (loss : Y → Y → ℝ)
    (hloss : IsInvariantLoss ρY loss)
    (hdiff :
      ∀ θ : Param arch,
        DifferentiableAt ℝ
          (augmentedRisk arch μ νdata loss)
          θ)
    (flow : ℝ → Param arch → Param arch)
    (hflow_measurable :
      ∀ t : ℝ,
        Measurable (flow t))
    (hUnique :
      HasUniqueProjectedGradientFlows
        arch
        constraint
        (augmentedRisk arch μ νdata loss))
    (hFlow :
      IsProjectedGradientFlowMap
        arch
        constraint
        (augmentedRisk arch μ νdata loss)
        flow)
    (ν₀ : MeasureTheory.Measure (Param arch))
    (hν₀ :
      IsInvariantParameterDistribution arch ν₀)
    (hν₀_supported :
      IsSupportedOnAffineConstraint
        arch constraint ν₀)
    (hintegrable :
      ∀ t : ℝ,
        MeasureTheory.Integrable
          (fun θ : Param arch => θ)
          (parameterDistributionAtTime
            arch ν₀ flow t)) :
    ∀ t : ℝ,
      IsEquivariantMap ρX ρY
        (parameterMeanNetworkAtTime
          arch ν₀ flow t) := by
  have hdistribution_invariant :
      ∀ t : ℝ,
        IsInvariantParameterDistribution arch
          (parameterDistributionAtTime
            arch ν₀ flow t) := by
    exact
      augmentedRisk_projectedGradientFlow_preserves_invariantDistribution
        arch
        constraint
        μ
        hμ
        νdata
        loss
        hloss
        hdiff
        flow
        hflow_measurable
        hUnique
        hFlow
        ν₀
        hν₀
        hν₀_supported
  intro t
  unfold parameterMeanNetworkAtTime
  exact parameterMeanNetwork_equivariant
    arch
    (parameterDistributionAtTime
      arch ν₀ flow t)
    (hdistribution_invariant t)
    (hintegrable t)

/--
Under projected augmented-risk gradient flow, the mean parameter at
every time is fixed by the parameter action.
-/
theorem augmentedRisk_projectedGradientFlow_parameterMeanAtTime_invariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace G]
    [MeasurableSpace X]
    [MeasurableSpace Y]
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    [BorelSpace (Param arch)]
    (constraint : InvariantAffineConstraint arch)
    (μ : MeasureTheory.Measure G)
    (hμ : IsLeftInvariantIntegral μ)
    (νdata : MeasureTheory.Measure (X × Y))
    (loss : Y → Y → ℝ)
    (hloss : IsInvariantLoss ρY loss)
    (hdiff :
      ∀ θ : Param arch,
        DifferentiableAt ℝ
          (augmentedRisk arch μ νdata loss)
          θ)
    (flow : ℝ → Param arch → Param arch)
    (hflow_measurable :
      ∀ t : ℝ,
        Measurable (flow t))
    (hUnique :
      HasUniqueProjectedGradientFlows
        arch
        constraint
        (augmentedRisk arch μ νdata loss))
    (hFlow :
      IsProjectedGradientFlowMap
        arch
        constraint
        (augmentedRisk arch μ νdata loss)
        flow)
    (ν₀ : MeasureTheory.Measure (Param arch))
    (hν₀ :
      IsInvariantParameterDistribution arch ν₀)
    (hν₀_supported :
      IsSupportedOnAffineConstraint
        arch constraint ν₀)
    (hintegrable :
      ∀ t : ℝ,
        MeasureTheory.Integrable
          (fun θ : Param arch => θ)
          (parameterDistributionAtTime
            arch ν₀ flow t)) :
    ∀ (t : ℝ) (g : G),
      paramAct arch g
          (parameterMeanAtTime
            arch ν₀ flow t) =
        parameterMeanAtTime
          arch ν₀ flow t := by
  have hdistribution_invariant :
      ∀ t : ℝ,
        IsInvariantParameterDistribution arch
          (parameterDistributionAtTime
            arch ν₀ flow t) := by
    exact
      augmentedRisk_projectedGradientFlow_preserves_invariantDistribution
        arch
        constraint
        μ
        hμ
        νdata
        loss
        hloss
        hdiff
        flow
        hflow_measurable
        hUnique
        hFlow
        ν₀
        hν₀
        hν₀_supported
  intro t g
  unfold parameterMeanAtTime
  exact parameterMean_invariant
    arch
    (parameterDistributionAtTime
      arch ν₀ flow t)
    (hdistribution_invariant t)
    (hintegrable t)
    g

/--
Under projected gradient flow of the augmented risk, the pointwise
output mean is equivariant at every time.

The initial parameter distribution is assumed to be:

* invariant under the parameter action;
* supported on the invariant affine constraint.

Equivariance of the projected gradient-flow map is derived rather than
assumed. It follows from augmented-risk invariance, gradient
equivariance, commutation of the tangent projection with the parameter
action, and uniqueness of projected gradient-flow solutions.
-/
theorem augmentedRisk_projectedGradientFlow_outputMeanAtTime_equivariant
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace G]
    [MeasurableSpace X]
    [MeasurableSpace Y]
    [BorelSpace Y]
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    [BorelSpace (Param arch)]
    (constraint : InvariantAffineConstraint arch)
    (μ : MeasureTheory.Measure G)
    (hμ : IsLeftInvariantIntegral μ)
    (νdata : MeasureTheory.Measure (X × Y))
    (loss : Y → Y → ℝ)
    (hloss : IsInvariantLoss ρY loss)
    (hdiff :
      ∀ θ : Param arch,
        DifferentiableAt ℝ
          (augmentedRisk arch μ νdata loss)
          θ)
    (flow : ℝ → Param arch → Param arch)
    (hflow_measurable :
      ∀ t : ℝ,
        Measurable (flow t))
    (hUnique :
      HasUniqueProjectedGradientFlows
        arch
        constraint
        (augmentedRisk arch μ νdata loss))
    (hFlow :
      IsProjectedGradientFlowMap
        arch
        constraint
        (augmentedRisk arch μ νdata loss)
        flow)
    (ν₀ : MeasureTheory.Measure (Param arch))
    (hν₀ :
      IsInvariantParameterDistribution arch ν₀)
    (hν₀_supported :
      IsSupportedOnAffineConstraint
        arch constraint ν₀)
    (heval :
      ∀ x : X,
        Measurable
          (fun θ : Param arch =>
            eval arch θ x))
    (hintegrable :
      ∀ (t : ℝ) (g : G) (x : X),
        MeasureTheory.Integrable
          (fun θ : Param arch =>
            eval arch
              (paramAct arch g⁻¹ θ)
              x)
          (parameterDistributionAtTime
            arch ν₀ flow t)) :
    ∀ t : ℝ,
      IsEquivariantMap ρX ρY
        (outputMeanAtTime
          arch ν₀ flow t) := by
  have hdistribution_invariant :
      ∀ t : ℝ,
        IsInvariantParameterDistribution arch
          (parameterDistributionAtTime
            arch ν₀ flow t) := by
    exact
      augmentedRisk_projectedGradientFlow_preserves_invariantDistribution
        arch
        constraint
        μ
        hμ
        νdata
        loss
        hloss
        hdiff
        flow
        hflow_measurable
        hUnique
        hFlow
        ν₀
        hν₀
        hν₀_supported
  intro t
  unfold outputMeanAtTime
  exact outputMean_equivariant
    arch
    (parameterDistributionAtTime
      arch ν₀ flow t)
    (hdistribution_invariant t)
    heval
    (hintegrable t)

/--
Pointwise form of equivariance of the output mean under projected
augmented-risk gradient flow:

    outputMeanAtTime(t, ρX(g)x)
      =
    ρY(g)(outputMeanAtTime(t, x)).
-/
theorem augmentedRisk_projectedGradientFlow_outputMeanAtTime_equivariant_pointwise
    {X Y : FDHilbert.{uV}}
    {ρX : UnitaryRepresentation G X}
    {ρY : UnitaryRepresentation G Y}
    [MeasurableSpace G]
    [MeasurableSpace X]
    [MeasurableSpace Y]
    [BorelSpace Y]
    (arch : Architecture G X ρX Y ρY)
    [MeasurableSpace (Param arch)]
    [BorelSpace (Param arch)]
    (constraint : InvariantAffineConstraint arch)
    (μ : MeasureTheory.Measure G)
    (hμ : IsLeftInvariantIntegral μ)
    (νdata : MeasureTheory.Measure (X × Y))
    (loss : Y → Y → ℝ)
    (hloss : IsInvariantLoss ρY loss)
    (hdiff :
      ∀ θ : Param arch,
        DifferentiableAt ℝ
          (augmentedRisk arch μ νdata loss)
          θ)
    (flow : ℝ → Param arch → Param arch)
    (hflow_measurable :
      ∀ t : ℝ,
        Measurable (flow t))
    (hUnique :
      HasUniqueProjectedGradientFlows
        arch
        constraint
        (augmentedRisk arch μ νdata loss))
    (hFlow :
      IsProjectedGradientFlowMap
        arch
        constraint
        (augmentedRisk arch μ νdata loss)
        flow)
    (ν₀ : MeasureTheory.Measure (Param arch))
    (hν₀ :
      IsInvariantParameterDistribution arch ν₀)
    (hν₀_supported :
      IsSupportedOnAffineConstraint
        arch constraint ν₀)
    (heval :
      ∀ x : X,
        Measurable
          (fun θ : Param arch =>
            eval arch θ x))
    (hintegrable :
      ∀ (t : ℝ) (g : G) (x : X),
        MeasureTheory.Integrable
          (fun θ : Param arch =>
            eval arch
              (paramAct arch g⁻¹ θ)
              x)
          (parameterDistributionAtTime
            arch ν₀ flow t))
    (t : ℝ)
    (g : G)
    (x : X) :
    outputMeanAtTime arch ν₀ flow t
        (ρX.act g x) =
      ρY.act g
        (outputMeanAtTime arch ν₀ flow t x) := by
  have hequiv :
      IsEquivariantMap ρX ρY
        (outputMeanAtTime
          arch ν₀ flow t) := by
    exact
      augmentedRisk_projectedGradientFlow_outputMeanAtTime_equivariant
        arch
        constraint
        μ
        hμ
        νdata
        loss
        hloss
        hdiff
        flow
        hflow_measurable
        hUnique
        hFlow
        ν₀
        hν₀
        hν₀_supported
        heval
        hintegrable
        t
  exact (hequiv g x).symm

end Architecture
end NeuralNetworkEquivariance
