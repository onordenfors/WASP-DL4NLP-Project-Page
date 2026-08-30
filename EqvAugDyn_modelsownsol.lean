import Mathlib.Algebra.Group.Action.Basic
import Mathlib.Algebra.BigOperators.Group.Finset.Basic
import Mathlib.Analysis.InnerProductSpace.LinearMap
import Mathlib.LinearAlgebra.FiniteDimensional.Defs
import Mathlib.Algebra.Group.TransferInstance
import Mathlib.Algebra.Module.TransferInstance
import Mathlib.LinearAlgebra.FiniteDimensional.Basic
import Mathlib.Topology.Algebra.Module.FiniteDimension
import Mathlib.Analysis.Calculus.ContDiff.Basic
import Mathlib.Analysis.Calculus.ContDiff.Comp

/-!
# Equivariant neural networks

This file currently contains:

1. Equivariant maps.
2. Complete real inner-product spaces with orthogonal group representations.
3. Affine-layer parameters.
4. Equivariant activation specifications.
5. General inductive network architectures.
6. Recursively defined parameter types.
7. Recursive network realization.

No loss, gradient, ODE, or probability assumptions are introduced here.
-/

set_option autoImplicit false

universe uG uX uY uZ uV

namespace EquivariantNN

/-!
## Equivariant maps
-/

/--
A map `f : X → Y` is equivariant with respect to the actions of `G` on `X`
and `Y` when

`f (g • x) = g • f x`

for every `g : G` and `x : X`.

The group `G` is an explicit argument because, in general, it cannot be
inferred from the type `X → Y`.
-/
def IsEquivariant
    (G : Type uG)
    {X : Type uX}
    {Y : Type uY}
    [Group G]
    [MulAction G X]
    [MulAction G Y]
    (f : X → Y) : Prop :=
  ∀ (g : G) (x : X), f (g • x) = g • f x

namespace IsEquivariant

variable
  {G : Type uG}
  {X : Type uX}
  {Y : Type uY}
  {Z : Type uZ}

variable [Group G]
variable [MulAction G X]
variable [MulAction G Y]
variable [MulAction G Z]

/-- The identity map is equivariant. -/
@[simp]
theorem id :
    IsEquivariant G (fun x : X => x) := by
  intro g x
  rfl

/-- The composition of equivariant maps is equivariant. -/
theorem comp
    {f : X → Y}
    {h : Y → Z}
    (hh : IsEquivariant G h)
    (hf : IsEquivariant G f) :
    IsEquivariant G (h ∘ f) := by
  intro g x
  change h (f (g • x)) = g • h (f x)
  rw [hf g x]
  exact hh g (f x)

/-- Restatement of the defining equivariance equation. -/
theorem map_smul
    {f : X → Y}
    (hf : IsEquivariant G f)
    (g : G)
    (x : X) :
    f (g • x) = g • f x :=
  hf g x

/-- A constant map is equivariant if its value is fixed by every group element. -/
theorem const
    {y : Y}
    (hy : ∀ g : G, g • y = y) :
    IsEquivariant G (fun _ : X => y) := by
  intro g x
  exact (hy g).symm

/-- Equivariance is preserved under equality of functions. -/
theorem congr
    {f h : X → Y}
    (hf : IsEquivariant G f)
    (hfh : f = h) :
    IsEquivariant G h := by
  subst h
  exact hf

end IsEquivariant

/-!
## Orthogonal representation spaces
-/

/--
A finite-dimensional complete real inner-product space equipped with an
orthogonal representation of `G`.

Finite dimensionality is recorded explicitly because it will later be used
to construct finite-dimensional parameter spaces and apply the differential
and ODE APIs.
-/
structure RepSpace
    (G : Type uG)
    [Group G] where
  carrier : Type uV
  [normedAddCommGroup : NormedAddCommGroup carrier]
  [innerProductSpace : InnerProductSpace ℝ carrier]
  [completeSpace : CompleteSpace carrier]
  [finiteDimensional : FiniteDimensional ℝ carrier]
  ρ : G →* (carrier ≃ₗᵢ[ℝ] carrier)

namespace RepSpace

variable {G : Type uG}
variable [Group G]

/-- A representation space coerces to its carrier type. -/
instance : CoeSort (RepSpace G) (Type uV) where
  coe V := V.carrier

/-- Recover the bundled normed additive commutative group structure. -/
instance instNormedAddCommGroup
    (V : RepSpace G) :
    NormedAddCommGroup V :=
  V.normedAddCommGroup

/-- Recover the bundled real inner-product-space structure. -/
instance instInnerProductSpace
    (V : RepSpace G) :
    InnerProductSpace ℝ V :=
  V.innerProductSpace

/-- Recover the bundled completeness structure. -/
instance instCompleteSpace
    (V : RepSpace G) :
    CompleteSpace V :=
  V.completeSpace

/-- Recover the bundled finite-dimensionality assumption. -/
instance instFiniteDimensional
    (V : RepSpace G) :
    FiniteDimensional ℝ V :=
  V.finiteDimensional

/--
The group action induced by the bundled orthogonal representation.
-/
instance instMulAction
    (V : RepSpace G) :
    MulAction G V where
  smul g x := V.ρ g x
  one_smul x := by
    change V.ρ 1 x = x
    simp
  mul_smul g h x := by
    change V.ρ (g * h) x = V.ρ g (V.ρ h x)
    simp

variable (V : RepSpace G)

/-- Unfold the action induced by the representation. -/
@[simp]
theorem smul_def
    (g : G)
    (x : V) :
    g • x = V.ρ g x :=
  rfl

/-- The identity group element is represented by the identity map. -/
@[simp]
theorem rho_one_apply
    (x : V) :
    V.ρ 1 x = x := by
  simp

/-- Group multiplication is represented by composition. -/
@[simp]
theorem rho_mul_apply
    (g h : G)
    (x : V) :
    V.ρ (g * h) x = V.ρ g (V.ρ h x) := by
  simp

/-- Applying a representation and then the inverse representation cancels. -/
@[simp]
theorem rho_inv_apply_rho
    (g : G)
    (x : V) :
    V.ρ g⁻¹ (V.ρ g x) = x := by
  rw [← rho_mul_apply V g⁻¹ g x]
  simp

/-- Applying an inverse representation and then the representation cancels. -/
@[simp]
theorem rho_apply_rho_inv
    (g : G)
    (x : V) :
    V.ρ g (V.ρ g⁻¹ x) = x := by
  rw [← rho_mul_apply V g g⁻¹ x]
  simp

/-- The group action maps zero to zero. -/
@[simp]
theorem smul_zero
    (g : G) :
    g • (0 : V) = 0 := by
  change V.ρ g 0 = 0
  exact (V.ρ g).map_zero

/-- The group action preserves addition. -/
@[simp]
theorem smul_add
    (g : G)
    (x y : V) :
    g • (x + y) = g • x + g • y := by
  change V.ρ g (x + y) = V.ρ g x + V.ρ g y
  exact (V.ρ g).map_add x y

/-- The group action preserves negation. -/
@[simp]
theorem smul_neg
    (g : G)
    (x : V) :
    g • (-x) = -(g • x) := by
  change V.ρ g (-x) = -(V.ρ g x)
  exact (V.ρ g).map_neg x

/-- The group action preserves subtraction. -/
@[simp]
theorem smul_sub
    (g : G)
    (x y : V) :
    g • (x - y) = g • x - g • y := by
  change V.ρ g (x - y) = V.ρ g x - V.ρ g y
  exact (V.ρ g).map_sub x y

/-- The group action commutes with real scalar multiplication. -/
@[simp]
theorem smul_real_smul
    (g : G)
    (r : ℝ)
    (x : V) :
    g • (r • x) = r • (g • x) := by
  change V.ρ g (r • x) = r • V.ρ g x
  exact (V.ρ g).map_smul r x

/-- Every group element preserves norms. -/
@[simp]
theorem norm_smul
    (g : G)
    (x : V) :
    ‖g • x‖ = ‖x‖ := by
  change ‖V.ρ g x‖ = ‖x‖
  exact (V.ρ g).norm_map x

/-- Every group element preserves distances. -/
@[simp]
theorem dist_smul_smul
    (g : G)
    (x y : V) :
    dist (g • x) (g • y) = dist x y := by
  change dist (V.ρ g x) (V.ρ g y) = dist x y
  exact (V.ρ g).dist_map x y

/-- Every group element preserves the real inner product. -/
@[simp]
theorem inner_smul_smul
    (g : G)
    (x y : V) :
    inner ℝ (g • x) (g • y) = inner ℝ x y := by
  change inner ℝ (V.ρ g x) (V.ρ g y) = inner ℝ x y
  exact (V.ρ g).inner_map_map x y

/-- Acting by a group element is injective. -/
theorem smul_injective
    (g : G) :
    Function.Injective (fun x : V => g • x) := by
  intro x y hxy
  have h :
      g⁻¹ • (g • x) = g⁻¹ • (g • y) :=
    congrArg (fun z : V => g⁻¹ • z) hxy
  simpa only [inv_smul_smul] using h

/-- Acting by a group element is surjective. -/
theorem smul_surjective
    (g : G) :
    Function.Surjective (fun x : V => g • x) := by
  intro y
  refine ⟨g⁻¹ • y, ?_⟩
  exact smul_inv_smul g y

/-- Acting by a group element is bijective. -/
theorem smul_bijective
    (g : G) :
    Function.Bijective (fun x : V => g • x) :=
  ⟨smul_injective V g, smul_surjective V g⟩

/-- Each group element acts continuously. -/
theorem continuous_smul
    (g : G) :
    Continuous (fun x : V => g • x) := by
  change Continuous (fun x : V => V.ρ g x)
  exact (V.ρ g).continuous

end RepSpace

/-!
## Affine-layer parameters
-/

variable {G : Type uG}
variable [Group G]

/--
Parameters of an affine map from a representation space `V` to a
representation space `W`.

The represented affine map is

`x ↦ weight x + bias`.
-/
structure AffineParams
    (V W : RepSpace G) where
  weight : V →L[ℝ] W
  bias : W

namespace AffineParams

variable
  {G : Type uG}
  [Group G]
  {V W : RepSpace G}

/-- Evaluate an affine layer. -/
def eval
    (p : AffineParams V W)
    (x : V) : W :=
  p.weight x + p.bias

/-- Unfold affine-layer evaluation. -/
@[simp]
theorem eval_apply
    (p : AffineParams V W)
    (x : V) :
    p.eval x = p.weight x + p.bias :=
  rfl

end AffineParams

/-!
## Equivariant activation specifications
-/

/--
A possibly nonlinear activation map together with a proof that it is
equivariant.

No linearity, continuity, differentiability, or smoothness is assumed here.
-/
structure ActivationSpec
    (G : Type uG)
    [Group G]
    (V W : RepSpace G) where
  toFun : V → W
  equivariant' : IsEquivariant G toFun

namespace ActivationSpec

variable
  {G : Type uG}
  [Group G]
  {V W Z : RepSpace G}

/-- An activation specification can be applied as a function. -/
instance instCoeFun :
    CoeFun (ActivationSpec G V W) (fun _ => V → W) where
  coe σ := σ.toFun

/-- The function underlying an activation specification is equivariant. -/
theorem equivariant
    (σ : ActivationSpec G V W) :
    IsEquivariant G σ.toFun :=
  σ.equivariant'

/-- The defining equivariance equation of an activation. -/
@[simp]
theorem map_smul
    (σ : ActivationSpec G V W)
    (g : G)
    (x : V) :
    σ (g • x) = g • σ x :=
  σ.equivariant' g x

/-- The identity activation. -/
def id
    (V : RepSpace G) :
    ActivationSpec G V V where
  toFun := fun x : V => x
  equivariant' := by
    intro g x
    rfl

/-- Composition of two equivariant activation maps. -/
def comp
    (τ : ActivationSpec G W Z)
    (σ : ActivationSpec G V W) :
    ActivationSpec G V Z where
  toFun := fun x : V => τ (σ x)
  equivariant' := by
    intro g x
    change τ (σ (g • x)) = g • τ (σ x)
    calc
      τ (σ (g • x))
          = τ (g • σ x) := by
              rw [σ.map_smul]
      _ = g • τ (σ x) := by
              rw [τ.map_smul]

/-- The identity activation acts as the identity function. -/
@[simp]
theorem id_apply
    (V : RepSpace G)
    (x : V) :
    ActivationSpec.id V x = x :=
  rfl

/-- Evaluation of a composed activation. -/
@[simp]
theorem comp_apply
    (τ : ActivationSpec G W Z)
    (σ : ActivationSpec G V W)
    (x : V) :
    τ.comp σ x = τ (σ x) :=
  rfl

end ActivationSpec

/-!
## General inductive network architectures
-/

/--
A finite feed-forward network architecture indexed by its input and output
representation spaces.

All representation spaces in one architecture have carriers in the same
universe `uV`.
-/
inductive Architecture
    (G : Type uG)
    [Group G] :
    RepSpace.{uG, uV} G →
      RepSpace.{uG, uV} G →
      Type (max uG (uV + 1))
  | affine
      (V W : RepSpace.{uG, uV} G) :
      Architecture G V W
  | activation
      {V W : RepSpace.{uG, uV} G}
      (σ : ActivationSpec G V W) :
      Architecture G V W
  | comp
      {U V W : RepSpace.{uG, uV} G}
      (first : Architecture G U V)
      (second : Architecture G V W) :
      Architecture G U W

namespace Architecture

variable
  {G : Type uG}
  [Group G]

variable
  {U V W Z : RepSpace.{uG, uV} G}

/-!
## Recursive parameter types
-/

/--
The parameter type associated with an architecture.

An affine node has a weight and bias, an activation node has no trainable
parameters, and a composition contains the parameters of both subnetworks.
-/
def Params
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W) :
    Type uV :=
  match A with
  | .affine V W => AffineParams V W
  | .activation _ => PUnit
  | .comp first second => Params first × Params second

/-!
## Recursive realization
-/

/--
Evaluate an architecture with a given parameter value.

For a composition, the first architecture is evaluated first and its output is
passed to the second architecture.
-/
def realize
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W) :
    Params A → V → W :=
  match A with
  | .affine _ _ => fun p x => p.eval x
  | .activation σ => fun _ x => σ x
  | .comp first second =>
      fun p x => realize second p.2 (realize first p.1 x)

/-!
## Realization simplification lemmas
-/

/-- Realization of an affine architecture. -/
@[simp]
theorem realize_affine
    (V W : RepSpace.{uG, uV} G)
    (p : AffineParams V W)
    (x : V) :
    realize (Architecture.affine V W) p x =
      p.weight x + p.bias :=
  rfl

/-- Realization of an activation architecture. -/
@[simp]
theorem realize_activation
    (σ : ActivationSpec G V W)
    (p : Params (Architecture.activation σ))
    (x : V) :
    realize (Architecture.activation σ) p x =
      σ x :=
  rfl

/-- Realization of a sequential composition. -/
@[simp]
theorem realize_comp
    (first : Architecture G U V)
    (second : Architecture G V W)
    (p : Params (Architecture.comp first second))
    (x : U) :
    realize (Architecture.comp first second) p x =
      realize second p.2 (realize first p.1 x) :=
  rfl

/-!
## Convenient architecture constructors
-/

/-- A single affine layer. -/
def affineLayer
    (V W : RepSpace.{uG, uV} G) :
    Architecture G V W :=
  Architecture.affine V W

/-- A fixed equivariant activation layer. -/
def activationLayer
    (σ : ActivationSpec G V W) :
    Architecture G V W :=
  Architecture.activation σ

/-- An affine layer followed by an equivariant activation. -/
def affineThenActivation
    (U V W : RepSpace.{uG, uV} G)
    (σ : ActivationSpec G V W) :
    Architecture G U W :=
  Architecture.comp
    (Architecture.affine U V)
    (Architecture.activation σ)

/-- Two affine layers separated by an equivariant activation. -/
def twoAffineLayers
    (U V W : RepSpace.{uG, uV} G)
    (σ : ActivationSpec G V V) :
    Architecture G U W :=
  Architecture.comp
    (Architecture.comp
      (Architecture.affine U V)
      (Architecture.activation σ))
    (Architecture.affine V W)

/-!
## Parameter-type reduction lemmas
-/

/-- An affine node has an affine parameter value. -/
theorem params_affine_eq
    (V W : RepSpace.{uG, uV} G) :
    Params (Architecture.affine V W) =
      AffineParams V W :=
  rfl

/-- An activation node has no trainable parameters. -/
theorem params_activation_eq
    (σ : ActivationSpec G V W) :
    Params (Architecture.activation σ) =
      PUnit :=
  rfl

/-- A composition contains the parameters of both component architectures. -/
theorem params_comp_eq
    (first : Architecture G U V)
    (second : Architecture G V W) :
    Params (Architecture.comp first second) =
      (Params first × Params second) :=
  rfl

end Architecture

end EquivariantNN

namespace EquivariantNN

/-!
## Group action on affine parameters

For an affine layer

`x ↦ W x + b`

from `V` to `W`, the induced parameter action is

`g • (W, b) = (ρ_W(g) ∘ W ∘ ρ_V(g⁻¹), ρ_W(g)b)`.

The inverse on the input side is necessary for covariance of affine-layer
evaluation.
-/

namespace AffineParams

variable
  {G : Type uG}
  [Group G]

variable
  {V W : RepSpace.{uG, uV} G}

/-- The continuous linear map underlying a representation element. -/
def representationCLM
    (V : RepSpace.{uG, uV} G)
    (g : G) :
    V →L[ℝ] V :=
  (V.ρ g).toLinearIsometry.toContinuousLinearMap

/-- Evaluation of the continuous linear map underlying a representation. -/
@[simp]
theorem representationCLM_apply
    (V : RepSpace.{uG, uV} G)
    (g : G)
    (x : V) :
    representationCLM V g x = g • x :=
  rfl

/--
The induced action of `G` on the parameters of an affine layer.

The transformed weight is obtained by conjugating with the input and output
representations. The transformed bias is acted on by the output
representation.
-/
def act
    (g : G)
    (p : AffineParams V W) :
    AffineParams V W where
  weight :=
    (representationCLM W g).comp
      (p.weight.comp (representationCLM V g⁻¹))
  bias := g • p.bias

/-- The weight component of the transformed affine parameters. -/
@[simp]
theorem act_weight
    (g : G)
    (p : AffineParams V W) :
    (act g p).weight =
      (representationCLM W g).comp
        (p.weight.comp (representationCLM V g⁻¹)) :=
  rfl

/-- The transformed weight evaluated at an input. -/
@[simp]
theorem act_weight_apply
    (g : G)
    (p : AffineParams V W)
    (x : V) :
    (act g p).weight x =
      g • p.weight (g⁻¹ • x) :=
  rfl

/-- The bias component of the transformed affine parameters. -/
@[simp]
theorem act_bias
    (g : G)
    (p : AffineParams V W) :
    (act g p).bias = g • p.bias :=
  rfl

/--
Two affine parameter values are equal if their weights and biases are equal.
-/
@[ext]
theorem ext
    {p q : AffineParams V W}
    (hweight : p.weight = q.weight)
    (hbias : p.bias = q.bias) :
    p = q := by
  cases p with
  | mk pWeight pBias =>
      cases q with
      | mk qWeight qBias =>
          simp_all


/--
The identity group element leaves affine parameters unchanged.

This is proved from the definition of the induced action.
-/
theorem one_act
    (p : AffineParams V W) :
    act (1 : G) p = p := by
  apply AffineParams.ext
  · apply ContinuousLinearMap.ext
    intro x
    simp [act, representationCLM]
  · simp [act]

/--
Acting by a product is the same as acting successively.

This proves the multiplication law required for a `MulAction`.
-/
theorem mul_act
    (g h : G)
    (p : AffineParams V W) :
    act (g * h) p = act g (act h p) := by
  apply AffineParams.ext
  · apply ContinuousLinearMap.ext
    intro x
    simp [act, representationCLM, mul_smul]
  · simp [act, mul_smul]
/--
The `MulAction` on affine-layer parameters induced by the input and output
representations.
-/
instance instMulAction :
    MulAction G (AffineParams V W) where
  smul := act
  one_smul := one_act
  mul_smul := mul_act

/-- Unfold the group action on affine parameters. -/
@[simp]
theorem smul_def
    (g : G)
    (p : AffineParams V W) :
    g • p = act g p :=
  rfl

/-- The weight of affine parameters after applying the group action. -/
@[simp]
theorem smul_weight
    (g : G)
    (p : AffineParams V W) :
    (g • p).weight =
      (representationCLM W g).comp
        (p.weight.comp (representationCLM V g⁻¹)) :=
  rfl

/-- Evaluate the transformed weight at an input. -/
@[simp]
theorem smul_weight_apply
    (g : G)
    (p : AffineParams V W)
    (x : V) :
    (g • p).weight x =
      g • p.weight (g⁻¹ • x) :=
  rfl

/-- The bias of affine parameters after applying the group action. -/
@[simp]
theorem smul_bias
    (g : G)
    (p : AffineParams V W) :
    (g • p).bias = g • p.bias :=
  rfl

/-!
## Covariance of affine-layer evaluation
-/

/--
Affine-layer evaluation is covariant under the induced parameter action.

Both the parameters and the input are transformed by the same group element,
and the resulting output is transformed by that group element.
-/
theorem eval_smul
    (g : G)
    (p : AffineParams V W)
    (x : V) :
    (g • p).eval (g • x) =
      g • p.eval x := by
  change
    g • p.weight (g⁻¹ • (g • x)) + g • p.bias =
      g • (p.weight x + p.bias)
  rw [inv_smul_smul]
  rw [RepSpace.smul_add]

end AffineParams

/-!
## Recursive action on architecture parameters

The parameter action is now defined recursively over arbitrary finite
architectures.
-/

namespace Architecture

variable
  {G : Type uG}
  [Group G]

variable
  {U V W : RepSpace.{uG, uV} G}

/--
The recursively defined action on the parameter type of an architecture.

* Affine parameters use the conjugation action defined above.
* Activation nodes have no parameters.
* Composition acts componentwise on the parameters of both subnetworks.
-/
def paramsAct
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W)
    (g : G)
    (p : Params A) :
    Params A :=
  match A with
  | .affine _ _ => AffineParams.act g p
  | .activation _ => PUnit.unit
  | .comp first second =>
      (paramsAct first g p.1, paramsAct second g p.2)

/--
The identity element leaves all parameters unchanged.

The proof is by structural induction on the architecture.
-/
theorem paramsAct_one
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W)
    (p : Params A) :
    paramsAct A (1 : G) p = p := by
  induction A with
  | affine V W =>
      exact AffineParams.one_act p
  | activation σ =>
      cases p
      rfl
  | comp first second ihFirst ihSecond =>
      rcases p with ⟨pFirst, pSecond⟩
      change
        (paramsAct first 1 pFirst, paramsAct second 1 pSecond) =
          (pFirst, pSecond)
      rw [ihFirst pFirst, ihSecond pSecond]

/--
Acting by a product is the same as acting successively on all parameters.

The proof is by structural induction on the architecture.
-/
theorem paramsAct_mul
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W)
    (g h : G)
    (p : Params A) :
    paramsAct A (g * h) p =
      paramsAct A g (paramsAct A h p) := by
  induction A with
  | affine V W =>
      exact AffineParams.mul_act g h p
  | activation σ =>
      cases p
      rfl
  | comp first second ihFirst ihSecond =>
      rcases p with ⟨pFirst, pSecond⟩
      change
        (paramsAct first (g * h) pFirst,
          paramsAct second (g * h) pSecond) =
        (paramsAct first g (paramsAct first h pFirst),
          paramsAct second g (paramsAct second h pSecond))
      rw [ihFirst pFirst, ihSecond pSecond]

/--
The group action on the parameter type of a fixed architecture.

This instance is installed only after both action laws have been proved.
-/
instance instParamsMulAction
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W) :
    MulAction G (Params A) where
  smul := paramsAct A
  one_smul := paramsAct_one A
  mul_smul := paramsAct_mul A

/-- Unfold the group action on architecture parameters. -/
@[simp]
theorem params_smul_def
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W)
    (g : G)
    (p : Params A) :
    g • p = paramsAct A g p :=
  rfl

/-- The recursive parameter action on an affine node. -/
@[simp]
theorem paramsAct_affine
    (V W : RepSpace.{uG, uV} G)
    (g : G)
    (p : Params (Architecture.affine V W)) :
    paramsAct (Architecture.affine V W) g p =
      AffineParams.act g p :=
  rfl

/-- The recursive parameter action on an activation node. -/
@[simp]
theorem paramsAct_activation
    (σ : ActivationSpec G V W)
    (g : G)
    (p : Params (Architecture.activation σ)) :
    paramsAct (Architecture.activation σ) g p =
      PUnit.unit := by
  cases p
  rfl

/-- The recursive parameter action on a composition. -/
@[simp]
theorem paramsAct_comp
    (first : Architecture G U V)
    (second : Architecture G V W)
    (g : G)
    (p : Params (Architecture.comp first second)) :
    paramsAct (Architecture.comp first second) g p =
      (paramsAct first g p.1, paramsAct second g p.2) :=
  rfl

/-!
## Covariance of arbitrary network realization
-/

/--
The realization of every architecture is covariant under the recursive
parameter action.

This theorem is proved by structural induction on the architecture.
-/
theorem realize_paramsAct
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W)
    (g : G)
    (p : Params A)
    (x : V) :
    realize A (paramsAct A g p) (g • x) =
      g • realize A p x := by
  induction A with
  | affine V W =>
      exact AffineParams.eval_smul g p x
  | activation σ =>
      exact σ.map_smul g x
  | comp first second ihFirst ihSecond =>
      rcases p with ⟨pFirst, pSecond⟩
      change
        realize second
            (paramsAct second g pSecond)
            (realize first
              (paramsAct first g pFirst)
              (g • x)) =
          g • realize second pSecond
            (realize first pFirst x)
      rw [ihFirst pFirst x]
      exact ihSecond pSecond (realize first pFirst x)

/--
The realization covariance theorem expressed using the installed
`MulAction` on the parameter type.
-/
theorem realize_smul
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W)
    (g : G)
    (p : Params A)
    (x : V) :
    realize A (g • p) (g • x) =
      g • realize A p x := by
  exact realize_paramsAct A g p x

end Architecture

end EquivariantNN

namespace EquivariantNN

/-!
## Finite indexed datasets

A dataset is represented as a family of input-target pairs indexed by `ι`.
Finiteness is carried separately by a `Fintype ι` instance when defining
empirical risks.

Using an index type rather than a set permits repeated samples.
-/

universe uI uR

/--
An indexed dataset with inputs in `X` and targets in `Y`.

Finiteness is not part of this structure. Finite empirical risks assume a
`Fintype` instance on `ι`.
-/
structure Dataset
    (ι : Type uI)
    (X : Type uX)
    (Y : Type uY) where
  input : ι → X
  target : ι → Y

/-!
## Full group-orbit augmentation

The orbit-augmented dataset is indexed by `G × ι`. It contains one augmented
sample for every group element and every original sample index.

Repeated orbit points are intentionally retained. Thus augmentation is by
group elements, rather than by distinct elements of the set-theoretic orbit.
-/

namespace Dataset

variable
  {G : Type uG}
  [Group G]

variable
  {ι : Type uI}
  {X : Type uX}
  {Y : Type uY}

variable
  [MulAction G X]
  [MulAction G Y]

/--
Full simultaneous group-orbit augmentation of inputs and targets.

The augmented sample indexed by `(g, i)` is

`(g • D.input i, g • D.target i)`.
-/
def orbitAugment
    (D : Dataset ι X Y) :
    Dataset (G × ι) X Y where
  input gi := gi.1 • D.input gi.2
  target gi := gi.1 • D.target gi.2

/-- The input at an index of the orbit-augmented dataset. -/
@[simp]
theorem orbitAugment_input
    (D : Dataset ι X Y)
    (g : G)
    (i : ι) :
    D.orbitAugment.input (g, i) =
      g • D.input i :=
  rfl

/-- The target at an index of the orbit-augmented dataset. -/
@[simp]
theorem orbitAugment_target
    (D : Dataset ι X Y)
    (g : G)
    (i : ι) :
    D.orbitAugment.target (g, i) =
      g • D.target i :=
  rfl

end Dataset

/-!
## Generic invariant losses

No analytic, metric, or order-theoretic property of the loss is imposed in
this section.

In particular, we do not assume that the loss is:

* nonnegative;
* symmetric;
* zero on equal arguments;
* continuous;
* differentiable;
* convex;
* a metric;
* squared error.

The only hypothesis used to prove augmented-risk invariance is invariance
under the diagonal action on the prediction and target.
-/

/--
A two-argument loss is invariant under the diagonal action of `G` if acting
on the prediction and target by the same group element does not change the
loss.
-/
def IsInvariantLoss
    (G : Type uG)
    {Y : Type uY}
    {R : Type uR}
    [Group G]
    [MulAction G Y]
    (loss : Y → Y → R) :
    Prop :=
  ∀ (g : G) (prediction target : Y),
    loss (g • prediction) (g • target) =
      loss prediction target

namespace IsInvariantLoss

variable
  {G : Type uG}
  {Y : Type uY}
  {R : Type uR}

variable [Group G]
variable [MulAction G Y]

/-- Restatement of diagonal loss invariance. -/
theorem map_smul
    {loss : Y → Y → R}
    (hloss : IsInvariantLoss G loss)
    (g : G)
    (prediction target : Y) :
    loss (g • prediction) (g • target) =
      loss prediction target :=
  hloss g prediction target

end IsInvariantLoss

/-!
## Finite-group reindexing

Left multiplication by a fixed group element is a permutation of the group.
The following lemmas isolate the finite-sum reindexing argument used in the
augmented-risk theorem.
-/

namespace GroupReindex

variable
  {G : Type uG}
  [Group G]


/-- Left multiplication by a fixed group element as an equivalence. -/
def leftMulEquiv
    (g : G) :
    G ≃ G where
  toFun h := g * h
  invFun h := g⁻¹ * h
  left_inv h := by
    simp
  right_inv h := by
    simp

/-- Evaluation of the left-multiplication equivalence. -/
@[simp]
theorem leftMulEquiv_apply
    (g h : G) :
    leftMulEquiv g h = g * h :=
  rfl

/-- Evaluation of the inverse left-multiplication equivalence. -/
@[simp]
theorem leftMulEquiv_symm_apply
    (g h : G) :
    (leftMulEquiv g).symm h = g⁻¹ * h :=
  rfl

/--
A finite sum over `G` is unchanged when its index is left-multiplied by a
fixed group element.
-/
theorem sum_left_mul
    [Fintype G]
    {R : Type uR}
    [AddCommMonoid R]
    (g : G)
    (f : G → R) :
    (∑ h : G, f (g * h)) =
      ∑ h : G, f h := by
  exact (leftMulEquiv g).sum_comp f

/--
A finite sum can be reindexed by left multiplication in the reverse
direction.
-/
theorem sum_eq_sum_left_mul
    [Fintype G]
    {R : Type uR}
    [AddCommMonoid R]
    (g : G)
    (f : G → R) :
    (∑ h : G, f h) =
      ∑ h : G, f (g * h) := by
  symm
  exact sum_left_mul g f

/--
A finite sum over `G` is unchanged when its index is left-multiplied by the
inverse of a fixed group element.
-/
theorem sum_inv_mul
    [Fintype G]
    {R : Type uR}
    [AddCommMonoid R]
    (g : G)
    (f : G → R) :
    (∑ h : G, f (g⁻¹ * h)) =
      ∑ h : G, f h := by
  exact sum_left_mul g⁻¹ f

end GroupReindex

/-!
## Augmented empirical risk

The augmented risk is defined as an unnormalized finite sum. Normalization
can be added later as a corollary and does not affect invariance.
-/

namespace Architecture

variable
  {G : Type uG}
  [Group G]
  [Fintype G]

variable
  {ι : Type uI}
  [Fintype ι]

variable
  {X Y : RepSpace.{uG, uV} G}

variable
  {R : Type uR}
  [AddCommMonoid R]

/--
The unnormalized empirical risk over the fully orbit-augmented dataset.

For every original sample `i` and every group element `h`, the risk contains
the term

`loss (realize A p (h • D.input i)) (h • D.target i)`.
-/
def augmentedRisk
    (A : Architecture G X Y)
    (D : Dataset ι X Y)
    (loss : Y → Y → R)
    (p : Params A) :
    R :=
  ∑ i : ι, ∑ h : G,
    loss
      (realize A p (h • D.input i))
      (h • D.target i)

/-- Unfold the augmented empirical risk. -/
theorem augmentedRisk_def
    (A : Architecture G X Y)
    (D : Dataset ι X Y)
    (loss : Y → Y → R)
    (p : Params A) :
    augmentedRisk A D loss p =
      ∑ i : ι, ∑ h : G,
        loss
          (realize A p (h • D.input i))
          (h • D.target i) :=
  rfl

/-!
## Pointwise transformed-loss identity
-/

omit [Fintype G] [Fintype ι] [AddCommMonoid R] in
/--
After transforming the network parameters by `g`, the loss at the augmented
sample indexed by `g * h` equals the original loss at the sample indexed by
`h`.

This uses only realization covariance, the group action law, and diagonal
invariance of the loss.
-/
theorem transformed_augmented_term
    (A : Architecture G X Y)
    (D : Dataset ι X Y)
    (loss : Y → Y → R)
    (hloss : IsInvariantLoss G loss)
    (g h : G)
    (p : Params A)
    (i : ι) :
    loss
        (realize A (g • p) ((g * h) • D.input i))
        ((g * h) • D.target i) =
      loss
        (realize A p (h • D.input i))
        (h • D.target i) := by
  rw [mul_smul, mul_smul]
  rw [realize_smul]
  exact hloss
    g
    (realize A p (h • D.input i))
    (h • D.target i)

/-!
## Invariance of augmented empirical risk
-/

/--
The fully orbit-augmented empirical risk is invariant under the induced
action on network parameters.

The only property assumed about `loss` is diagonal invariance:

`loss (g • y) (g • y') = loss y y'`.

No continuity, differentiability, convexity, positivity, symmetry, or
specific loss formula is used.
-/
theorem augmentedRisk_invariant
    (A : Architecture G X Y)
    (D : Dataset ι X Y)
    (loss : Y → Y → R)
    (hloss : IsInvariantLoss G loss)
    (g : G)
    (p : Params A) :
    augmentedRisk A D loss (g • p) =
      augmentedRisk A D loss p := by
  unfold augmentedRisk
  apply Finset.sum_congr rfl
  intro i _
  calc
    (∑ h : G,
        loss
          (realize A (g • p) (h • D.input i))
          (h • D.target i)) =
      ∑ h : G,
        loss
          (realize A (g • p) ((g * h) • D.input i))
          ((g * h) • D.target i) := by
      exact
        GroupReindex.sum_eq_sum_left_mul
          g
          (fun h : G =>
            loss
              (realize A (g • p) (h • D.input i))
              (h • D.target i))
    _ =
      ∑ h : G,
        loss
          (realize A p (h • D.input i))
          (h • D.target i) := by
      apply Finset.sum_congr rfl
      intro h _
      exact transformed_augmented_term
        A D loss hloss g h p i

end Architecture

end EquivariantNN

namespace EquivariantNN

/-!
## Linear structure on affine parameters

An affine parameter consists of a continuous linear map and a bias vector.
Its canonical vector-space structure is transported from the product

`(V →L[ℝ] W) × W`.
-/

namespace AffineParams

variable
  {G : Type uG}
  [Group G]

variable
  {V W : RepSpace.{uG, uV} G}

/--
The canonical equivalence between affine parameters and the product of the
weight space with the bias space.
-/
def equivProd :
    AffineParams V W ≃ ((V →L[ℝ] W) × W) where
  toFun p := (p.weight, p.bias)
  invFun p :=
    {
      weight := p.1
      bias := p.2
    }
  left_inv p := by
    cases p
    rfl
  right_inv p := by
    cases p
    rfl

/-- Evaluation of the affine-parameter product equivalence. -/
@[simp]
theorem equivProd_apply
    (p : AffineParams V W) :
    equivProd p = (p.weight, p.bias) :=
  rfl

/-- The weight component after applying the inverse product equivalence. -/
@[simp]
theorem equivProd_symm_weight
    (p : (V →L[ℝ] W) × W) :
    (equivProd.symm p).weight = p.1 :=
  rfl

/-- The bias component after applying the inverse product equivalence. -/
@[simp]
theorem equivProd_symm_bias
    (p : (V →L[ℝ] W) × W) :
    (equivProd.symm p).bias = p.2 :=
  rfl

/--
The additive commutative group structure on affine parameters, transported
from the corresponding product.
-/
instance instAddCommGroup :
    AddCommGroup (AffineParams V W) :=
  equivProd.addCommGroup

/--
The canonical additive equivalence between affine parameters and the product
of their weight and bias components.
-/
def addEquivProd :
    AffineParams V W ≃+ ((V →L[ℝ] W) × W) :=
  equivProd.addEquiv

/--
The real module structure on affine parameters, transported through the
additive equivalence with the product parameter space.
-/
instance instModule :
    Module ℝ (AffineParams V W) :=
  AddEquiv.module ℝ addEquivProd

/-!
## Componentwise operation formulas

These lemmas form the public API for the transported algebraic structures.
-/

/-- The weight component of zero affine parameters is zero. -/
@[simp]
theorem zero_weight :
    (0 : AffineParams V W).weight = 0 :=
  rfl

/-- The bias component of zero affine parameters is zero. -/
@[simp]
theorem zero_bias :
    (0 : AffineParams V W).bias = 0 :=
  rfl

/-- The weight component of a sum is the sum of the weights. -/
@[simp]
theorem add_weight
    (p q : AffineParams V W) :
    (p + q).weight = p.weight + q.weight :=
  rfl

/-- The bias component of a sum is the sum of the biases. -/
@[simp]
theorem add_bias
    (p q : AffineParams V W) :
    (p + q).bias = p.bias + q.bias :=
  rfl

/-- The weight component of a negation is the negated weight. -/
@[simp]
theorem neg_weight
    (p : AffineParams V W) :
    (-p).weight = -p.weight :=
  rfl

/-- The bias component of a negation is the negated bias. -/
@[simp]
theorem neg_bias
    (p : AffineParams V W) :
    (-p).bias = -p.bias :=
  rfl

/-- The weight component of a difference is the difference of the weights. -/
@[simp]
theorem sub_weight
    (p q : AffineParams V W) :
    (p - q).weight = p.weight - q.weight :=
  rfl

/-- The bias component of a difference is the difference of the biases. -/
@[simp]
theorem sub_bias
    (p q : AffineParams V W) :
    (p - q).bias = p.bias - q.bias :=
  rfl

/--
The weight component of a real scalar multiple is the corresponding scalar
multiple of the weight.
-/
@[simp]
theorem real_smul_weight
    (r : ℝ)
    (p : AffineParams V W) :
    (r • p).weight = r • p.weight :=
  rfl

/--
The bias component of a real scalar multiple is the corresponding scalar
multiple of the bias.
-/
@[simp]
theorem real_smul_bias
    (r : ℝ)
    (p : AffineParams V W) :
    (r • p).bias = r • p.bias :=
  rfl

/-!
## Linearity of the affine parameter action
-/

/-- The induced action on affine parameters preserves zero. -/
@[simp]
theorem act_zero
    (g : G) :
    act g (0 : AffineParams V W) = 0 := by
  apply AffineParams.ext
  · apply ContinuousLinearMap.ext
    intro x
    change
      g • ((0 : V →L[ℝ] W) (g⁻¹ • x)) =
        (0 : V →L[ℝ] W) x
    simp
  · change g • (0 : W) = 0
    exact RepSpace.smul_zero W g

/-- The induced action on affine parameters preserves addition. -/
@[simp]
theorem act_add
    (g : G)
    (p q : AffineParams V W) :
    act g (p + q) = act g p + act g q := by
  apply AffineParams.ext
  · apply ContinuousLinearMap.ext
    intro x
    change
      g • ((p.weight + q.weight) (g⁻¹ • x)) =
        g • p.weight (g⁻¹ • x) +
          g • q.weight (g⁻¹ • x)
    rw [add_apply]
    exact RepSpace.smul_add W g
      (p.weight (g⁻¹ • x))
      (q.weight (g⁻¹ • x))
  · change
      g • (p.bias + q.bias) =
        g • p.bias + g • q.bias
    exact RepSpace.smul_add W g p.bias q.bias

/--
The induced action on affine parameters preserves real scalar
multiplication.
-/
@[simp]
theorem act_real_smul
    (g : G)
    (r : ℝ)
    (p : AffineParams V W) :
    act g (r • p) = r • act g p := by
  apply AffineParams.ext
  · apply ContinuousLinearMap.ext
    intro x
    change
      g • ((r • p.weight) (g⁻¹ • x)) =
        r • (g • p.weight (g⁻¹ • x))
    rw [smul_apply]
    exact RepSpace.smul_real_smul W g r
      (p.weight (g⁻¹ • x))
  · change
      g • (r • p.bias) =
        r • (g • p.bias)
    exact RepSpace.smul_real_smul W g r p.bias

/--
The action of a fixed group element on affine parameters, bundled as a real
linear map.
-/
def actionLinearMap
    (g : G) :
    AffineParams V W →ₗ[ℝ] AffineParams V W where
  toFun := act g
  map_add' := act_add g
  map_smul' := act_real_smul g

/-- Evaluation of the bundled linear action. -/
@[simp]
theorem actionLinearMap_apply
    (g : G)
    (p : AffineParams V W) :
    actionLinearMap g p = g • p :=
  rfl

/--
Applying the action of `g` and then the action of `g⁻¹` returns the original
affine parameters.
-/
theorem actionLinearMap_inv_apply
    (g : G)
    (p : AffineParams V W) :
    actionLinearMap g⁻¹ (actionLinearMap g p) = p := by
  change g⁻¹ • (g • p) = p
  exact inv_smul_smul g p

/--
Applying the action of `g⁻¹` and then the action of `g` returns the original
affine parameters.
-/
theorem actionLinearMap_apply_inv
    (g : G)
    (p : AffineParams V W) :
    actionLinearMap g (actionLinearMap g⁻¹ p) = p := by
  change g • (g⁻¹ • p) = p
  exact smul_inv_smul g p

/--
The action of a fixed group element on affine parameters, bundled as a real
linear equivalence.
-/
def actionLinearEquiv
    (g : G) :
    AffineParams V W ≃ₗ[ℝ] AffineParams V W where
  toLinearMap := actionLinearMap g
  invFun := actionLinearMap g⁻¹
  left_inv := actionLinearMap_inv_apply g
  right_inv := actionLinearMap_apply_inv g

/-- Evaluation of the bundled linear equivalence. -/
@[simp]
theorem actionLinearEquiv_apply
    (g : G)
    (p : AffineParams V W) :
    actionLinearEquiv g p = g • p :=
  rfl

/-- Evaluation of the inverse bundled linear equivalence. -/
@[simp]
theorem actionLinearEquiv_symm_apply
    (g : G)
    (p : AffineParams V W) :
    (actionLinearEquiv g).symm p = g⁻¹ • p :=
  rfl

end AffineParams


/-!
## Finite-group reindexing
-/

namespace GroupReindexing

variable
  {G : Type uG}
  [Group G]
  [Fintype G]

/--
Left multiplication by a fixed group element, regarded as an equivalence of
the underlying type.
-/
def leftMulEquiv
    (g : G) :
    G ≃ G where
  toFun h := g * h
  invFun h := g⁻¹ * h
  left_inv h := by
    simp [mul_assoc]
  right_inv h := by
    simp [mul_assoc]

/-- Evaluation of the left-multiplication equivalence. -/
@[simp]
theorem leftMulEquiv_apply
    (g h : G) :
    leftMulEquiv g h = g * h :=
  rfl

/-- Evaluation of the inverse left-multiplication equivalence. -/
@[simp]
theorem leftMulEquiv_symm_apply
    (g h : G) :
    (leftMulEquiv g).symm h = g⁻¹ * h :=
  rfl

/--
A finite sum over a group is unchanged by left-multiplying its index.
-/
theorem sum_left_mul
    {M : Type*}
    [AddCommMonoid M]
    (g : G)
    (f : G → M) :
    (∑ h : G, f (g * h)) =
      ∑ h : G, f h := by
  simpa only [leftMulEquiv_apply] using
    (Equiv.sum_comp (leftMulEquiv g) f)

/--
A finite sum over a group is unchanged by reindexing with
`h ↦ g⁻¹ * h`.
-/
theorem sum_inv_mul
    {M : Type*}
    [AddCommMonoid M]
    (g : G)
    (f : G → M) :
    (∑ h : G, f (g⁻¹ * h)) =
      ∑ h : G, f h := by
  simpa only using sum_left_mul (g := g⁻¹) f

end GroupReindexing

/-!
## Generic orbit-augmented empirical risk
-/

namespace Architecture

variable
  {G : Type uG}
  [Group G]
  [Fintype G]

variable
  {ι : Type*}
  [Fintype ι]

variable
  {X Y : RepSpace.{uG, uV} G}


/-!
## Pointwise transformed-loss identity
-/

/--
The loss contribution from transformed parameters at augmentation index `h`
equals the original-parameter contribution at the reindexed group element
`g⁻¹ * h`.

This theorem uses only:

1. covariance of network realization;
2. the group-action laws;
3. diagonal invariance of the loss.
-/
theorem transformed_loss_term
    (A : Architecture G X Y)
    (D : Dataset ι X Y)
    (loss : Y → Y → ℝ)
    (hloss : IsInvariantLoss G loss)
    (g h : G)
    (p : Params A)
    (i : ι) :
    loss
        (realize A (g • p) (h • D.input i))
        (h • D.target i) =
      loss
        (realize A p ((g⁻¹ * h) • D.input i))
        ((g⁻¹ * h) • D.target i) := by
  have hInput :
      h • D.input i =
        g • ((g⁻¹ * h) • D.input i) := by
    simp [mul_smul]

  have hTarget :
      h • D.target i =
        g • ((g⁻¹ * h) • D.target i) := by
    simp [mul_smul]

  rw [hInput, hTarget]
  rw [realize_smul]
  exact hloss g
    (realize A p ((g⁻¹ * h) • D.input i))
    ((g⁻¹ * h) • D.target i)


/-!
## Optional normalized augmented empirical risk
-/

/--
The normalized augmented empirical risk.

No nonemptiness assumption is needed for its definition. If either indexing
type is empty, the finite sum is zero.
-/
noncomputable def meanAugmentedRisk
    (A : Architecture G X Y)
    (D : Dataset ι X Y)
    (loss : Y → Y → ℝ)
    (p : Params A) :
    ℝ :=
  ((Fintype.card ι : ℝ) * (Fintype.card G : ℝ))⁻¹ *
    augmentedRisk A D loss p

/--
The normalized augmented empirical risk is invariant whenever the loss is
diagonally invariant.
-/
theorem meanAugmentedRisk_invariant
    (A : Architecture G X Y)
    (D : Dataset ι X Y)
    (loss : Y → Y → ℝ)
    (hloss : IsInvariantLoss G loss)
    (g : G)
    (p : Params A) :
    meanAugmentedRisk A D loss (g • p) =
      meanAugmentedRisk A D loss p := by
  unfold meanAugmentedRisk
  rw [augmentedRisk_invariant A D loss hloss g p]

end Architecture

end EquivariantNN
namespace EquivariantNN

/-!
## Linear structure on parameters of arbitrary architectures

The parameter type of an architecture is recursively built from:

* `AffineParams V W` for affine nodes;
* `PUnit` for activation nodes;
* products for composed architectures.

The corresponding additive and real-module structures are constructed by
structural recursion on the architecture.
-/

namespace Architecture

variable
  {G : Type uG}
  [Group G]

noncomputable section

/-!
### Trivial linear structure on activation parameters

An activation node has no trainable parameters. Its parameter type is
`PUnit`, equipped here with its unique additive and real-linear structures.
-/

/-- The unique zero element of `PUnit`. -/
instance instPUnitZero :
    Zero PUnit where
  zero := PUnit.unit

/-- The unique addition operation on `PUnit`. -/
instance instPUnitAdd :
    Add PUnit where
  add _ _ := PUnit.unit

/-- The unique negation operation on `PUnit`. -/
instance instPUnitNeg :
    Neg PUnit where
  neg _ := PUnit.unit

/-- The unique subtraction operation on `PUnit`. -/
instance instPUnitSub :
    Sub PUnit where
  sub _ _ := PUnit.unit

/-- The unique natural-number scalar action on `PUnit`. -/
instance instPUnitNatSMul :
    SMul ℕ PUnit where
  smul _ _ := PUnit.unit

/-- The unique integer scalar action on `PUnit`. -/
instance instPUnitIntSMul :
    SMul ℤ PUnit where
  smul _ _ := PUnit.unit

/-- The unique real scalar action on `PUnit`. -/
instance instPUnitRealSMul :
    SMul ℝ PUnit where
  smul _ _ := PUnit.unit

/-- `PUnit` as the trivial additive commutative group. -/
instance instPUnitAddCommGroup :
    AddCommGroup PUnit where
  add_assoc _ _ _ := rfl
  zero_add _ := rfl
  add_zero _ := rfl
  add_comm _ _ := rfl
  neg_add_cancel _ := rfl
  nsmul := fun _ _ => PUnit.unit
  nsmul_zero _ := rfl
  nsmul_succ _ _ := rfl
  zsmul := fun _ _ => PUnit.unit
  zsmul_zero' _ := rfl
  zsmul_succ' _ _ := rfl
  zsmul_neg' _ _ := rfl

/-- `PUnit` as the trivial real module. -/
instance instPUnitModule :
    Module ℝ PUnit where
  smul := fun _ _ => PUnit.unit
  one_smul _ := rfl
  mul_smul _ _ _ := rfl
  smul_add _ _ _ := rfl
  smul_zero _ := rfl
  add_smul _ _ _ := rfl
  zero_smul _ := rfl

/-!
### Recursive additive structure
-/

/--
The additive commutative group structure on the parameter type of an
architecture.

The definition is instance-reducible, allowing Lean to unfold it when the
architecture is known to be affine, an activation, or a composition.
-/
@[instance_reducible]
noncomputable def paramsAddCommGroup
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W) :
    AddCommGroup (Params A) :=
  match A with
  | .affine _ _ =>
      AffineParams.instAddCommGroup
  | .activation _ =>
      instPUnitAddCommGroup
  | .comp first second =>
      letI : AddCommGroup (Params first) :=
        paramsAddCommGroup first
      letI : AddCommGroup (Params second) :=
        paramsAddCommGroup second
      inferInstanceAs
        (AddCommGroup (Params first × Params second))

/--
Install the recursively constructed additive commutative group structure.
-/
noncomputable instance instParamsAddCommGroup
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W) :
    AddCommGroup (Params A) :=
  paramsAddCommGroup A

/-!
### Recursive real-module structure
-/

/--
The real module structure on the parameter type of an architecture.

The additive structure used here is the recursively constructed additive
structure above.
-/
@[instance_reducible]
noncomputable def paramsModule
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W) :
    Module ℝ (Params A) :=
  match A with
  | .affine _ _ =>
      AffineParams.instModule
  | .activation _ =>
      instPUnitModule
  | .comp first second =>
      letI : AddCommGroup (Params first) :=
        paramsAddCommGroup first
      letI : AddCommGroup (Params second) :=
        paramsAddCommGroup second
      letI : Module ℝ (Params first) :=
        paramsModule first
      letI : Module ℝ (Params second) :=
        paramsModule second
      inferInstanceAs
        (Module ℝ (Params first × Params second))

/--
Install the recursively constructed real module structure.
-/
noncomputable instance instParamsModule
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W) :
    Module ℝ (Params A) :=
  paramsModule A

/-!
## Linearity of the recursive parameter action
-/

/--
The recursive parameter action preserves addition.

This is proved by structural induction on the architecture.
-/
theorem paramsAct_add
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W)
    (g : G)
    (p q : Params A) :
    paramsAct A g (p + q) =
      paramsAct A g p + paramsAct A g q := by
  induction A with
  | affine V W =>
      exact AffineParams.act_add g p q
  | activation σ =>
      cases p
      cases q
      rfl
  | comp first second ihFirst ihSecond =>
      rcases p with ⟨pFirst, pSecond⟩
      rcases q with ⟨qFirst, qSecond⟩
      change
        (paramsAct first g (pFirst + qFirst),
          paramsAct second g (pSecond + qSecond)) =
        (paramsAct first g pFirst + paramsAct first g qFirst,
          paramsAct second g pSecond + paramsAct second g qSecond)
      rw [ihFirst pFirst qFirst, ihSecond pSecond qSecond]

/--
The recursive parameter action preserves real scalar multiplication.

This is proved by structural induction on the architecture.
-/
theorem paramsAct_real_smul
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W)
    (g : G)
    (r : ℝ)
    (p : Params A) :
    paramsAct A g (r • p) =
      r • paramsAct A g p := by
  induction A with
  | affine V W =>
      exact AffineParams.act_real_smul g r p
  | activation σ =>
      cases p
      rfl
  | comp first second ihFirst ihSecond =>
      rcases p with ⟨pFirst, pSecond⟩
      change
        (paramsAct first g (r • pFirst),
          paramsAct second g (r • pSecond)) =
        (r • paramsAct first g pFirst,
          r • paramsAct second g pSecond)
      rw [ihFirst pFirst, ihSecond pSecond]

/-!
## Parameter action as a linear map
-/

/--
The action of a fixed group element on the complete parameter type, bundled
as a real-linear map.
-/
noncomputable def paramsActionLinearMap
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W)
    (g : G) :
    Params A →ₗ[ℝ] Params A where
  toFun := paramsAct A g
  map_add' := paramsAct_add A g
  map_smul' := paramsAct_real_smul A g

/-- Evaluation of the parameter-action linear map. -/
@[simp]
theorem paramsActionLinearMap_apply
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W)
    (g : G)
    (p : Params A) :
    paramsActionLinearMap A g p =
      g • p :=
  rfl

/--
The action of the inverse group element is a left inverse to the action of
the original element.
-/
theorem paramsActionLinearMap_inv_apply
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W)
    (g : G)
    (p : Params A) :
    paramsActionLinearMap A g⁻¹
        (paramsActionLinearMap A g p) =
      p := by
  change g⁻¹ • (g • p) = p
  exact inv_smul_smul g p

/--
The action of the inverse group element is a right inverse to the action of
the original element.
-/
theorem paramsActionLinearMap_apply_inv
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W)
    (g : G)
    (p : Params A) :
    paramsActionLinearMap A g
        (paramsActionLinearMap A g⁻¹ p) =
      p := by
  change g • (g⁻¹ • p) = p
  exact smul_inv_smul g p

/--
The action of a fixed group element on the parameter space, bundled as a
real-linear equivalence.
-/
noncomputable def paramsActionLinearEquiv
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W)
    (g : G) :
    Params A ≃ₗ[ℝ] Params A where
  toLinearMap := paramsActionLinearMap A g
  invFun := paramsActionLinearMap A g⁻¹
  left_inv := paramsActionLinearMap_inv_apply A g
  right_inv := paramsActionLinearMap_apply_inv A g

/-- Evaluation of the parameter-action linear equivalence. -/
@[simp]
theorem paramsActionLinearEquiv_apply
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W)
    (g : G)
    (p : Params A) :
    paramsActionLinearEquiv A g p =
      g • p :=
  rfl

/-- Evaluation of the inverse parameter-action linear equivalence. -/
@[simp]
theorem paramsActionLinearEquiv_symm_apply
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W)
    (g : G)
    (p : Params A) :
    (paramsActionLinearEquiv A g).symm p =
      g⁻¹ • p :=
  rfl

end

end Architecture

end EquivariantNN

namespace EquivariantNN

/-!
## Finite-dimensionality of architecture parameter spaces

Every representation space is finite-dimensional by definition.

The weight space of an affine layer is finite-dimensional because it is a
space of continuous linear maps between finite-dimensional spaces. Affine
parameters are linearly equivalent to the product of the weight space and the
bias space.

Finally, parameter spaces of arbitrary architectures are finite-dimensional
by structural recursion.
-/

namespace AffineParams

variable
  {G : Type uG}
  [Group G]

variable
  {V W : RepSpace.{uG, uV} G}

noncomputable section

/--
The canonical linear equivalence between affine parameters and the product
of the weight space with the bias space.

This is constructed directly to ensure that it uses the already installed
module structure on `AffineParams`.
-/
def linearEquivProd :
    AffineParams V W ≃ₗ[ℝ] ((V →L[ℝ] W) × W) where
  toFun p := (p.weight, p.bias)
  invFun p :=
    {
      weight := p.1
      bias := p.2
    }
  left_inv p := by
    cases p
    rfl
  right_inv p := by
    cases p
    rfl
  map_add' p q := by
    rfl
  map_smul' r p := by
    rfl

/-- Evaluation of the affine-parameter linear equivalence. -/
@[simp]
theorem linearEquivProd_apply
    (p : AffineParams V W) :
    linearEquivProd p = (p.weight, p.bias) :=
  rfl

/-- Evaluation of the inverse affine-parameter linear equivalence. -/
@[simp]
theorem linearEquivProd_symm_apply
    (p : (V →L[ℝ] W) × W) :
    linearEquivProd.symm p =
      {
        weight := p.1
        bias := p.2
      } :=
  rfl

/-- The weight component of the inverse linear equivalence. -/
@[simp]
theorem linearEquivProd_symm_weight
    (p : (V →L[ℝ] W) × W) :
    (linearEquivProd.symm p).weight = p.1 := by
  change
    ({
      weight := p.1
      bias := p.2
    } : AffineParams V W).weight = p.1
  rfl

/-- The bias component of the inverse linear equivalence. -/
@[simp]
theorem linearEquivProd_symm_bias
    (p : (V →L[ℝ] W) × W) :
    (linearEquivProd.symm p).bias = p.2 := by
  change
    ({
      weight := p.1
      bias := p.2
    } : AffineParams V W).bias = p.2
  rfl

/--
The space of continuous linear maps between two representation spaces is
finite-dimensional.
-/
instance instFiniteDimensionalContinuousLinearMap :
    FiniteDimensional ℝ (V →L[ℝ] W) := by
  exact ContinuousLinearMap.finiteDimensional

/--
Affine parameter space is finite-dimensional.

The product of the weight and bias spaces is finite-dimensional. The result
is transported back to affine parameters through `linearEquivProd`.
-/
instance instFiniteDimensional :
    FiniteDimensional ℝ (AffineParams V W) := by
  let e :
      ((V →L[ℝ] W) × W) ≃ₗ[ℝ] AffineParams V W :=
    linearEquivProd.symm
  exact LinearEquiv.finiteDimensional e

end

end AffineParams

namespace Architecture

variable
  {G : Type uG}
  [Group G]

noncomputable section

/-!
### Finite-dimensionality of activation parameters
-/

/--
The trivial parameter space `PUnit` is finite-dimensional.

Its underlying type is finite, so its real-module structure is finitely
generated.
-/
instance instPUnitFiniteDimensional :
    FiniteDimensional ℝ PUnit := by
  infer_instance

/-!
### Recursive finite-dimensionality
-/

/--
The parameter space of every finite architecture is finite-dimensional.

The proof follows the architecture recursively:

* affine parameters are finite-dimensional;
* activation parameters form the trivial finite-dimensional space;
* a composition uses the product of its two parameter spaces.
-/
theorem paramsFiniteDimensional
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W) :
    FiniteDimensional ℝ (Params A) :=
  match A with
  | .affine _ _ =>
      AffineParams.instFiniteDimensional
  | .activation _ =>
      instPUnitFiniteDimensional
  | .comp first second =>
      letI : AddCommGroup (Params first) :=
        paramsAddCommGroup first
      letI : AddCommGroup (Params second) :=
        paramsAddCommGroup second
      letI : Module ℝ (Params first) :=
        paramsModule first
      letI : Module ℝ (Params second) :=
        paramsModule second
      letI : FiniteDimensional ℝ (Params first) :=
        paramsFiniteDimensional first
      letI : FiniteDimensional ℝ (Params second) :=
        paramsFiniteDimensional second
      inferInstanceAs
        (FiniteDimensional ℝ (Params first × Params second))

/--
Install finite-dimensionality for the parameter type of every fixed
architecture.
-/
instance instParamsFiniteDimensional
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W) :
    FiniteDimensional ℝ (Params A) :=
  paramsFiniteDimensional A

end

end Architecture

end EquivariantNN


namespace EquivariantNN

/-!
## Smooth activation specifications

Equivariance and smoothness are deliberately separated.

An `ActivationSpec` stores only equivariance. The predicate below adds
smoothness when calculus is required.
-/

namespace ActivationSpec

variable
  {G : Type uG}
  [Group G]

variable
  {V W Z : RepSpace.{uG, uV} G}

/-- An activation specification is continuously differentiable to order `n`. -/
def IsContDiff
    (n : WithTop ℕ∞)
    (σ : ActivationSpec G V W) :
    Prop :=
  ContDiff ℝ n σ.toFun

/-- A smooth activation is differentiable. -/
theorem differentiable
    (σ : ActivationSpec G V W)
    (hσ : σ.IsContDiff (⊤ : WithTop ℕ∞)) :
    Differentiable ℝ σ.toFun := by
  exact hσ.differentiable (by simp)

/-- A smooth activation is continuous. -/
theorem continuous
    (σ : ActivationSpec G V W)
    (hσ : σ.IsContDiff (⊤ : WithTop ℕ∞)) :
    Continuous σ.toFun :=
  hσ.continuous

/-- The identity activation is continuously differentiable to every order. -/
theorem id_isContDiff
    (V : RepSpace.{uG, uV} G)
    (n : WithTop ℕ∞) :
    (ActivationSpec.id V).IsContDiff n := by
  exact contDiff_id

/-- Composition preserves continuous differentiability. -/
theorem comp_isContDiff
    (τ : ActivationSpec G W Z)
    (σ : ActivationSpec G V W)
    (n : WithTop ℕ∞)
    (hτ : τ.IsContDiff n)
    (hσ : σ.IsContDiff n) :
    (τ.comp σ).IsContDiff n := by
  change ContDiff ℝ n (fun x : V => τ.toFun (σ.toFun x))
  exact ContDiff.comp hτ hσ

/-- Smoothness is monotone in the differentiability order. -/
theorem isContDiff_of_le
    (σ : ActivationSpec G V W)
    {m n : WithTop ℕ∞}
    (hσ : σ.IsContDiff n)
    (hmn : m ≤ n) :
    σ.IsContDiff m := by
  exact ContDiff.of_le hσ hmn

end ActivationSpec

/-!
## Smooth architectures

An architecture is continuously differentiable to order `n` when every
activation occurring in it is continuously differentiable to order `n`.

This predicate does not yet assert that realization is smooth jointly in
parameters and inputs. That theorem will be proved later, after the parameter
spaces have their required normed structures.
-/

namespace Architecture

variable
  {G : Type uG}
  [Group G]

variable
  {U V W : RepSpace.{uG, uV} G}

/--
An architecture is smooth to order `n` when every activation appearing in
the architecture is smooth to order `n`.
-/
def IsContDiff
    (n : WithTop ℕ∞)
    {V W : RepSpace.{uG, uV} G} :
    Architecture G V W → Prop
  | .affine _ _ =>
      True
  | .activation σ =>
      σ.IsContDiff n
  | .comp first second =>
      first.IsContDiff n ∧ second.IsContDiff n

/-- Every affine architecture is smooth to every order. -/
@[simp]
theorem affine_isContDiff
    (V W : RepSpace.{uG, uV} G)
    (n : WithTop ℕ∞) :
    (Architecture.affine V W).IsContDiff n := by
  trivial

/-- Smoothness of an activation architecture reduces to its activation. -/
@[simp]
theorem activation_isContDiff_iff
    (σ : ActivationSpec G V W)
    (n : WithTop ℕ∞) :
    (Architecture.activation σ).IsContDiff n ↔
      σ.IsContDiff n :=
  Iff.rfl

/-- Smoothness of a composition reduces to smoothness of both components. -/
@[simp]
theorem comp_isContDiff_iff
    (first : Architecture G U V)
    (second : Architecture G V W)
    (n : WithTop ℕ∞) :
    (Architecture.comp first second).IsContDiff n ↔
      first.IsContDiff n ∧ second.IsContDiff n :=
  Iff.rfl

/-- Extract smoothness of the first component of a composition. -/
theorem IsContDiff.first
    {first : Architecture G U V}
    {second : Architecture G V W}
    {n : WithTop ℕ∞}
    (h : (Architecture.comp first second).IsContDiff n) :
    first.IsContDiff n :=
  h.1

/-- Extract smoothness of the second component of a composition. -/
theorem IsContDiff.second
    {first : Architecture G U V}
    {second : Architecture G V W}
    {n : WithTop ℕ∞}
    (h : (Architecture.comp first second).IsContDiff n) :
    second.IsContDiff n :=
  h.2

/-- Combine smoothness proofs for two composed architectures. -/
theorem IsContDiff.comp
    {first : Architecture G U V}
    {second : Architecture G V W}
    {n : WithTop ℕ∞}
    (hFirst : first.IsContDiff n)
    (hSecond : second.IsContDiff n) :
    (Architecture.comp first second).IsContDiff n :=
  ⟨hFirst, hSecond⟩

/--
Smoothness is monotone in the requested differentiability order.

If an architecture is `Cⁿ`, it is `Cᵐ` whenever `m ≤ n`.
-/
theorem isContDiff_of_le
    {V W : RepSpace.{uG, uV} G}
    (A : Architecture G V W)
    {m n : WithTop ℕ∞}
    (hA : A.IsContDiff n)
    (hmn : m ≤ n) :
    A.IsContDiff m := by
  induction A with
  | affine V W =>
      trivial
  | activation σ =>
      exact ActivationSpec.isContDiff_of_le σ hA hmn
  | comp first second ihFirst ihSecond =>
      exact ⟨ihFirst hA.1, ihSecond hA.2⟩

end Architecture

/-!
## Smooth invariant losses

The algebraic invariance and analytic smoothness assumptions are bundled
together, while remaining separate fields.

The earlier augmented-risk invariance theorem continues to use only the
`invariant` field.
-/

/--
A diagonally invariant loss that is continuously differentiable to order
`n` as a function on prediction-target pairs.
-/
structure SmoothInvariantLoss
    (G : Type uG)
    [Group G]
    (Y : RepSpace.{uG, uV} G)
    (n : WithTop ℕ∞) where
  toFun : Y → Y → ℝ
  invariant : IsInvariantLoss G toFun
  contDiff :
    ContDiff ℝ n
      (fun z : Y × Y => toFun z.1 z.2)


universe uData
namespace SmoothInvariantLoss

variable
  {G : Type uG}
  [Group G]

variable
  {Y : RepSpace.{uG, uV} G}
  {n : WithTop ℕ∞}

/-- A smooth invariant loss can be applied as a two-argument function. -/
instance instCoeFun :
    CoeFun
      (SmoothInvariantLoss G Y n)
      (fun _ => Y → Y → ℝ) where
  coe L := L.toFun

/-- The defining diagonal invariance equation. -/
@[simp]
theorem map_smul
    (L : SmoothInvariantLoss G Y n)
    (g : G)
    (y y' : Y) :
    L (g • y) (g • y') = L y y' :=
  L.invariant g y y'

/-- The underlying loss satisfies the algebraic invariance predicate. -/
theorem isInvariantLoss
    (L : SmoothInvariantLoss G Y n) :
    IsInvariantLoss G L.toFun :=
  L.invariant

/-- The loss, regarded as a function on a product, is smooth. -/
theorem contDiff_prod
    (L : SmoothInvariantLoss G Y n) :
    ContDiff ℝ n
      (fun z : Y × Y => L z.1 z.2) :=
  L.contDiff

/--
A smooth invariant loss inherits augmented-risk invariance from the generic
algebraic theorem.

Only the `invariant` field is used.
-/
theorem augmentedRisk_invariant
    {ι : Type uData}
    [Fintype ι]
    [Fintype G]
    {X : RepSpace.{uG, uV} G}
    (L : SmoothInvariantLoss G Y n)
    (A : Architecture G X Y)
    (D : Dataset ι X Y)
    (g : G)
    (p : Architecture.Params A) :
    Architecture.augmentedRisk A D L.toFun (g • p) =
      Architecture.augmentedRisk A D L.toFun p := by
  exact Architecture.augmentedRisk_invariant
    A D L.toFun L.invariant g p

end SmoothInvariantLoss

end EquivariantNN
namespace EquivariantNN

/-!
## Parameter-space geometry

To define gradients, the parameter type of an architecture must be a complete
real inner-product space.

Moreover, the induced group action on parameters must agree with a
representation by real-linear isometric equivalences.

This structure records exactly those geometric properties. It does not assume
gradient covariance, risk invariance, or flow covariance.

A later section will construct this structure using the Hilbert-Schmidt inner
product on affine weight spaces and product inner products on composed
architectures.
-/

class ParameterGeometry
    {G : Type uG}
    [Group G]
    {X Y : RepSpace.{uG, uV} G}
    (A : Architecture G X Y)
    extends NormedAddCommGroup (Architecture.Params A),
      InnerProductSpace ℝ (Architecture.Params A),
      CompleteSpace (Architecture.Params A) where
  actionLinearIsometry :
    G →* (Architecture.Params A ≃ₗᵢ[ℝ] Architecture.Params A)
  actionLinearIsometry_apply :
    ∀ (g : G) (p : Architecture.Params A),
      actionLinearIsometry g p = g • p

namespace ParameterGeometry

variable
  {G : Type uG}
  [Group G]

variable
  {X Y : RepSpace.{uG, uV} G}

variable
  {A : Architecture G X Y}

variable
  [ParameterGeometry A]

/--
The linear isometry representing a group element on the parameter space.
-/
def representation
    (A : Architecture G X Y)
    [ParameterGeometry A]
    (g : G) :
    Architecture.Params A ≃ₗᵢ[ℝ] Architecture.Params A :=
  ParameterGeometry.actionLinearIsometry g

/--
The geometric representation agrees with the previously defined recursive
action on network parameters.
-/
@[simp]
theorem representation_apply
    (g : G)
    (p : Architecture.Params A) :
    representation A g p = g • p :=
  ParameterGeometry.actionLinearIsometry_apply g p

/--
The parameter action preserves norms.
-/
@[simp]
theorem norm_smul
    (g : G)
    (p : Architecture.Params A) :
    ‖g • p‖ = ‖p‖ := by
  rw [← representation_apply (A := A) g p]
  exact (representation A g).norm_map p

/--
The parameter action preserves distances.
-/
@[simp]
theorem dist_smul_smul
    (g : G)
    (p q : Architecture.Params A) :
    dist (g • p) (g • q) = dist p q := by
  rw [
    ← representation_apply (A := A) g p,
    ← representation_apply (A := A) g q
  ]
  exact (representation A g).dist_map p q

/--
The parameter action preserves the inner product.
-/
@[simp]
theorem inner_smul_smul
    (g : G)
    (p q : Architecture.Params A) :
    inner ℝ (g • p) (g • q) =
      inner ℝ p q := by
  rw [
    ← representation_apply (A := A) g p,
    ← representation_apply (A := A) g q
  ]
  exact (representation A g).inner_map_map p q

/--
The action on the parameter space is continuous.
-/
theorem continuous_smul
    (g : G) :
    Continuous
      (fun p : Architecture.Params A => g • p) := by
  have h :
      Continuous
        (fun p : Architecture.Params A =>
          representation A g p) :=
    (representation A g).continuous
  simpa only [representation_apply] using h

/--
The parameter action is an isometry.
-/
theorem isometry_smul
    (g : G) :
    Isometry
      (fun p : Architecture.Params A => g • p) := by
  intro p q
  exact dist_smul_smul g p q

/--
The parameter action is injective.

This also follows algebraically from the group action, but the geometric
proof records that it is an isometry.
-/
theorem smul_injective
    (g : G) :
    Function.Injective
      (fun p : Architecture.Params A => g • p) :=
  (isometry_smul g).injective

/--
The parameter action is surjective.
-/
theorem smul_surjective
    (g : G) :
    Function.Surjective
      (fun p : Architecture.Params A => g • p) := by
  intro p
  refine ⟨g⁻¹ • p, ?_⟩
  exact smul_inv_smul g p

/--
The parameter action is bijective.
-/
theorem smul_bijective
    (g : G) :
    Function.Bijective
      (fun p : Architecture.Params A => g • p) :=
  ⟨smul_injective g, smul_surjective g⟩

/--
The previously constructed linear action agrees with the linear map
underlying the geometric representation.

This connects the algebraic parameter-action development to the Hilbert-space
geometry used for gradients.
-/
theorem paramsActionLinearMap_eq
    (g : G) :
    Architecture.paramsActionLinearMap A g =
      (representation A g).toLinearEquiv.toLinearMap := by
  apply LinearMap.ext
  intro p
  simp only [
    Architecture.paramsActionLinearMap_apply,
    LinearIsometryEquiv.coe_toLinearEquiv,
    representation_apply
  ]

/--
The previously constructed linear equivalence agrees with the linear
equivalence underlying the geometric representation.
-/
theorem paramsActionLinearEquiv_eq
    (g : G) :
    Architecture.paramsActionLinearEquiv A g =
      (representation A g).toLinearEquiv := by
  apply LinearEquiv.ext
  intro p
  simp only [
    Architecture.paramsActionLinearEquiv_apply,
    LinearIsometryEquiv.coe_toLinearEquiv,
    representation_apply
  ]

/--
Moving a group action from the first argument of the inner product to the
second introduces the inverse group element.

This identity will be useful in the gradient-covariance proof.
-/
theorem inner_smul_left
    (g : G)
    (p q : Architecture.Params A) :
    inner ℝ (g • p) q =
      inner ℝ p (g⁻¹ • q) := by
  calc
    inner ℝ (g • p) q =
        inner ℝ (g • p) (g • (g⁻¹ • q)) := by
          rw [smul_inv_smul]
    _ =
        inner ℝ p (g⁻¹ • q) := by
          exact inner_smul_smul g p (g⁻¹ • q)

/--
Moving a group action from the second argument of the inner product to the
first introduces the inverse group element.
-/
theorem inner_smul_right
    (g : G)
    (p q : Architecture.Params A) :
    inner ℝ p (g • q) =
      inner ℝ (g⁻¹ • p) q := by
  calc
    inner ℝ p (g • q) =
        inner ℝ (g • (g⁻¹ • p)) (g • q) := by
          rw [smul_inv_smul]
    _ =
        inner ℝ (g⁻¹ • p) q := by
          exact inner_smul_smul g (g⁻¹ • p) q

end ParameterGeometry

end EquivariantNN
