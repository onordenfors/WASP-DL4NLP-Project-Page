import Mathlib

open scoped BigOperators

namespace AugmentedRisk

universe u p x y

variable {G : Type u} {Params : Type p} {X : Type x} {Y : Type y}

/-- Ordinary sample risk as a function of the network parameters. -/
def risk (loss : Y → Y → ℝ) (network : Params → X → Y)
    (θ : Params) (x₀ : X) (y₀ : Y) : ℝ :=
  loss (network θ x₀) y₀

/--
Augmented sample risk: the group acts on the input and target output,
not directly in the definition on the parameters.
-/
noncomputable def augmentedRisk [Group G] [Fintype G]
    [MulAction G X] [MulAction G Y]
    (loss : Y → Y → ℝ) (network : Params → X → Y)
    (θ : Params) (x₀ : X) (y₀ : Y) : ℝ :=
  (∑ g : G, loss (network θ (g • x₀)) (g • y₀)) /
    (Fintype.card G : ℝ)

/--
The parameter-orbit average of ordinary risk.

The orbit is indexed by `g⁻¹`. Since inversion is a bijection of a group,
this gives the same uniformly indexed orbit as indexing by `g`.
-/
noncomputable def parameterOrbitAverage [Group G] [Fintype G]
    [MulAction G Params]
    (loss : Y → Y → ℝ) (network : Params → X → Y)
    (θ : Params) (x₀ : X) (y₀ : Y) : ℝ :=
  (∑ g : G, risk loss network (g⁻¹ • θ) x₀ y₀) /
    (Fintype.card G : ℝ)

/--
Network transport assumption.

Transforming the input can be moved through the network at the cost of:

1. transforming the network output by `g`; and
2. moving the parameters by `g⁻¹`.
-/
def NetworkTransport [Group G]
    [MulAction G Params] [MulAction G X] [MulAction G Y]
    (network : Params → X → Y) : Prop :=
  ∀ (g : G) (θ : Params) (x₀ : X),
    network θ (g • x₀) =
      g • network (g⁻¹ • θ) x₀

/--
The loss is invariant under the simultaneous action on the prediction
and target.
-/
def InvariantLoss [Group G] [MulAction G Y]
    (loss : Y → Y → ℝ) : Prop :=
  ∀ (g : G) (ŷ y₀ : Y),
    loss (g • ŷ) (g • y₀) = loss ŷ y₀

/--
Under network transport and loss invariance, augmentation in input-output
space equals averaging the ordinary risk along the parameter orbit.
-/
theorem augmentedRisk_eq_parameterOrbitAverage
    [Group G] [Fintype G]
    [MulAction G Params] [MulAction G X] [MulAction G Y]
    (loss : Y → Y → ℝ)
    (network : Params → X → Y)
    (hnetwork : NetworkTransport (G := G) network)
    (hloss : InvariantLoss (G := G) loss)
    (θ : Params) (x₀ : X) (y₀ : Y) :
    augmentedRisk (G := G) loss network θ x₀ y₀ =
      parameterOrbitAverage (G := G) loss network θ x₀ y₀ := by
  unfold augmentedRisk parameterOrbitAverage risk
  congr 1
  apply Finset.sum_congr rfl
  intro g hg
  rw [hnetwork g θ x₀]
  exact hloss g (network (g⁻¹ • θ) x₀) y₀

/--
Every parameter occurring in the parameter-orbit average belongs to
the orbit of the original parameter.
-/
theorem inverse_smul_mem_parameterOrbit
    [Group G] [MulAction G Params]
    (g : G) (θ : Params) :
    g⁻¹ • θ ∈ MulAction.orbit G θ := by
  exact ⟨g⁻¹, rfl⟩

end AugmentedRisk
