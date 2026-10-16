import Mathlib

/-!
# Gaussian initialization on an invariant affine parameter space

This file isolates the only probabilistic fact needed in the argument:
a centered isotropic Gaussian law is invariant under every linear isometry.
A law is represented extensionally by its expectation functional on all real-valued
test functions. This avoids measurable-space side conditions while retaining the
usual notion of equality in distribution.
-/

open Function

namespace GaussianAffineInvariant

/-- An extensional law, represented by expectations of real-valued test functions. -/
abbrev Law (X : Type*) := (X → ℝ) → ℝ

/-- Push a law forward along a function. -/
def pushforward {X Y : Type*} (f : X → Y) (μ : Law X) : Law Y :=
  fun φ => μ (φ ∘ f)

@[simp] theorem pushforward_id {X : Type*} (μ : Law X) :
    pushforward id μ = μ := by
  rfl

@[simp] theorem pushforward_comp {X Y Z : Type*}
    (f : Y → Z) (g : X → Y) (μ : Law X) :
    pushforward f (pushforward g μ) = pushforward (f ∘ g) μ := by
  rfl

/-- A law is invariant under a transformation when its pushforward is unchanged. -/
def Invariant {X : Type*} (μ : Law X) (T : X → X) : Prop :=
  pushforward T μ = μ

/-- The probabilistic input: `γ` is a standard isotropic Gaussian law.

The field `unitary_invariant` is the familiar theorem that a standard Gaussian is
unchanged by an orthogonal/unitary change of coordinates. Keeping this fact in a
structure makes the affine argument independent of a particular construction of
Gaussian probability measures.
-/
structure StandardGaussianLaw (V : Type*) [NormedAddCommGroup V]
    [InnerProductSpace ℝ V] where
  law : Law V
  unitary_invariant : ∀ U : V ≃ₗᵢ[ℝ] V, Invariant law U

/-- Translation by the chosen affine base point. -/
def translate {V : Type*} [Add V] (p : V) : V → V :=
  fun v => p + v

/-- The affine action obtained from a unitary representation and a fixed point `p`.
It is the conjugate of the linear action by translation by `p`. -/
def affineAction {G V : Type*} [Group G] [NormedAddCommGroup V]
    [InnerProductSpace ℝ V] (ρ : G →* (V ≃ₗᵢ[ℝ] V)) (p : V) (g : G) : V → V :=
  fun x => p + ρ g (x - p)

/-- The fixed point really is fixed by the induced affine action. -/
theorem affineAction_fixed {G V : Type*} [Group G] [NormedAddCommGroup V]
    [InnerProductSpace ℝ V] (ρ : G →* (V ≃ₗᵢ[ℝ] V)) (p : V) (g : G) :
    affineAction ρ p g p = p := by
  simp [affineAction]

/-- Translation intertwines the linear representation with its affine action. -/
theorem affineAction_translate {G V : Type*} [Group G] [NormedAddCommGroup V]
    [InnerProductSpace ℝ V] (ρ : G →* (V ≃ₗᵢ[ℝ] V)) (p : V) (g : G) :
    affineAction ρ p g ∘ translate p = translate p ∘ (ρ g : V → V) := by
  funext v
  simp [affineAction, translate]

/-- **Gaussian affine-invariance theorem.**

Let `ρ` be a unitary representation on `V`, and let `p` be the fixed element used
as the origin of the invariant affine parameter space. Initialize a parameter as
`p + Z`, where `Z` has a standard isotropic Gaussian law. Then the resulting law
is invariant under every affine transformation `x ↦ p + ρ(g)(x - p)`.
-/
theorem gaussian_initialization_invariant
    {G V : Type*} [Group G] [NormedAddCommGroup V] [InnerProductSpace ℝ V]
    (ρ : G →* (V ≃ₗᵢ[ℝ] V)) (p : V) (γ : StandardGaussianLaw V) (g : G) :
    Invariant (pushforward (translate p) γ.law) (affineAction ρ p g) := by
  unfold Invariant
  rw [pushforward_comp]
  rw [affineAction_translate]
  rw [← pushforward_comp]
  rw [γ.unitary_invariant (ρ g)]

/-- Pointwise version: the affine Gaussian law is invariant for every group element. -/
theorem gaussian_initialization_invariant_all
    {G V : Type*} [Group G] [NormedAddCommGroup V] [InnerProductSpace ℝ V]
    (ρ : G →* (V ≃ₗᵢ[ℝ] V)) (p : V) (γ : StandardGaussianLaw V) :
    ∀ g : G, Invariant (pushforward (translate p) γ.law) (affineAction ρ p g) := by
  intro g
  exact gaussian_initialization_invariant ρ p γ g

end GaussianAffineInvariant
