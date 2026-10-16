import Mathlib

/-!
# Weak equivariance of neural networks with equivariant activations

`G` is a group.  Every layer space `X` is a vector space with a linear `G`-action
`X.rho : G →* End X`.  A network is a chain of layers `x ↦ σ (W x + b)` with
`G`-equivariant activations `σ`.  The weights `θ = (W₁, b₁, …, W_L, b_L)` are stored
inside the network, and `G` acts on them layer by layer by

  `g • (W, b) = (ρ_out g ∘ W ∘ ρ_in g⁻¹ ,  ρ_out g b)`.

Theorem: `f_θ (g x) = g f_{g⁻¹ • θ} (x)`.
-/

set_option autoImplicit false

universe u

namespace WeakEquivariance

/-- A vector space together with a linear action of the group `G`. -/
structure GRep (k G : Type u) [Field k] [Group G] where
  carrier : Type u
  [instAddCommGroup : AddCommGroup carrier]
  [instModule : Module k carrier]
  rho : Representation k G carrier

attribute [instance] GRep.instAddCommGroup GRep.instModule

variable {k G : Type u} [Field k] [Group G]

/-- A `G`-equivariant activation function on a representation. -/
structure EquivAct (X : GRep k G) where
  toFun : X.carrier → X.carrier
  equivariant : ∀ (g : G) (v : X.carrier), toFun (X.rho g v) = X.rho g (toFun v)

theorem rho_inv_self (X : GRep k G) (g : G) (v : X.carrier) :
    X.rho g⁻¹ (X.rho g v) = v := by
  have h : X.rho g⁻¹ * X.rho g = 1 := by
    rw [← map_mul]
    simp
  exact congrArg (fun f : Module.End k X.carrier => f v) h

theorem rho_self_inv (X : GRep k G) (g : G) (v : X.carrier) :
    X.rho g (X.rho g⁻¹ v) = v := by
  have h := rho_inv_self X g⁻¹ v
  rwa [inv_inv] at h

/-- A network `X → Y₁ → ⋯ → Z` with its parameters: each layer consists of a linear map `W`,
a bias `b`, and an equivariant activation `σ` on the target space. -/
inductive Net : GRep k G → GRep k G → Type (u + 1)
  | nil (X : GRep k G) : Net X X
  | cons (X Y Z : GRep k G) (σ : EquivAct Y) (W : X.carrier →ₗ[k] Y.carrier) (b : Y.carrier)
      (N : Net Y Z) : Net X Z

/-- The function `f_θ` computed by the network. -/
def Net.eval : {X Z : GRep k G} → Net X Z → X.carrier → Z.carrier
  | _, _, .nil _, x => x
  | _, _, .cons _ _ _ σ W b N, x => Net.eval N (σ.toFun (W x + b))

/-- The action of `g` on the parameters:
`g • (W, b) = (ρ_out(g) ∘ W ∘ ρ_in(g⁻¹), ρ_out(g) b)`, layer by layer. -/
def Net.act (g : G) : {X Z : GRep k G} → Net X Z → Net X Z
  | _, _, .nil X => .nil X
  | _, _, .cons X Y Z σ W b N =>
      .cons X Y Z σ (Y.rho g ∘ₗ W ∘ₗ X.rho g⁻¹) (Y.rho g b) (Net.act g N)

/-- Acting by `g` after acting by `g⁻¹` is the identity on parameters. -/
theorem Net.act_inv_cancel (g : G) :
    ∀ {X Z : GRep k G} (N : Net X Z), Net.act g (Net.act g⁻¹ N) = N
  | _, _, .nil _ => rfl
  | _, _, .cons X Y Z σ W b N => by
      change Net.cons X Y Z σ
          (Y.rho g ∘ₗ (Y.rho g⁻¹ ∘ₗ W ∘ₗ X.rho g⁻¹⁻¹) ∘ₗ X.rho g⁻¹)
          (Y.rho g (Y.rho g⁻¹ b)) (Net.act g (Net.act g⁻¹ N))
        = Net.cons X Y Z σ W b N
      have e1 : Y.rho g ∘ₗ (Y.rho g⁻¹ ∘ₗ W ∘ₗ X.rho g⁻¹⁻¹) ∘ₗ X.rho g⁻¹ = W := by
        apply LinearMap.ext
        intro v
        simp [LinearMap.comp_apply]
      have e2 : Y.rho g (Y.rho g⁻¹ b) = b := rho_self_inv Y g b
      have e3 : Net.act g (Net.act g⁻¹ N) = N := Net.act_inv_cancel g N
      rw [e1, e2, e3]

/-- Key lemma, by induction on the depth: `f_{g • θ}(g x) = g f_θ(x)`. -/
theorem Net.eval_act (g : G) :
    ∀ {X Z : GRep k G} (N : Net X Z) (x : X.carrier),
      (Net.act g N).eval (X.rho g x) = Z.rho g (N.eval x)
  | _, _, .nil _, _ => rfl
  | _, _, .cons X Y Z σ W b N, x => by
      have h1 : (Y.rho g ∘ₗ W ∘ₗ X.rho g⁻¹) (X.rho g x) + Y.rho g b
          = Y.rho g (W x + b) := by
        simp [LinearMap.comp_apply]
      change Net.eval (Net.act g N)
            (σ.toFun ((Y.rho g ∘ₗ W ∘ₗ X.rho g⁻¹) (X.rho g x) + Y.rho g b))
          = Z.rho g (Net.eval N (σ.toFun (W x + b)))
      rw [h1, σ.equivariant]
      exact Net.eval_act g N _

/-- **Weak equivariance.**  For every network `f_θ` with equivariant activations,
`f_θ (g x) = g f_{g⁻¹ θ} (x)`. -/
theorem weak_equivariance {X Z : GRep k G} (N : Net X Z) (g : G) (x : X.carrier) :
    N.eval (X.rho g x) = Z.rho g ((Net.act g⁻¹ N).eval x) := by
  have h := Net.eval_act g (Net.act g⁻¹ N) x
  rw [Net.act_inv_cancel g N] at h
  exact h

end WeakEquivariance
