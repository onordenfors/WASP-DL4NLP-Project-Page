import Mathlib

universe uG uK uX

namespace EquivariantNetwork

/--
A linear representation written explicitly as an action.

This is equivalent to the usual presentation as a group homomorphism
`G →* GL(V)`, but its fields make the calculations in this file
particularly transparent.
-/
structure LinRep
    (G : Type uG) (𝕜 : Type uK) (V : Type uX)
    [Group G] [Field 𝕜]
    [AddCommGroup V] [Module 𝕜 V] where
  act : G → V → V
  act_one : act 1 = id
  act_mul : ∀ g h, act (g * h) = act g ∘ act h
  map_add : ∀ g x y, act g (x + y) = act g x + act g y
  map_smul : ∀ g (a : 𝕜) x, act g (a • x) = a • act g x

namespace LinRep

variable
    {G : Type uG} {𝕜 : Type uK} {V : Type uX}
    [Group G] [Field 𝕜]
    [AddCommGroup V] [Module 𝕜 V]

@[simp]
theorem act_one_apply (R : LinRep G 𝕜 V) (x : V) :
    R.act 1 x = x := by
  have h := congrFun R.act_one x
  exact h

@[simp]
theorem act_mul_apply (R : LinRep G 𝕜 V)
    (g h : G) (x : V) :
    R.act (g * h) x = R.act g (R.act h x) := by
  have hgh := congrFun (R.act_mul g h) x
  exact hgh

@[simp]
theorem act_inv_act (R : LinRep G 𝕜 V) (g : G) (x : V) :
    R.act g⁻¹ (R.act g x) = x := by
  rw [← R.act_mul_apply]
  simp

@[simp]
theorem act_act_inv (R : LinRep G 𝕜 V) (g : G) (x : V) :
    R.act g (R.act g⁻¹ x) = x := by
  rw [← R.act_mul_apply]
  simp

/-- The linear map underlying the action of `g`. -/
def linearMap (R : LinRep G 𝕜 V) (g : G) : V →ₗ[𝕜] V where
  toFun := R.act g
  map_add' := R.map_add g
  map_smul' := R.map_smul g

@[simp]
theorem linearMap_apply (R : LinRep G 𝕜 V) (g : G) (x : V) :
    R.linearMap g x = R.act g x :=
  rfl

end LinRep

section Network

variable
    {G : Type uG} {𝕜 : Type uK}
    [Group G] [Field 𝕜]

variable
    (X : ℕ → Type uX)
    [∀ i, AddCommGroup (X i)]
    [∀ i, Module 𝕜 (X i)]

/-- The parameter space: one linear map between each pair of adjacent layers. -/
abbrev Params :=
  ∀ i : ℕ, X i →ₗ[𝕜] X (i + 1)

/--
The conjugation action on parameters:
`(g • θ)ᵢ = ρᵢ₊₁(g) ∘ θᵢ ∘ ρᵢ(g⁻¹)`.
-/
def paramAct
    (R : ∀ i, LinRep G 𝕜 (X i))
    (g : G)
    (θ : Params (𝕜 := 𝕜) X) :
    Params (𝕜 := 𝕜) X :=
  fun i =>
    (R (i + 1)).linearMap g ∘ₗ
      ((θ i) ∘ₗ (R i).linearMap g⁻¹)

/--
The conjugation action is a linear representation on the complete
parameter space.
-/
def parameterRepresentation
    (R : ∀ i, LinRep G 𝕜 (X i)) :
    LinRep G 𝕜 (Params (𝕜 := 𝕜) X) where
  act := paramAct (X := X) R

  act_one := by
    funext θ
    funext i
    ext x
    simp [paramAct]

  act_mul := by
    intro g h
    funext θ
    funext i
    ext x
    simp only [paramAct, LinearMap.comp_apply, LinRep.linearMap_apply]
    rw [LinRep.act_mul_apply]
    rw [mul_inv_rev]
    rw [LinRep.act_mul_apply]
    rfl

  map_add := by
    intro g θ η
    funext i
    ext x
    simp [paramAct, LinRep.map_add]

  map_smul := by
    intro g a θ
    funext i
    ext x
    simp [paramAct]

/--
Evaluation of a network of depth `n`.

After each linear layer `θ i`, the activation `σ i` is applied.
-/
def forward
    (σ : ∀ i, X (i + 1) → X (i + 1))
    (θ : Params (𝕜 := 𝕜) X) :
    ∀ n : ℕ, X 0 → X n
  | 0 => id
  | n + 1 => fun x => σ n (θ n (forward σ θ n x))

@[simp]
theorem forward_zero
    (σ : ∀ i, X (i + 1) → X (i + 1))
    (θ : Params (𝕜 := 𝕜) X)
    (x : X 0) :
    forward X σ θ 0 x = x :=
  rfl

@[simp]
theorem forward_succ
    (σ : ∀ i, X (i + 1) → X (i + 1))
    (θ : Params (𝕜 := 𝕜) X)
    (n : ℕ)
    (x : X 0) :
    forward X σ θ (n + 1) x =
      σ n (θ n (forward X σ θ n x)) :=
  rfl

/--
Auxiliary identity: transforming the parameters by `g⁻¹` transports
the output by `ρₙ(g⁻¹)`.
-/
theorem forward_param_inv
    (R : ∀ i, LinRep G 𝕜 (X i))
    (σ : ∀ i, X (i + 1) → X (i + 1))
    (hσ :
      ∀ i g z,
        σ i ((R (i + 1)).act g z) =
          (R (i + 1)).act g (σ i z))
    (θ : Params (𝕜 := 𝕜) X)
    (g : G)
    (x : X 0) :
    ∀ n,
      forward X σ
          (paramAct (X := X) R g⁻¹ θ) n x =
        (R n).act g⁻¹
          (forward X σ θ n ((R 0).act g x)) := by
  intro n
  induction n with
  | zero =>
      simp [forward]
  | succ n ih =>
      simp only [forward]
      rw [ih]
      simp only [paramAct, LinearMap.comp_apply, LinRep.linearMap_apply]
      simp only [inv_inv]
      rw [LinRep.act_act_inv]
      exact hσ n g⁻¹
        (θ n (forward X σ θ n ((R 0).act g x)))

/--
Main equivariance theorem.

If every activation is equivariant, then the parameter space carries
the conjugation representation and

`f_θ (ρ₀(g) x) = ρₙ(g) (f_{ρ(g⁻¹)θ} x)`.
-/
theorem network_equivariance
    (R : ∀ i, LinRep G 𝕜 (X i))
    (σ : ∀ i, X (i + 1) → X (i + 1))
    (hσ :
      ∀ i g z,
        σ i ((R (i + 1)).act g z) =
          (R (i + 1)).act g (σ i z))
    (θ : Params (𝕜 := 𝕜) X)
    (g : G)
    (x : X 0)
    (n : ℕ) :
    forward X σ θ n ((R 0).act g x) =
      (R n).act g
        (forward X σ
          ((parameterRepresentation (X := X) R).act g⁻¹ θ)
          n x) := by
  have h :=
    congrArg
      (fun z => (R n).act g z)
      (forward_param_inv
        (X := X) R σ hσ θ g x n)
  simpa [parameterRepresentation] using h.symm

end Network

end EquivariantNetwork
