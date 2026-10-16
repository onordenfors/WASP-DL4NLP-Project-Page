import Mathlib.Tactic

section EquivariantNN

-- 1. Custom Group Representation class with explicit G parameter
class GroupRep (G : Type*) [Group G] (V : Type*) where
  act : G → V → V
  act_one : ∀ v : V, act 1 v = v
  act_mul : ∀ (g h : G) (v : V), act (g * h) v = act g (act h v)

export GroupRep (act act_one act_mul)

-- 2. Representation on Hom spaces (Linear maps)
instance homGroupRep (G : Type*) [Group G] (V W : Type*)
    [GroupRep G V] [GroupRep G W] : GroupRep G (V → W) where
  act g theta v := act g (theta (act g⁻¹ v))
  act_one theta := by
    ext v
    rw [inv_one, act_one, act_one]
  act_mul g h theta := by
    ext v
    dsimp
    rw [mul_inv_rev]
    exact act_mul g h (theta (act h⁻¹ (act g⁻¹ v)))

-- 3. Pi representation for parameter tuples (Theta parameter space)
instance piGroupRep (G : Type*) [Group G] {I : Type*} (S : I → Type*)
    [∀ i, GroupRep G (S i)] : GroupRep G (Π i, S i) where
  act g theta i := act g (theta i)
  act_one theta := by
    ext i
    apply act_one
  act_mul g h theta := by
    ext i
    apply act_mul

-- 4. Equivariant activation property
def IsEquivariant (G : Type*) [Group G] {V W : Type*}
    [GroupRep G V] [GroupRep G W] (sigma : V → W) : Prop :=
  ∀ (g : G) (v : V), sigma (act g v) = act g (sigma v)

-- Helper lemmas for group action cancellation
lemma act_cancel (G : Type*) [Group G] {V : Type*} [GroupRep G V] (g : G) (v : V) :
    act g (act g⁻¹ v) = v := by
  rw [← act_mul, mul_inv_cancel, act_one]

lemma act_inv_cancel (G : Type*) [Group G] {V : Type*} [GroupRep G V] (g : G) (v : V) :
    act g⁻¹ (act g v) = v := by
  rw [← act_mul, inv_mul_cancel, act_one]

-- 5. L-Layer Network Evaluation with Linear Layers and Intermediate Equivariant Activations
def evalNet (G : Type*) [Group G] : ∀ {L : ℕ} (X : Fin L → Type*) [inst : ∀ i, GroupRep G (X i)],
    (Π (i : Fin (L - 1)), X i.castSucc → X i.succ) →
    (∀ (i : Fin (L - 2)), X i.succ → X i.succ) →
    X 0 → X (Fin.last (L - 1))
  | _, 0, _, _, _, _, x => x
  | _, 1, _, _, _, _, x => x
  | _, 2, _, _, theta, _, x => theta ⟨0, by omega⟩ x
  | G, L' + 3, X, inst, theta, sigma, x =>
    let x1 := theta ⟨0, by omega⟩ x
    let x1_act := sigma ⟨0, by omega⟩ x1
    let X_tail (i : Fin (L' + 2)) : Type* := X ⟨i.1 + 1, by omega⟩
    have inst_tail (i : Fin (L' + 2)) : GroupRep G (X_tail i) := inst ⟨i.1 + 1, by omega⟩
    let theta_tail (i : Fin (L' + 1)) : X_tail i.castSucc → X_tail i.succ :=
      fun v => theta ⟨i.1 + 1, by omega⟩ v
    let sigma_tail (i : Fin L') : X_tail i.succ → X_tail i.succ :=
      fun v => sigma ⟨i.1 + 1, by omega⟩ v
    evalNet G X_tail inst_tail theta_tail sigma_tail x1_act

-- 6. Theorem: Full L-layer network equivariance under parameter representation action
theorem network_equivariant (G : Type*) [Group G] :
    ∀ {L : ℕ} (X : Fin L → Type*) [inst : ∀ i, GroupRep G (X i)]
    (theta : Π (i : Fin (L - 1)), X i.castSucc → X i.succ)
    (sigma : ∀ (i : Fin (L - 2)), X i.succ → X i.succ)
    (h_sigma : ∀ i, IsEquivariant G (sigma i))
    (g : G) (x : X 0),
    evalNet G X theta sigma (act g x) =
    act g (evalNet G X (piGroupRep G _).act g⁻¹ theta sigma x) := by
  intro L
  induction' L using Nat.recOn with L ih
  · intro X inst theta sigma h_sigma g x
    rfl
  · intro X inst theta sigma h_sigma g x
    match L with
    | 0 =>
      dsimp [evalNet]
      rfl
    | 1 =>
      dsimp [evalNet]
      rfl
    | 2 =>
      dsimp [evalNet]
      dsimp [piGroupRep, homGroupRep]
      rw [inv_inv]
      exact act_cancel G g (theta ⟨0, by omega⟩ x)
    | L'' + 3 =>
      dsimp [evalNet]
      have h_sig := h_sigma ⟨0, by omega⟩ g (theta ⟨0, by omega⟩ (act g x))
      let X_tail (i : Fin (L'' + 2)) : Type* := X ⟨i.1 + 1, by omega⟩
      have inst_tail (i : Fin (L'' + 2)) : GroupRep G (X_tail i) := inst ⟨i.1 + 1, by omega⟩
      let theta_tail (i : Fin (L'' + 1)) : X_tail i.castSucc → X_tail i.succ :=
        fun v => theta ⟨i.1 + 1, by omega⟩ v
      let sigma_tail (i : Fin L'') : X_tail i.succ → X_tail i.succ :=
        fun v => sigma ⟨i.1 + 1, by omega⟩ v
      have h_sigma_tail : ∀ i, IsEquivariant G (sigma_tail i) := fun i => h_sigma ⟨i.1 + 1, by omega⟩

      have ih_tail := ih X_tail inst_tail theta_tail sigma_tail h_sigma_tail g (sigma ⟨0, by omega⟩ (theta ⟨0, by omega⟩ (act g x)))
      dsimp [piGroupRep, homGroupRep]
      rw [inv_inv]
      rw [act_cancel]
      rw [← h_sig]
      exact ih_tail

end EquivariantNN
