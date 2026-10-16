import Mathlib

section WeakEquivariant

variable (G : Type*) [Group G]

-- Instance showing X → Y is a valid MulAction under the natural conjugation action
instance {X Y : Type*} [MulAction G X] [MulAction G Y] : MulAction G (X → Y) where
  smul g theta x := g • theta (g⁻¹ • x)
  one_smul theta := by
    ext x
    dsimp [SMul.smul]
    rw [inv_one, one_smul, one_smul]
  mul_smul g h theta := by
    ext x
    dsimp [SMul.smul]
    rw [mul_inv_rev, mul_smul, mul_smul]

-- Definition of weak equivariance for a network layer f : (X → Y) → X → Y
def WeakEquivariant {X Y : Type*} [MulAction G X] [MulAction G Y] (f : (X → Y) → X → Y) : Prop :=
  ∀ (g : G) (theta : X → Y) (x : X), f theta (g • x) = g • f (g⁻¹ • theta) x

-- Theorem: Standard neural network evaluation satisfies weak equivariance from first principles
theorem eval_weak_equiv {X Y : Type*} [MulAction G X] [MulAction G Y] :
    WeakEquivariant G (fun (theta : X → Y) (x : X) => theta x) := by
  intro g theta x
  dsimp [WeakEquivariant, SMul.smul]
  rw [inv_inv]
  rw [← mul_smul]
  rw [mul_inv_cancel, one_smul]

end WeakEquivariant
