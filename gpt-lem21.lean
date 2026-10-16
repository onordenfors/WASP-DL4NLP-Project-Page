import Mathlib

/-!
# Weak equivariance of neural networks of arbitrary depth

All feature spaces in this version are represented by a single type `X`
with a `G`-action. The layers and activations may vary with the layer
index.

The parameters of an `L`-layer network are represented by

    Fin L → Θ.

Thus `θ i` is the parameter of layer `i`.

The main result is

    network θ (g • x) = g • network (g⁻¹ • θ) x

for every number `L` of layers.
-/

universe uG uX uΘ

/--
A parametrized map `X → Y` satisfying weak equivariance:

    F θ (g • x) = g • F (g⁻¹ • θ) x.
-/
structure WELayer
    (G : Type uG)
    (X : Type uX)
    (Y : Type uX)
    (Θ : Type uΘ)
    [Group G]
    [MulAction G X]
    [MulAction G Y]
    [MulAction G Θ] where
  toFun : Θ → X → Y
  weak_equivariant :
    ∀ (g : G) (θ : Θ) (x : X),
      toFun θ (g • x) =
        g • toFun (g⁻¹ • θ) x

namespace WELayer

variable
    {G : Type uG}
    {X Y Z : Type uX}
    {Θ Η : Type uΘ}

variable
    [Group G]
    [MulAction G X]
    [MulAction G Y]
    [MulAction G Z]
    [MulAction G Θ]
    [MulAction G Η]

/--
A `WELayer` can be used as a function of its parameter and input.
-/
instance : CoeFun (WELayer G X Y Θ) (fun _ => Θ → X → Y) where
  coe F := F.toFun

/--
The defining weak-equivariance property, written using function notation.
-/
theorem map_smul
    (F : WELayer G X Y Θ)
    (g : G)
    (θ : Θ)
    (x : X) :
    F θ (g • x) =
      g • F (g⁻¹ • θ) x :=
  F.weak_equivariant g θ x

/--
Postcomposition with an equivariant activation preserves weak
equivariance.
-/
def postActivation
    (F : WELayer G X Y Θ)
    (σ : Y → Z)
    (hσ :
      ∀ (g : G) (y : Y),
        σ (g • y) = g • σ y) :
    WELayer G X Z Θ where
  toFun θ x :=
    σ (F θ x)
  weak_equivariant := by
    intro g θ x
    rw [F.map_smul]
    exact hσ g (F (g⁻¹ • θ) x)

@[simp]
theorem postActivation_apply
    (F : WELayer G X Y Θ)
    (σ : Y → Z)
    (hσ :
      ∀ (g : G) (y : Y),
        σ (g • y) = g • σ y)
    (θ : Θ)
    (x : X) :
    F.postActivation σ hσ θ x =
      σ (F θ x) :=
  rfl

/--
The composition of two weakly equivariant parametrized maps is weakly
equivariant. The parameter space of the composite is the product of
the two parameter spaces.
-/
def comp
    (F₂ : WELayer G Y Z Η)
    (F₁ : WELayer G X Y Θ) :
    WELayer G X Z (Θ × Η) where
  toFun θ x :=
    F₂ θ.2 (F₁ θ.1 x)
  weak_equivariant := by
    intro g θ x
    rw [F₁.map_smul]
    rw [F₂.map_smul]
    rfl

@[simp]
theorem comp_apply
    (F₂ : WELayer G Y Z Η)
    (F₁ : WELayer G X Y Θ)
    (θ : Θ × Η)
    (x : X) :
    F₂.comp F₁ θ x =
      F₂ θ.2 (F₁ θ.1 x) :=
  rfl

end WELayer

/-!
## Networks with an arbitrary number of layers

An `L`-layer network has parameter space

    Fin L → Θ.

The action on this parameter space is the pointwise action:

    (g • θ) i = g • θ i.

This `MulAction` instance is already provided by Mathlib, so no
custom action instance is required.
-/

namespace NeuralNetwork

variable
    {G : Type uG}
    {X : Type uX}
    {Θ : Type uΘ}

variable
    [Group G]
    [MulAction G X]
    [MulAction G Θ]

/--
The parameter space of a network with `L` layers.

An element `θ : Parameters Θ L` assigns one parameter `θ i : Θ`
to each layer `i : Fin L`.
-/
abbrev Parameters (Θ : Type uΘ) (L : Nat) :=
  Fin L → Θ

/--
Recursively evaluate the first `L` layers.

For zero layers the network is the identity.

For `L + 1` layers, first evaluate the first `L` layers and then
apply layer `L`. The parameter of the final layer is

    θ (Fin.last L),

while the parameters of the preceding layers are obtained by restricting
`θ` along `Fin.castSucc`.
-/
def evalNetwork
    (layers : Nat → WELayer G X X Θ) :
    (L : Nat) → Parameters Θ L → X → X
  | 0, _, x =>
      x
  | L + 1, θ, x =>
      layers L
        (θ (Fin.last L))
        (evalNetwork layers L
          (fun i => θ i.castSucc)
          x)

@[simp]
theorem evalNetwork_zero
    (layers : Nat → WELayer G X X Θ)
    (θ : Parameters Θ 0)
    (x : X) :
    evalNetwork layers 0 θ x = x :=
  rfl

@[simp]
theorem evalNetwork_succ
    (layers : Nat → WELayer G X X Θ)
    (L : Nat)
    (θ : Parameters Θ (L + 1))
    (x : X) :
    evalNetwork layers (L + 1) θ x =
      layers L
        (θ (Fin.last L))
        (evalNetwork layers L
          (fun i => θ i.castSucc)
          x) :=
  rfl

/--
Weak equivariance for a network with an arbitrary number `L` of layers:

    evalNetwork layers L θ (g • x)
      =
    g • evalNetwork layers L (g⁻¹ • θ) x.

The proof is by induction on the number of layers.
-/
theorem evalNetwork_weak_equivariant
    (layers : Nat → WELayer G X X Θ)
    (L : Nat)
    (g : G)
    (θ : Parameters Θ L)
    (x : X) :
    evalNetwork layers L θ (g • x) =
      g • evalNetwork layers L (g⁻¹ • θ) x := by
  induction L with
  | zero =>
      rfl
  | succ L ih =>
      change
        layers L
            (θ (Fin.last L))
            (evalNetwork layers L
              (fun i => θ i.castSucc)
              (g • x))
          =
        g •
          layers L
            ((g⁻¹ • θ) (Fin.last L))
            (evalNetwork layers L
              (fun i => (g⁻¹ • θ) i.castSucc)
              x)
      rw [ih]
      rw [(layers L).map_smul]
      rfl

/--
Package the recursively defined `L`-layer network as one weakly
equivariant parametrized map.
-/
def network
    (layers : Nat → WELayer G X X Θ)
    (L : Nat) :
    WELayer G X X (Parameters Θ L) where
  toFun θ x :=
    evalNetwork layers L θ x
  weak_equivariant := by
    intro g θ x
    exact evalNetwork_weak_equivariant
      layers L g θ x

@[simp]
theorem network_apply
    (layers : Nat → WELayer G X X Θ)
    (L : Nat)
    (θ : Parameters Θ L)
    (x : X) :
    network layers L θ x =
      evalNetwork layers L θ x :=
  rfl

/--
The final weak-equivariance theorem for a neural network of arbitrary
depth.
-/
theorem neural_network_weak_equivariance
    (layers : Nat → WELayer G X X Θ)
    (L : Nat)
    (g : G)
    (θ : Parameters Θ L)
    (x : X) :
    network layers L θ (g • x) =
      g • network layers L (g⁻¹ • θ) x := by
  exact (network layers L).map_smul g θ x

/-!
## Networks with equivariant activations

Here `linearLayers n` denotes the parametrized preactivation map of
layer `n`, while `activations n` denotes the activation applied after
that map.
-/

/--
Attach an equivariant activation to each parametrized layer.
-/
def layersWithActivations
    (linearLayers : Nat → WELayer G X X Θ)
    (activations : Nat → X → X)
    (activations_equivariant :
      ∀ (n : Nat) (g : G) (x : X),
        activations n (g • x) =
          g • activations n x) :
    Nat → WELayer G X X Θ :=
  fun n =>
    (linearLayers n).postActivation
      (activations n)
      (activations_equivariant n)

/--
Evaluation of an arbitrary-depth network with equivariant activations.
-/
def evalNetworkWithActivations
    (linearLayers : Nat → WELayer G X X Θ)
    (activations : Nat → X → X)
    (activations_equivariant :
      ∀ (n : Nat) (g : G) (x : X),
        activations n (g • x) =
          g • activations n x)
    (L : Nat)
    (θ : Parameters Θ L)
    (x : X) :
    X :=
  evalNetwork
    (layersWithActivations
      linearLayers
      activations
      activations_equivariant)
    L θ x

/--
A neural network with any finite number of weakly equivariant
parametrized layers and equivariant activations is weakly equivariant.
-/
theorem network_with_activations_weak_equivariance
    (linearLayers : Nat → WELayer G X X Θ)
    (activations : Nat → X → X)
    (activations_equivariant :
      ∀ (n : Nat) (g : G) (x : X),
        activations n (g • x) =
          g • activations n x)
    (L : Nat)
    (g : G)
    (θ : Parameters Θ L)
    (x : X) :
    evalNetworkWithActivations
        linearLayers
        activations
        activations_equivariant
        L θ (g • x)
      =
    g •
      evalNetworkWithActivations
        linearLayers
        activations
        activations_equivariant
        L (g⁻¹ • θ) x := by
  exact evalNetwork_weak_equivariant
    (layersWithActivations
      linearLayers
      activations
      activations_equivariant)
    L g θ x
end NeuralNetwork
