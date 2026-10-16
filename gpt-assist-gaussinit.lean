import Mathlib
import Mathlib.Probability.Distributions.Gaussian.Multivariate

open MeasureTheory
open ProbabilityTheory

noncomputable section

namespace AffineGaussianInvariance

universe uG uH

variable {G : Type uG}
variable {H : Type uH}

variable [Group G]

variable [NormedAddCommGroup H]
variable [InnerProductSpace ℝ H]
variable [FiniteDimensional ℝ H]
variable [MeasurableSpace H]
variable [BorelSpace H]

/-- The coordinate space `ℝ^p`, equipped with its Euclidean structure. -/
abbrev Coord (p : ℕ) := EuclideanSpace ℝ (Fin p)

/-- The affine Gaussian parametrization `c ↦ A + L c`. -/
def affineParam {p : ℕ}
    (A : H)
    (L : Coord p →L[ℝ] H) :
    Coord p → H :=
  fun c => A + L c

/-- The distribution of `θ = A + Lc`, where `c` is standard Gaussian. -/
def affineGaussianLaw {p : ℕ}
    (A : H)
    (L : Coord p →L[ℝ] H) :
    Measure H :=
  Measure.map
    (affineParam A L)
    (stdGaussian (Coord p))

/-- The map `c ↦ A + Lc` is measurable. -/
theorem measurable_affineParam {p : ℕ}
    (A : H)
    (L : Coord p →L[ℝ] H) :
    Measurable (affineParam A L) := by
  apply Continuous.measurable
  exact continuous_const.add L.continuous
/--
If `ρ(g)` fixes `A`, and an orthogonal coordinate action `coordAct(g)`
intertwines `L` with `ρ(g)`, then the law of `A + Lc` is invariant under
`ρ(g)`.

The conclusion

    Measure.map (ρ g) (affineGaussianLaw A L)
      = affineGaussianLaw A L

is precisely the statement that

    ρ(g)(A + Lc) = A + Lc

in distribution.
-/
theorem affineGaussianLaw_invariant
    {p : ℕ}
    (ρ : G →* (H ≃ₗᵢ[ℝ] H))
    (A : H)
    (L : Coord p →L[ℝ] H)
    (coordAct : G → Coord p ≃ₗᵢ[ℝ] Coord p)
    (hA : ∀ g : G, ρ g A = A)
    (hintertwine :
      ∀ (g : G) (c : Coord p),
        ρ g (L c) = L (coordAct g c)) :
    ∀ g : G,
      Measure.map (ρ g) (affineGaussianLaw A L)
        = affineGaussianLaw A L := by
  intro g
  have hρ : Measurable (ρ g : H → H) :=
    (ρ g).continuous.measurable
  have hθ : Measurable (affineParam A L) :=
    measurable_affineParam A L
  have hcoord : Measurable (coordAct g : Coord p → Coord p) :=
    (coordAct g).continuous.measurable
  have hpointwise :
      (ρ g : H → H) ∘ affineParam A L
        = affineParam A L ∘ (coordAct g : Coord p → Coord p) := by
    funext c
    simp only [Function.comp_apply, affineParam]
    rw [map_add, hA g, hintertwine g c]
  calc
    Measure.map (ρ g) (affineGaussianLaw A L)
        =
      Measure.map
        ((ρ g : H → H) ∘ affineParam A L)
        (stdGaussian (Coord p)) := by
          rw [affineGaussianLaw]
          exact Measure.map_map hρ hθ
    _ =
      Measure.map
        (affineParam A L ∘ (coordAct g : Coord p → Coord p))
        (stdGaussian (Coord p)) := by
          rw [hpointwise]
    _ =
      Measure.map
        (affineParam A L)
        (Measure.map
          (coordAct g : Coord p → Coord p)
          (stdGaussian (Coord p))) := by
            symm
            exact Measure.map_map hθ hcoord
    _ =
      Measure.map
        (affineParam A L)
        (stdGaussian (Coord p)) := by
          rw [ProbabilityTheory.stdGaussian_map]
    _ = affineGaussianLaw A L := by
      rfl

/--
The same result, phrased as equality in distribution of the two random
variables on the canonical Gaussian probability space.
-/
theorem affineGaussian_eqInDistribution
    {p : ℕ}
    (ρ : G →* (H ≃ₗᵢ[ℝ] H))
    (A : H)
    (L : Coord p →L[ℝ] H)
    (coordAct : G → Coord p ≃ₗᵢ[ℝ] Coord p)
    (hA : ∀ g : G, ρ g A = A)
    (hintertwine :
      ∀ (g : G) (c : Coord p),
        ρ g (L c) = L (coordAct g c)) :
    ∀ g : G,
      Measure.map
          (fun c : Coord p => ρ g (A + L c))
          (stdGaussian (Coord p))
        =
      Measure.map
          (fun c : Coord p => A + L c)
          (stdGaussian (Coord p)) := by
  intro g
  have hρ : Measurable (ρ g : H → H) :=
    (ρ g).continuous.measurable
  have hparam : Measurable (affineParam A L) :=
    measurable_affineParam A L
  have hinvariant :
      Measure.map
          (ρ g : H → H)
          (Measure.map
            (affineParam A L)
            (stdGaussian (Coord p)))
        =
      Measure.map
          (affineParam A L)
          (stdGaussian (Coord p)) :=
    affineGaussianLaw_invariant
      (ρ := ρ)
      (A := A)
      (L := L)
      (coordAct := coordAct)
      hA
      hintertwine
      g
  calc
    Measure.map
        (fun c : Coord p => ρ g (A + L c))
        (stdGaussian (Coord p))
        =
      Measure.map
        ((ρ g : H → H) ∘ affineParam A L)
        (stdGaussian (Coord p)) := by
          congr 1
    _ =
      Measure.map
        (ρ g : H → H)
        (Measure.map
          (affineParam A L)
          (stdGaussian (Coord p))) := by
            symm
            exact Measure.map_map hρ hparam
    _ =
      Measure.map
        (affineParam A L)
        (stdGaussian (Coord p)) :=
          hinvariant
    _ =
      Measure.map
        (fun c : Coord p => A + L c)
        (stdGaussian (Coord p)) := by
          congr 1

end AffineGaussianInvariance
