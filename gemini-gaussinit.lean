import Mathlib.MeasureTheory.Measure.Map
import Mathlib.MeasureTheory.Constructions.BorelSpace.Basic
import Mathlib.Analysis.InnerProductSpace.Basic
import Mathlib.Algebra.Group.Defs

/-!
# Invariance of the Parameter Probability Law under Unitary Representations

We formalize that the pushforward probability measure of an affine parameter space
is invariant under a unitary representation, given the fixed-point hypothesis
and base measure invariance.
-/

open MeasureTheory

variable {G : Type*} [Group G]
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
variable [MeasurableSpace V] [BorelSpace V]

/-- A representation ρ of a group G on a vector space V. -/
structure Representation where
  toFun : G → V →L[ℝ] V
  map_one' : toFun 1 = ContinuousLinearMap.id ℝ V
  map_mul' (g h : G) : toFun (g * h) = (toFun g).comp (toFun h)

/-- The parameter distribution defined as the pushforward of a base measure `μ`
    by the affine translation `w ↦ v0 + w`. -/
noncomputable def ParameterDistribution (v0 : V) (μ : Measure V) : Measure V :=
  Measure.map (fun w => v0 + w) μ

/-- Main Theorem: The probability law (pushforward measure) of the affine parameter
    is invariant under isomorphism/unitary group actions. -/
theorem parameter_law_invariant
    (ρ : Representation)
    (v0 : V)
    (μ : Measure V)
    (h_fixed : ∀ (g : G), ρ.toFun g v0 = v0)
    (h_measure_inv : ∀ (g : G), Measure.map (ρ.toFun g) μ = μ)
    (g : G) :
    Measure.map (ρ.toFun g) (ParameterDistribution v0 μ) =
    ParameterDistribution v0 μ := by
  unfold ParameterDistribution
  have hm_rho : Measurable (ρ.toFun g) := (ρ.toFun g).continuous.measurable
  have hm_trans : Measurable (fun w => v0 + w) := continuous_const_add v0 |>.measurable
  rw [Measure.map_map hm_rho hm_trans]
  have h_comm : (ρ.toFun g) ∘ (fun w => v0 + w) = (fun x => v0 + x) ∘ (ρ.toFun g) := by
    ext w
    dsimp
    rw [ContinuousLinearMap.map_add, h_fixed g]
  rw [h_comm, ← Measure.map_map hm_trans hm_rho, h_measure_inv g]
