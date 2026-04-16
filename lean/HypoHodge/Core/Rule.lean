import HypoHodge.Core.Context
import Hypostructure.Core.Rule

namespace HypoHodge.Core

abbrev RuleKind := Hypostructure.Core.RuleKind

abbrev RuleKind.backend : RuleKind := Hypostructure.Core.RuleKind.backend
abbrev RuleKind.bridge : RuleKind := Hypostructure.Core.RuleKind.bridge
abbrev RuleKind.promotion : RuleKind := Hypostructure.Core.RuleKind.promotion
abbrev RuleKind.incUpgrade : RuleKind := Hypostructure.Core.RuleKind.incUpgrade

abbrev Rule := Hypostructure.Core.Rule CertTag

abbrev RuleSet := Hypostructure.Core.RuleSet CertTag

abbrev Rule.enabled (r : Rule) (Γ : Context) : Prop :=
  Hypostructure.Core.Rule.enabled r Γ

instance instDecidableEnabled (r : Rule) (Γ : Context) : Decidable (r.enabled Γ) := by
  simpa [Rule.enabled] using (Hypostructure.Core.instDecidableEnabled r Γ)

abbrev fireRule (r : Rule) (Γ : Context) : Context :=
  Hypostructure.Core.fireRule r Γ

abbrev step : RuleSet → Context → Context :=
  Hypostructure.Core.step

abbrev enabled_iff_subset := @Hypostructure.Core.enabled_iff_subset CertTag _

abbrev fireRule_eq_insert_of_enabled := @Hypostructure.Core.fireRule_eq_insert_of_enabled CertTag _

abbrev fireRule_eq_self_of_disabled := @Hypostructure.Core.fireRule_eq_self_of_disabled CertTag _

abbrev subset_fireRule := @Hypostructure.Core.subset_fireRule CertTag _

abbrev monotone_fireRule := @Hypostructure.Core.monotone_fireRule CertTag _

abbrev subset_step := @Hypostructure.Core.subset_step CertTag _

abbrev monotone_step := @Hypostructure.Core.monotone_step CertTag _

abbrev step_append := @Hypostructure.Core.step_append CertTag _

end HypoHodge.Core
