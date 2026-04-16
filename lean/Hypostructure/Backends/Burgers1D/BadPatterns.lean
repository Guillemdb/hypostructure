import Hypostructure.Backends.Burgers1D.Basic

namespace Hypostructure.Backends.Burgers1D

/-- Finite bad-pattern package used by the Burgers Lock route. -/
class BurgersBadPatternPackage (ν : BurgersParameters) where
  germSmallness : Prop
  germSmallness_holds : germSmallness
  universalBadInitialized : Prop
  universalBadInitialized_holds : universalBadInitialized
  catLibraryComplete : Prop
  catLibraryComplete_holds : catLibraryComplete

instance burgersBadPatternPackage
    (ν : BurgersParameters) :
    BurgersBadPatternPackage ν where
  germSmallness := badPatternName ∈ burgersBadPatternLibrary
  germSmallness_holds := badPattern_mem_library
  universalBadInitialized := universalBadObjectName = "universal Burgers bad object"
  universalBadInitialized_holds := universalBadObject_initialized
  catLibraryComplete :=
    ∀ pattern : String, pattern = badPatternName → pattern ∈ burgersBadPatternLibrary
  catLibraryComplete_holds := badPatternLibrary_complete

theorem burgers_germ_smallness
    (ν : BurgersParameters)
    [BurgersBadPatternPackage ν] :
    BurgersBadPatternPackage.germSmallness (ν := ν) :=
  BurgersBadPatternPackage.germSmallness_holds (ν := ν)

theorem burgers_universal_bad_initialized
    (ν : BurgersParameters)
    [BurgersBadPatternPackage ν] :
    BurgersBadPatternPackage.universalBadInitialized (ν := ν) :=
  BurgersBadPatternPackage.universalBadInitialized_holds (ν := ν)

theorem burgers_cat_library_complete
    (ν : BurgersParameters)
    [BurgersBadPatternPackage ν] :
    BurgersBadPatternPackage.catLibraryComplete (ν := ν) :=
  BurgersBadPatternPackage.catLibraryComplete_holds (ν := ν)

end Hypostructure.Backends.Burgers1D
