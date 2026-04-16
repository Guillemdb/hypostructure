import HypoHodge.Core.CertTag

namespace Hypostructure.Instances.Hodge

abbrev CertTag := HypoHodge.Core.CertTag

abbrev CertTag.isPending : CertTag → Bool :=
  HypoHodge.Core.CertTag.isInc

end Hypostructure.Instances.Hodge
