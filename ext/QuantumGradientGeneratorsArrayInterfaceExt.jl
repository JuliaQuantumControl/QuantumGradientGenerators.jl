module QuantumGradientGeneratorsArrayInterfaceExt

import ArrayInterface
import QuantumGradientGenerators: GradVector
import QuantumControl.QuantumPropagators.Interfaces: supports_vector_interface

# The following methods are necessary for `check_state`, because `similar`
# can return a `GradVector`, and the results of `similar` must be a mutable
# array.

ArrayInterface.ismutable(::Type{GradVector{N,T}}) where {N,T} =
    ArrayInterface.ismutable(T) && supports_vector_interface(T)


# Direct access to the elements of a `GradVector` is possible, but should be
# discouraged

ArrayInterface.fast_scalar_indexing(::Type{<:GradVector}) = false

end
