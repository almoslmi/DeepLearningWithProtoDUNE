// This is a C++ header file which must include all the necessary headers for the 
// class, struct and enum entries referenced in classes_def.xml

#include "canvas/Persistency/Common/Wrapper.h"
#include "canvas/Persistency/Common/Assns.h" 

#include "dune/Protodune/DeepLearning/Tools/DataStruct.h"

template class std::vector<ProtoDuneDL::HitsStruct>;
template class std::vector<ProtoDuneDL::AllHitsStruct>;
template class std::vector<ProtoDuneDL::PrimaryParticleStruct>;


