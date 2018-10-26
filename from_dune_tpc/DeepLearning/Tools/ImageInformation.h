#ifndef IMAGE_INFORMATION_H
#define IMAGE_INFORMATION_H

#include "larcore/Geometry/Geometry.h"

// DeepLearning includes
#include "dune/Protodune/DeepLearning/Tools/Constants.h"

namespace ProtoDuneDL
{
    void GetGlobalWire(geo::WireID wireID, unsigned int& globalWire, unsigned int& globalPlane)
    {
	// The beam enters from the left. TPC's 0, 3, 4, 7, 8, and 11 do not have a field.
	//      3     7    11
	//     ===== ===== =====
	//    ^ 2     6    10
	//    |
	//--> -----------------  CPA
	//
	//      1     5     9
	//    ===== ===== =====
	//      0     4     8

	globalPlane = wireID.Plane;

	// For drift in positive x direction, swap U and V
	if(wireID.Plane != 2)
	    {
		if(wireID.TPC == 0 || wireID.TPC == 1 || wireID.TPC == 4 || wireID.TPC == 5 || wireID.TPC == 8 || wireID.TPC == 9)
		    {
			globalPlane = wireID.Plane;
		    }
		else
		    {
			if(wireID.Plane == 0)
			    {
				globalPlane = 1;
			    }
			else
			    {
				globalPlane = 0;
			    }
		    }

	    }

	if(globalPlane != 1)
	    {
		globalWire = wireID.Wire + (wireID.TPC / 4) * ProtoDuneDL::WiresOffsetPerTPC[wireID.Plane];
	    }
	else
	    {
		globalWire = wireID.Wire + ((11 - wireID.TPC) / 4) * ProtoDuneDL::WiresOffsetPerTPC[wireID.Plane];
	    }
    }

    unsigned int FlipGlobalWire(unsigned int wire, unsigned int plane)
    {
	return ProtoDuneDL::GlobalWiresPerPlane[plane] - wire - 1;
    }

    unsigned int TransformRange(unsigned int x, unsigned int xMin, unsigned int xMax, unsigned int newMin, unsigned int newMax)
    {
	return int((double(newMax - newMin) * double(x - xMin) / double(xMax - xMin)) + double(newMin));
    }


}//namespace ProtoDuneDL
#endif
