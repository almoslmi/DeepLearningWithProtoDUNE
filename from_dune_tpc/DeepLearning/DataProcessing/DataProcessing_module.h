////////////////////////////////////////////////////////////////////////
// Class:        DataProcessing
// Module Type:  analyzer
// File:         DataProcessing_module.h
// Author:       Arbin Timilsina, arbint@bnl.gov
////////////////////////////////////////////////////////////////////////

#ifndef DataProcessing_Module_H
#define DataProcessing_Module_H

// Framework includes
#include "art/Framework/Core/EDAnalyzer.h"

// Root includes
#include "TTree.h"
#include "TH2.h"

// DeepLearning includes
#include "dune/Protodune/DeepLearning/Tools/DataStruct.h"

namespace ProtoDuneDL
{
    class DataProcessing : public art::EDAnalyzer
	{
	public:
	    /////////////////////////////////////////////////////////
	    // Constructor and destructor
	    explicit DataProcessing(fhicl::ParameterSet const& pset);
	    virtual ~DataProcessing();
	    /////////////////////////////////////////////////////////

	    /////////////////////////////////////////////////////////
	    //Called once per event
	    void analyze(const art::Event& evt);
	    /////////////////////////////////////////////////////////

	    /////////////////////////////////////////////////////////
	    // Called once at the beginning of the job
	    void beginJob();

	    // Called once at the end of the job
	    void endJob();

	    // Called in to read .fcl file parameters
	    void reconfigure(fhicl::ParameterSet const& pset);
	    /////////////////////////////////////////////////////////

	private:

	    /////////////////////////////////////////////////////////
	    std::ofstream logFile;
	    unsigned int nEvents;
	    /////////////////////////////////////////////////////////

	    /////////////////////////////////////////////////////////
	    //  Module lablel for input hits
	    std::string fHitsModuleLabel;

	    // To debug
	    bool fDebug;
	    /////////////////////////////////////////////////////////

	    /////////////////////////////////////////////////////////
	    //Geomerty
	    art::ServiceHandle<geo::Geometry> geom;

	    //BackTracker
	    art::ServiceHandle<cheat::BackTrackerService> backTracker;

	    //ParticleInventory
	    art::ServiceHandle<cheat::ParticleInventoryService> particleInventory;
	    /////////////////////////////////////////////////////////

	    /////////////////////////////////////////////////////////
	    TTree* tHitsTree;
	    std::vector<ProtoDuneDL::HitsStruct> vHitsStruct;
	    std::vector<ProtoDuneDL::HitsStruct> * pHitsStruct = &vHitsStruct;

	    TTree* tAllHitsTree;
	    std::vector<ProtoDuneDL::AllHitsStruct> vAllHitsStruct;
	    std::vector<ProtoDuneDL::AllHitsStruct> * pAllHitsStruct = &vAllHitsStruct;

	    TTree* tPrimaryParticleTree;
	    std::vector<ProtoDuneDL::PrimaryParticleStruct> vPrimaryParticleStruct;
	    std::vector<ProtoDuneDL::PrimaryParticleStruct> * pPrimaryParticleStruct = &vPrimaryParticleStruct;
	    /////////////////////////////////////////////////////////

	}; // class DataProcessing
}// namespace ProtoDuneDL

DEFINE_ART_MODULE(ProtoDuneDL::DataProcessing)

#endif // DataProcessing_Module

