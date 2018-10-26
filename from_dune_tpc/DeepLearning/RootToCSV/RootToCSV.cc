// C++ includes
#include <iostream>
#include <fstream>
#include <string>

// ROOT includes
#include <TROOT.h>
#include <TFile.h>
#include <TTree.h>

// DeepLearning includes
#include "dune/Protodune/DeepLearning/Tools/DataStruct.h"
#include "dune/Protodune/DeepLearning/Tools/ImageInformation.h"

// Header file
#include "RootToCSV.h"

using namespace std;

RootToCSV :: RootToCSV(const TString input, const TString output)
    : InputFile(input), OutputDirectory(output)
{
}

void RootToCSV::SetInputFile(const TString input)
{
    InputFile = input;
}

void RootToCSV::SetOutputDirectory(const TString output)
{
    OutputDirectory = output;
}

///////////////////////////////////////////////////////////////////////////////////////
void RootToCSV::MakeCSV()
{

    // Select the Tree
    TFile *FileInput = new TFile(InputFile);

    std::vector<ProtoDuneDL::HitsStruct> * pHitsStruct = 0;
    TTree *TreeInput = (TTree*)FileInput->Get("DataProcessing/tHitsTree");
    TreeInput->SetBranchAddress("vHitsStruct", &pHitsStruct);

    ///////////////////////////////////////////////////////////////////////////////////////////
    unsigned int nEntries = TreeInput->GetEntries();
    cout<<endl;
    if(nEntries <= 0)
        {
            cout << "Error! There are no entries in the Tree." << nEntries << endl<<endl;
        }
    else
        {
            cout << "Total Entries in the Tree is: " << nEntries << endl<<endl;
        }

    ofstream FileOutput[6]; // 3 planes x (Feature + Label)
    TString FileName[6] = {"feature_u.csv", "feature_v.csv", "feature_w.csv", "label_u.csv", "label_v.csv", "label_w.csv"};

    for (unsigned int iFile = 0; iFile < 6; iFile++)
        {
            // Create and open the .csv file
            FileOutput[iFile].open(OutputDirectory + FileName[iFile]);

            // Write the file headers
            for (unsigned int iPixel = 0; iPixel < ProtoDuneDL::MaxTDCs * ProtoDuneDL::MaxWires; iPixel++)
                {
                    FileOutput[iFile] << "Pixel " << iPixel;
                    if(iPixel != ProtoDuneDL::MaxTDCs * ProtoDuneDL::MaxWires - 1)
                        {
                            FileOutput[iFile] << ",";
                        }
                }
            FileOutput[iFile] << endl;
        }

    for(unsigned int iEntry = 0; iEntry < nEntries; iEntry++)
        {
            TreeInput->GetEntry(iEntry);

            unsigned int nHits = pHitsStruct->size();
            //TO DO: Min hit cut can be made here
            //cout << "Total hits for entry " << iEntry + 1 << " is: " << nHits << endl;
	    cout << "Event: " << iEntry + 1 << endl;

            static double FeatureMap[ProtoDuneDL::MaxPlanes][ProtoDuneDL::MaxTDCs][ProtoDuneDL::MaxWires];
            static int LabelMap[ProtoDuneDL::MaxPlanes][ProtoDuneDL::MaxTDCs][ProtoDuneDL::MaxWires];
            for (unsigned int iPlane = 0; iPlane < ProtoDuneDL::MaxPlanes; iPlane++)
                {
                    for (unsigned int iTDC = 0; iTDC < ProtoDuneDL::MaxTDCs; iTDC++)
                        {
                            for (unsigned int iWire = 0; iWire < ProtoDuneDL::MaxWires; iWire++)
                                {

                                    FeatureMap[iPlane][iTDC][iWire] = 0.0;
                                    LabelMap[iPlane][iTDC][iWire] = 0;
                                }
                        }
                }

            for(unsigned int iHit = 0; iHit < nHits; iHit++)
                {
                    ProtoDuneDL::HitsStruct hHitsStruct = (*pHitsStruct)[iHit];
                    unsigned int plane = hHitsStruct.global_plane_index;
                    unsigned int tdc = hHitsStruct.tdc;

                    //TO DO: Flip wire for V plane?
                    unsigned int wire = hHitsStruct.global_wire_index;

                    // TO DO: Better downsampling
                    unsigned int tdcNew = ProtoDuneDL::TransformRange(tdc, 0, ProtoDuneDL::TDCsPerPlane[plane], 0, ProtoDuneDL::MaxTDCs);
                    unsigned int wireNew = ProtoDuneDL::TransformRange(wire, 0, ProtoDuneDL::GlobalWiresPerPlane[plane], 0, ProtoDuneDL::MaxWires);

                    double adc = hHitsStruct.adc;
                    if(adc > FeatureMap[plane][tdcNew][wireNew])
                        {
                            FeatureMap[plane][tdcNew][wireNew] = adc;
                            LabelMap[plane][tdcNew][wireNew] = hHitsStruct.origin;
                        }
                }

            for (unsigned int iPlane = 0; iPlane < ProtoDuneDL::MaxPlanes; iPlane++)
                {
                    int iCount = 0;
                    for (unsigned int iTDC = 0; iTDC < ProtoDuneDL::MaxTDCs; iTDC++)
                        {
                            for (unsigned int iWire = 0; iWire < ProtoDuneDL::MaxWires; iWire++)
                                {
                                    FileOutput[iPlane] << FeatureMap[iPlane][iTDC][iWire];
                                    FileOutput[iPlane + 3] << LabelMap[iPlane][iTDC][iWire];
                                    if(iCount != ProtoDuneDL::MaxTDCs * ProtoDuneDL::MaxWires - 1)
                                        {
                                            FileOutput[iPlane] << ",";
                                            FileOutput[iPlane + 3] << ",";
                                        }
                                    iCount++;
                                }
                        }
                    FileOutput[iPlane] << endl;
                    FileOutput[iPlane + 3] << endl;
                }
        }

    // Close the output file
    for (unsigned int iFile = 0; iFile < 6; iFile++)
        {
            FileOutput[iFile].close();
        }
}

int main(int argc, char* argv[])
{
    if(argc != 3)
        {
            cout << "Error! Please provide 2 arguments: input file and output directory name." << endl;
            return 1;
        }

    RootToCSV *myRootToCSV = new RootToCSV();
    myRootToCSV->SetInputFile(argv[1]);
    myRootToCSV->SetOutputDirectory(argv[2]);
    myRootToCSV->MakeCSV();

    return 0;
}
