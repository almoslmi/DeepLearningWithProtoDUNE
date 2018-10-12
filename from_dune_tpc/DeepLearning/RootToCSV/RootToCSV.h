#ifndef ROOT_TO_CSV_H
#define ROOT_TO_CSV_H

using namespace std;

class RootToCSV
{
 private :

    TString InputFile;
    TString OutputDirectory;

 public :

    // With default values
    RootToCSV(const TString input = "NONE", const TString output = "NONE");

    // To set values of choice
    void SetInputFile(const TString input);
    void SetOutputDirectory(const TString output);

    // To make CSV file
    void MakeCSV();
};

#endif
