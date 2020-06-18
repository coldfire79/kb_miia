/*
A KBase module: kb-miia
*/

module kb-miia {
    typedef structure {
        string report_name;
        string report_ref;
    } ReportResults;

    /*
        This example function accepts any number of parameters and returns results in a KBaseReport
    */
    funcdef run_MIIA(mapping<string,UnspecifiedObject> params) returns (ReportResults output) authentication required;

};
