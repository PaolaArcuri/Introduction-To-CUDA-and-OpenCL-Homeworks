#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <vector>
#include <cassert>

#include "AppConfiguration.h"

using namespace std;


int main(int argc, char **argv)
{
  
    assert (argc >1);
    struct AppConfig appConfig;

    try
    {
        loadAppConfig (argv[1], appConfig);

    }
    catch(const std::exception& ex)
    {
        std::cerr << "Error occurred: " << ex.what() << std::endl;
        std::cerr <<"You must specify every element of the xml"<<std::endl;
        return 0;
    }

    printAppConfig(appConfig);





    return 0;
}
