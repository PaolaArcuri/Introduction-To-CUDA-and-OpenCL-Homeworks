#ifndef APP_CONFIG_H
#define APP_CONFIG_H

#include <typeinfo>
#include "tinyxml2.h"
#include <string>


using namespace tinyxml2;
using namespace std;

enum APP_TYPE{VECTOR_ADD=1, MATRIX_MUL, DOT_PRODUCT };

struct AppConfig
{
    APP_TYPE app;

    int dim_A_X;
    int dim_A_Y;

    int dim_B_X;
    int dim_B_Y;

    int threads_per_block_X;
    int threads_per_block_Y;
    int threads_per_block_Z;
};

void loadAppConfig (const char* filename, AppConfig & appConfig)
{
    XMLDocument* doc = new XMLDocument();
	doc->LoadFile( filename);

	appConfig.app = static_cast<APP_TYPE>(std::stoi( doc->FirstChildElement( "app-config" )->FirstChildElement( "app")->GetText()));
   
    
    appConfig.dim_A_X = std::stoi( doc->FirstChildElement( "app-config" )->FirstChildElement( "data_size")->FirstChildElement( "A")->FirstChildElement( "x")->GetText() );
    appConfig.dim_A_Y = std::stoi( doc->FirstChildElement( "app-config" )->FirstChildElement( "data_size")->FirstChildElement( "A")->FirstChildElement( "y")->GetText() );

    appConfig.dim_B_X = std::stoi( doc->FirstChildElement( "app-config" )->FirstChildElement( "data_size")->FirstChildElement( "B")->FirstChildElement( "x")->GetText() );
    appConfig.dim_B_Y = std::stoi( doc->FirstChildElement( "app-config" )->FirstChildElement( "data_size")->FirstChildElement( "B")->FirstChildElement( "y")->GetText() );

    appConfig.threads_per_block_X = std::stoi( doc->FirstChildElement( "app-config" )->FirstChildElement( "threads-per-block")->FirstChildElement( "x")->GetText() );
    appConfig.threads_per_block_Y = std::stoi( doc->FirstChildElement( "app-config" )->FirstChildElement( "threads-per-block")->FirstChildElement( "y")->GetText() );
    appConfig.threads_per_block_Z = std::stoi( doc->FirstChildElement( "app-config" )->FirstChildElement( "threads-per-block")->FirstChildElement( "z")->GetText() );
	
 
}

void printAppConfig (AppConfig & appConfig)
{
    cout<< "App Configuration "<<endl<< "app_type: "<<appConfig.app <<"\n"<<"Dim A ("<< appConfig.dim_A_X<<", "<<appConfig.dim_A_Y<<")"<<endl <<"Dim B ("<< appConfig.dim_B_X<<", "<<appConfig.dim_B_Y<<")"<<endl;;
    cout<< "Thread configuration ("<<appConfig.threads_per_block_X<<", "<<appConfig.threads_per_block_Y<<", "<<appConfig.threads_per_block_Z<<")"<<endl;

}

#endif