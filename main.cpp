#include "application.h"
#include <iostream>

int main(int argc, char **argv)
{
    try
    {
        LightGBM::ApplicationLightGBM app;
        app.LoadModel("../model/");
        app.InitPredict();
        app.LoadData();
        app.Predict();
    }
    catch (const std::exception &ex)
    {
        std::cerr << "Met Exceptions:" << std::endl;
        std::cerr << ex.what() << std::endl;
    }
    catch (const std::string &ex)
    {
        std::cerr << "Met Exceptions:" << std::endl;
        std::cerr << ex << std::endl;
    }
    catch (...)
    {
        std::cerr << "Unknown Exceptions" << std::endl;
    }

    return 0;
}
