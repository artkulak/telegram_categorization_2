#include "application.h"
#include <iostream>

void print_vec(std::vector<std::pair<float, std::string>>& vec)
{
    for (auto i = vec.begin(); i != vec.end(); ++i)
    {
        std::cout << "Label: " << i -> second << " Proba: " << i -> first << std::endl; 
    }
}

int main(int argc, char **argv)
{
    std::string line = "Дикое Поле. Историческая рандомность, халдунианская антропология, зеленый тацитизм, пост-османские наблюдения (Турция, Ближний Восток, Балканы) и другие вещи.\n\nWhite Man’s Burden wearing a turban.";
    try
    {
        LightGBM::ApplicationLightGBM app;
        app.LoadModel("../../../models/ru/");
        app.InitPredict();
        std::vector<std::pair<float, std::string>> vec = app.Predict(line);
        print_vec(vec);
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
