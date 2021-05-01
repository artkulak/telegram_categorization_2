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
    std::string line1 = "Анонс! Завтра распродажа Smartavia: миллион билетов по России от 990₽\n#Разное\nhttps:\/\/ru.pirates.travel\/raznoe\/anons-zavtra-rasprodazha-smartavia-million-biletov-po-rossii-ot-990\/150199";
 
    try
    {
        LightGBM::ApplicationLightGBM app;
        app.LoadModel("../../../../resources/models/ru/");
        app.InitPredict();
        std::vector<std::pair<float, std::string>> vec = app.Predict(line);
        print_vec(vec);
        std::vector<std::pair<float, std::string>> vec1 = app.Predict(line1);
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
